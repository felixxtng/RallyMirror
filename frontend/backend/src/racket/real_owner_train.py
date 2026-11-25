"""Shared learning-check, evaluation and checkpoint helpers for v006."""
import json
import math
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .mediapipe_bundle import BODY_NAMES,sha,to_original,write_json
from .mediapipe_model import decode
from .owner_model import SelectedPlayerFusion
from .legacy_validation import evaluate as legacy_evaluate
from .real_owner_data import FORMAT,SIDES,RealPairs,flatten_queries,real_ownership_loss,scene_bounds,summarize_real


def warm_model(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format') not in ('racket_owner_v002',FORMAT) or saved.get('body_names')!=BODY_NAMES:
        raise ValueError('Expected selected v002/real-ownership checkpoint')
    model=SelectedPlayerFusion().to(device);model.load_state_dict(saved['state_dict']);return model


def checkpoint(model,m):
    return dict(format=FORMAT,state_dict=model.state_dict(),body_names=BODY_NAMES,image_size=m['image_size'],
        output='head_center',context='common union of expanded player body squares',coordinate_system='scene_crop_normalized_xy',
        context_padding=m.get('context_padding',0.),run_version=m.get('run_version','v004'),
        mediapipe_finetuned=False,initial_checkpoint_sha256=m['initial_checkpoint_sha256'],
        near_far_are_selected_queries_not_athlete_classes=True)




@torch.inference_mode()
def evaluate(model,dataset,device,batch=4,ablate=False):
    model.eval();rows=[]
    for images,poses,boxes,targets,indices in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
        image,pose,box,_=flatten_queries(images,poses,boxes,targets)
        if ablate:pose.zero_();box.zero_()
        point,mass=decode(model(image.to(device),pose.to(device),box.to(device)))
        point=point.cpu().reshape(-1,2,2).tolist();mass=mass.cpu().reshape(-1,2).tolist()
        for j,index in enumerate(indices.tolist()):
            r=dataset.records[index];w,h=r['original_size']
            for k,side in enumerate(SIDES):
                player=r['players'][side];opponent=r['players'][SIDES[1-k]]
                xy=to_original(point[j][k],r['crop'])
                valid=player['valid_target'] and all(math.isfinite(v) for v in xy) and 0<=xy[0]<w and 0<=xy[1]<h
                rows.append(dict(id=r['id']+':'+side,pair_id=r['id'],clip=r['clip'],frame=r['frame'],source_group=r['source_group'],side=side,
                    point_original=xy if valid else None,target_original=player['target_original'],opponent_target_original=opponent['target_original'],
                    error_original_pixels=math.dist(xy,player['target_original']) if valid else None,
                    opponent_error_original_pixels=math.dist(xy,opponent['target_original']) if valid else None,
                    head_separation_original_pixels=r['head_separation_original_pixels'],body_box_iou=r['body_box_iou'],
                    concentration=mass[j][k],pose_status=player['pose_status'],reason='ok' if valid else 'outside_context_or_video'))
    report=summarize_real(rows)
    report['overlap_subset']=summarize_real([r for r in rows if r['body_box_iou']>=.2])
    return report,rows


def legacy(model,bundle,m,device,batch):
    return legacy_evaluate(model, bundle, m['legacy_validation'], device, batch*2)


def gate_records(records,count=4):
    eligible=[r for r in records if r['split']=='train' and r['train_eligible'] and r['head_separation_original_pixels']>40]
    chosen=[]
    for group in sorted({r['source_group'] for r in eligible}):
        choices=[r for r in eligible if r['source_group']==group]
        chosen.append(max(choices,key=lambda r:r['head_separation_original_pixels']/r['crop'][2]))
        if len(chosen)==count:break
    if len(chosen)<2:raise ValueError('Need at least two real training pairs for the learning check')
    return chosen


def fit(model,training,validation,m,output,device,epochs,batch,sanity=False,patience=15):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    encoder=[];head=[]
    for name,p in model.named_parameters():(encoder if name.startswith('encoder.') else head).append(p)
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=.0005 if sanity else .00005),
        dict(params=head,lr=.0005 if sanity else .0002)],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000005)
    best=None;best_epoch=0;stale=0;history=[]
    loader=DataLoader(training,batch_size=batch,shuffle=True,num_workers=0)
    for epoch in range(1,epochs+1):
        start=time.monotonic();model.train();seen=0;loss_sum=0.;rank_sum=0.
        for step,(images,poses,boxes,targets,indices) in enumerate(loader,1):
            image,pose,box,target=flatten_queries(images,poses,boxes,targets)
            opposite=targets.flip(1).flatten(0,1).to(device)
            separated=torch.tensor([training.records[i]['head_separation_original_pixels']>40 for i in indices.tolist()],device=device).repeat_interleave(2)
            optimizer.zero_grad(set_to_none=True)
            logits=model(image.to(device),pose.to(device),box.to(device))
            loss,ce,ranking=real_ownership_loss(logits,target.to(device),opposite,separated)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite real ownership loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            seen+=len(images);loss_sum+=loss.item()*len(images);rank_sum+=ranking.item()*len(images)
            if step==1 or step%20==0 or step==len(loader):
                print(f'{"REAL OWNERSHIP CHECK" if sanity else "TRAIN"} epoch {epoch}/{epochs} batch {step}/{len(loader)} loss={loss.item():.4f} ownership_loss={ranking.item():.4f}',flush=True)
        scheduler.step();metrics,pred=evaluate(model,validation,device,batch)
        mean=metrics['mean_error_detected_only'];key=(metrics['pck20_including_misses'],-metrics['wrong_racket_within20_queries'],
            metrics['pck50_including_misses'],-(mean if mean is not None else float('inf')))
        if m.get('run_version')=='v005' and not sanity:
            key=(min(metrics['per_side'][s]['pck20_including_misses'] for s in SIDES),)+key
        history.append(dict(epoch=epoch,seconds=time.monotonic()-start,loss=loss_sum/seen,ownership_loss=rank_sum/seen,**metrics))
        write_json(output/'history.json',history)
        print(f'EPOCH {epoch}: PCK20={metrics["pck20_including_misses"]:.3f} near={metrics["per_side"]["near_side"]["pck20_including_misses"]:.3f} far={metrics["per_side"]["far_side"]["pck20_including_misses"]:.3f} wrong_racket={metrics["wrong_racket_within20_queries"]}/{metrics["ownership_eligible_queries"]}; {time.monotonic()-start:.1f}s',flush=True)
        if best is None or key>best:
            best=key;best_epoch=epoch;stale=0
            torch.save(checkpoint(model,m),output/'best.pt');write_json(output/'best_metrics.json',metrics);write_json(output/'best_predictions.json',pred)
        else:stale+=1
        if sanity and metrics['pck20_including_misses']>=.95 and metrics['wrong_racket_within20_queries']==0:break
        if not sanity and stale>=patience:print(f'Early stop: {patience} epochs without improvement; best epoch {best_epoch}',flush=True);break
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'])
    return json.loads((output/'best_metrics.json').read_text()),best_epoch


def previews(bundle,m,old,new,output):
    output=Path(output);(output/'previews').mkdir(exist_ok=True)
    before={r['id']:r for r in old};after={r['id']:r for r in new}
    version=m.get('run_version','v004');baseline=m.get('baseline_label','v002')
    writer=cv2.VideoWriter(str(output/f'previews/{baseline}_vs_{version}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8.,(1280,840))
    if not writer.isOpened():raise RuntimeError('Cannot write ownership preview')
    worst=[]
    try:
        for r in m['frames']:
            if r['split']!='val':continue
            original=cv2.imread(str(Path(bundle)/r['full_image']))
            if original is None:raise ValueError('Missing validation demo frame')
            panels=[];sx,sy=640/r['original_size'][0],360/r['original_size'][1]
            at=lambda p:(round(p[0]*sx),round(p[1]*sy))
            for side in SIDES:
                for name,lookup in [(baseline+' baseline (same inputs)',before),('real ownership '+version,after)]:
                    row=lookup[r['id']+':'+side];view=cv2.resize(original,(640,360))
                    box=r['players'][side]['owner_box'];cv2.rectangle(view,at(box[:2]),at(box[2:]),(0,255,255),1)
                    cv2.circle(view,at(row['target_original']),6,(0,255,0),2)
                    cv2.circle(view,at(row['opponent_target_original']),4,(255,255,0),1)
                    if row['point_original']:cv2.drawMarker(view,at(row['point_original']),(255,0,255),cv2.MARKER_CROSS,16,2)
                    header=np.zeros((40,640,3),np.uint8);err=row['error_original_pixels']
                    label=f'{name} | {side} | {r["clip"]} f{r["frame"]} | '+(f'{err:.1f}px' if err is not None else 'MISS')
                    cv2.putText(header,label,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
                    panels.append(np.vstack([header,view]))
            display=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:]),np.zeros((40,1280,3),np.uint8)])
            cv2.putText(display,'Same verified validation frames/context | Green: selected truth | Cyan: other truth | Magenta: prediction | Sampled montage, not continuous',
                (8,824),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            writer.write(display);writer.write(display)
            errors=[after[r['id']+':'+s]['error_original_pixels'] for s in SIDES]
            error=max(e if e is not None else float('inf') for e in errors)
            worst.append((error,r['id'],display))
            worst=sorted(worst,key=lambda item:item[0],reverse=True)[:8]
    finally:writer.release()
    for rank,(_,identity,frame) in enumerate(sorted(worst,key=lambda r:r[0],reverse=True)[:8],1):
        cv2.imwrite(str(output/'previews'/f'worst_{rank:02}_{identity}.jpg'),frame)


def archive_results(output,selected_name='racket_center_real_owner_v004.pt'):
    output=Path(output);destination=output.with_name(output.name+'_results.zip')
    selected=output/selected_name
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix=='.pt' and p!=selected and (selected.exists() or p!=output/'ownership_sanity/best.pt'):continue
            z.write(p,p.relative_to(output).as_posix())
    print('DOWNLOAD RESULTS:',destination,flush=True)
