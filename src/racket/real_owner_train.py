"""Fine-tune a returned checkpoint on real two-player ownership, with swap metrics."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .mediapipe_bundle import BODY_NAMES,sha,to_original,write_json
from .mediapipe_model import decode
from .owner_model import SelectedPlayerFusion
from .owner_data import OwnerSamples
from .owner_train import evaluate as legacy_evaluate
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


def verify(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    if m['version']!=FORMAT or m['body_names']!=BODY_NAMES:raise ValueError('Use the real ownership v004 bundle')
    for path,digest in m['file_hashes'].items():
        if not (bundle/path).is_file() or sha(bundle/path)!=digest:raise ValueError('Missing/changed upload: '+path)
    if sha(bundle/m.get('initial_checkpoint','models/baseline_v002.pt'))!=m['initial_checkpoint_sha256']:raise ValueError('Warm checkpoint provenance mismatch')
    groups={s:{r['source_group'] for r in m['frames'] if r['split']==s} for s in ['train','val']}
    if not all(groups.values()) or groups['train']&groups['val']:raise ValueError('Real pair match leakage')
    if len({r['id'] for r in m['frames']})!=len(m['frames']):raise ValueError('Duplicate frame ID')
    for r in m['frames']:
        if r['split'] not in ('train','val') or set(r['players'])!=set(SIDES):raise ValueError('Invalid split/player slots')
        if scene_bounds([r['players'][s]['owner_box'] for s in SIDES],m.get('context_padding',0.))!=r['crop']:raise ValueError('Context must depend on body boxes only')
        for p in r['players'].values():
            if len(p['pose'])!=17 or np.asarray(p['pose']).shape!=(17,3):raise ValueError('Wrong body feature layout')
            if not np.all(np.isfinite(p['pose'])):raise ValueError('Nonfinite pose')
            if not np.allclose(to_original(p['target'],r['crop']),p['target_original'],atol=1e-6):raise ValueError('Target transform mismatch')
    legacy_groups={r['source_group'] for r in m['legacy_validation']}
    if legacy_groups&groups['train']:raise ValueError('Legacy validation leaked into real training')
    if m.get('run_version')=='v005':
        val_ids={r['id'] for r in m['frames'] if r['split']=='val'}
        fixed=set(m['fixed_original_paired_validation_ids'])
        if len(fixed)!=77 or not fixed<=val_ids:raise ValueError('Frozen original77 validation cohort changed')
        if any(not p['valid_target'] for r in m['frames'] for p in r['players'].values()):raise ValueError('Target outside padded training context')
        refs=m['reference_v004_validation']
        if {r['id'] for r in refs}!=fixed or any(r['split']!='val' or r['source_group'] in groups['train'] for r in refs):
            raise ValueError('Native v004 reference cohort mismatch/leakage')
        for r in refs:
            if scene_bounds([r['players'][s]['owner_box'] for s in SIDES])!=r['crop']:raise ValueError('Native v004 crop changed')
    return m


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
    records=m['legacy_validation'];usable=[r for r in records if r['valid_input']]
    return legacy_evaluate(model,OwnerSamples(bundle,usable),device,batch*2,all_records=records)


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


def run(bundle,output,device='cuda',epochs=50,batch=4,sanity_epochs=150,verify_only=False,patience=15):
    bundle,output=Path(bundle).resolve(),Path(output).resolve();m=verify(bundle)
    train=[r for r in m['frames'] if r['split']=='train' and r['train_eligible']];val=[r for r in m['frames'] if r['split']=='val']
    print(f'VERIFIED REAL OWNERSHIP: {len(train)} training frames / {2*len(train)} selected-racket targets; {len(val)} validation frames / {2*len(val)} targets; {len(m["legacy_validation"])} legacy queries',flush=True)
    if verify_only:return True
    if output.exists():raise FileExistsError('Use a fresh output directory; preserve prior results')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: use a GPU PyTorch pod')
    torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.set_num_threads(4);cv2.setNumThreads(1);device=torch.device(device)
    output.mkdir(parents=True);warm=bundle/m.get('initial_checkpoint','models/baseline_v002.pt')
    version=m.get('run_version','v004');selected_name=f'racket_center_real_owner_{version}.pt'
    write_json(output/'run_config.json',dict(format=FORMAT,manifest_sha256=sha(bundle/'manifest.json'),epochs=epochs,
        batch_frames=batch,selected_queries_per_batch=batch*2,real_pairs_only=True,synthetic_training=False,
        patience=patience,context_padding=m.get('context_padding',0.),run_version=version,
        ownership_loss_weight=.25,checkpoint_selection=('minimum near/far PCK20, ' if version=='v005' else '')+'paired validation PCK20, fewer wrong-racket selections, PCK50, mean error',
        final_test_evaluated=False,legacy_scope='Original578 inputs retained separately; paired/common-context scores are a different cohort'))
    try:
        print('DEVICE',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
        write_json(output/'status.json',dict(status='baseline_evaluation',training_frames=len(train),validation_frames=len(val),full_training_started=False))
        validation=RealPairs(bundle,val);baseline=warm_model(warm,device)
        base_metrics,base_pred=evaluate(baseline,validation,device,batch)
        old_metrics,old_pred=legacy(baseline,bundle,m,device,batch)
        write_json(output/'baseline_paired_metrics.json',base_metrics);write_json(output/'baseline_paired_predictions.json',base_pred)
        write_json(output/'baseline_legacy578_metrics.json',old_metrics)
        write_json(output/'baseline_legacy578_predictions.json',old_pred)
        if m.get('reference_v004_validation'):
            native_metrics,native_pred=evaluate(baseline,RealPairs(bundle,m['reference_v004_validation']),device,batch)
            write_json(output/'baseline_v004_native77_metrics.json',native_metrics)
            write_json(output/'baseline_v004_native77_predictions.json',native_pred)
            expected=m['reference_v004_expected_metrics']
            if any(native_metrics[k]!=expected[k] for k in ['within_20px','within_50px','missing','wrong_racket_within20_queries']):
                raise RuntimeError('Native v004 baseline does not reproduce original counts; download diagnostics')
        gate=RealPairs(bundle,gate_records(train),size=256)
        print('REAL OWNERSHIP CHECK: four TRAIN images, each queried for both players; this is not held-out accuracy',flush=True)
        write_json(output/'status.json',dict(status='training_only_learning_check',full_training_started=False))
        gate_metrics,_=fit(baseline,gate,gate,m,output/'ownership_sanity',device,sanity_epochs,min(batch,4),True)
        passed=gate_metrics['pck20_including_misses']>=.95 and gate_metrics['wrong_racket_within20_queries']==0
        write_json(output/'ownership_gate.json',dict(passed=passed,metrics=gate_metrics,frame_ids=[r['id'] for r in gate.records],scope='training-only real ownership/geometry check'))
        if not passed:
            write_json(output/'status.json',dict(status='ownership_gate_failed_full_training_not_started'))
            print('Learning check failed; download diagnostics. Full training NOT started.',flush=True);return False
        del baseline
        if device.type=='cuda':torch.cuda.empty_cache()
        torch.manual_seed(42);model=warm_model(warm,device)
        print('FULL TRAINING STARTING: original '+m.get('baseline_label','v002')+' weights reloaded; gate-overfit weights discarded',flush=True)
        write_json(output/'status.json',dict(status='full_training_running',training_frames=len(train),validation_frames=len(val),full_training_started=True))
        metrics,epoch=fit(model,RealPairs(bundle,train,training=True),validation,m,output/'full',device,epochs,batch,patience=patience)
        metrics,pred=evaluate(model,validation,device,batch)
        ablated,_=evaluate(model,validation,device,batch,ablate=True)
        legacy_metrics,legacy_pred=legacy(model,bundle,m,device,batch)
        write_json(output/'paired_metrics.json',metrics);write_json(output/'paired_predictions.json',pred)
        write_json(output/'conditioning_zeroed_metrics.json',ablated)
        write_json(output/'legacy578_metrics.json',legacy_metrics);write_json(output/'legacy578_predictions.json',legacy_pred)
        write_json(output/'comparison.json',dict(baseline_label=m.get('baseline_label','v002'),baseline_same_paired_inputs=base_metrics,real_ownership=metrics,
            baseline_legacy578=old_metrics,real_ownership_legacy578=legacy_metrics,
            note='Paired cohort has both rackets labelled and verified boxes/common scene context. Do not compare its percentage directly with old578 near-side percentage.'))
        if version=='v005':
            fixed=set(m['fixed_original_paired_validation_ids'])
            for name,ids in [('frozen77',fixed),('additional367',{r['id'] for r in val}-fixed)]:
                write_json(output/f'{name}_metrics.json',summarize_real([r for r in pred if r['pair_id'] in ids]))
                write_json(output/f'baseline_{name}_metrics.json',summarize_real([r for r in base_pred if r['pair_id'] in ids]))
            goal=dict(rule='PCK20 strictly > 0.90 on each side separately; misses count as failures',
                near=metrics['per_side']['near_side']['pck20_including_misses'],far=metrics['per_side']['far_side']['pck20_including_misses'])
            goal['passed']=goal['near']>.90 and goal['far']>.90
            goal['scope']='Expanded444 verified located paired validation frames; not final test or full-video recall'
            write_json(output/'acceptance.json',goal)
            print('ACCEPTANCE',json.dumps(goal),flush=True)
        torch.save(checkpoint(model,m),output/selected_name)
        previews(bundle,m,base_pred,pred,output)
        write_json(output/'status.json',dict(status='complete',best_epoch=epoch,final_test_evaluated=False,
            far_side_evaluated=True,scope=f'{len(val)} reviewed paired validation frames, not all far-side video frames'))
        print('FINAL PAIRED METRICS',json.dumps(metrics),flush=True)
        print('LEGACY578 METRICS',json.dumps(legacy_metrics),flush=True)
        return True
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error)));raise
    finally:archive_results(output,selected_name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('.'));p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=4,help='Number of real frames; model evaluates twice as many selected-player queries')
    p.add_argument('--sanity-epochs',type=int,default=150);p.add_argument('--patience',type=int,default=20)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if min(a.epochs,a.batch_size,a.sanity_epochs,a.patience)<1:p.error('Epochs/batch/patience must be positive')
    output=a.output or Path('racket_ownership_run_'+json.loads((a.bundle/'manifest.json').read_text()).get('run_version','v004'))
    if run(a.bundle,output,a.device,a.epochs,a.batch_size,a.sanity_epochs,a.verify_only,a.patience) is False:raise SystemExit(2)


if __name__=='__main__':main()
