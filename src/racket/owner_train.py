"""v002 selected-player conditioning trial; no new manual labels required."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .mediapipe_bundle import BODY_NAMES,sha,to_original,write_json
from .mediapipe_model import decode,spatial_loss
from .mediapipe_train import summarize,draw_body,connect_racket,package_results
from .owner_data import OwnerSamples,pair_only_dataset,normalized_box
from .owner_model import FORMAT,SelectedPlayerFusion
from .player_constraint import decode_player


def verify(bundle):
    bundle=Path(bundle);m=json.loads((bundle/'manifest.json').read_text())
    if m['version']!=FORMAT or m['body_names']!=BODY_NAMES:raise ValueError('Use the v002 ownership bundle')
    for name,digest in m['file_hashes'].items():
        if not (bundle/name).is_file() or sha(bundle/name)!=digest:raise ValueError('Missing or changed input: '+name)
    samples=m['samples'];lookup={r['id']:r for r in samples}
    if len(lookup)!=len(samples):raise ValueError('Duplicate sample IDs')
    groups={split:{r['source_group'] for r in samples if r['split']==split} for split in ['train','val']}
    if not groups['train'] or not groups['val'] or groups['train']&groups['val']:raise ValueError('Match split violation')
    for pair in m['ownership_pairs']:
        if pair['a']==pair['b']:raise ValueError('Ownership pair duplicates one sample')
        if any(lookup[pair[k]]['split']!='train' or not lookup[pair[k]]['train_eligible'] for k in ['a','b']):
            raise ValueError('Non-training data in ownership pairs')
    return m


@torch.inference_mode()
def evaluate(model,dataset,device,batch=8,policy='none',ablate=False,all_records=None):
    model.eval();predictions=[]
    for images,pose,box,_,indices in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
        images,pose,box=images.to(device),pose.to(device),box.to(device)
        if ablate:pose.zero_();box.zero_()
        logits=model(images,pose,box)
        if policy=='none':point,mass=decode(logits);valid=torch.ones(len(images),dtype=torch.bool)
        else:
            # Unit-square geometry in the actual model input; works for both
            # natural crops and synthetic sanity images without GT-based limits.
            point,mass,valid=decode_player(logits,box,[[0,0,1]]*len(images),pose,policy)
        for xy,score,ok,index in zip(point.cpu().tolist(),mass.cpu().tolist(),valid.cpu().tolist(),indices.tolist()):
            record,transform=dataset.reference(index)
            original_unit=[(xy[k]-transform['offset'][k])/transform['scale'] for k in range(2)]
            original=to_original(original_unit,record['crop']);w,h=record['original_size']
            ok=ok and all(math.isfinite(v) for v in original) and 0<=original[0]<w and 0<=original[1]<h
            error=math.dist(original,record['target_original']) if ok else None
            example=dataset.examples[index]
            identity=record['id'] if example['kind']=='natural' else example['pair']['id']+f"_select{example['selected']}"
            predictions.append(dict(id=identity,clip=record['clip'],frame=record['frame'],source_group=record['source_group'],
                point_original=original if ok else None,target_original=record['target_original'],error_original_pixels=error,
                concentration=score,kind=example['kind']))
    if all_records is not None:
        mapped={r['id']:r for r in predictions}
        predictions=[mapped.get(r['id'],dict(id=r['id'],clip=r['clip'],frame=r['frame'],source_group=r['source_group'],
                     point_original=None,target_original=r['target_original'],error_original_pixels=None,kind='natural')) for r in all_records]
    metrics=summarize(predictions)
    for key in ['source_group','clip']:
        groups=defaultdict(list)
        for row in predictions:groups[row[key]].append(row)
        metrics['per_'+key]={name:summarize(rows) for name,rows in groups.items()}
    return metrics,predictions


def checkpoint(model,manifest):
    return dict(format=FORMAT,state_dict=model.state_dict(),body_names=BODY_NAMES,image_size=manifest['image_size'],
        output='head_center',coordinate_system='crop_normalized_xy',box_input='selected_body_box_normalized_to_crop',
        conditioning='body_box_mask_and_body_wrist_maps_early_and_late',mediapipe_finetuned=False,
        far_side_validated=False,annotation_target='selected_player_trained_on_near_side',
        init_checkpoint_sha256=manifest['initial_checkpoint_sha256'])


def fit(model,training,validation,bundle,manifest,output,device,epochs,batch,sanity=False,val_records=None):
    output.mkdir(parents=True,exist_ok=True)
    old=[];new=[]
    for name,param in model.named_parameters():
        (new if name.startswith('condition_') or name=='owner_gain' else old).append(param)
    optimizer=torch.optim.AdamW([dict(params=old,lr=.0001 if sanity else .00005),dict(params=new,lr=.001 if sanity else .0005)],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.000005)
    loader=DataLoader(training,batch_size=batch,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    best=(-1.,-1.,-float('inf'));best_epoch=0;stale=0;history=[]
    for epoch in range(1,epochs+1):
        start=time.monotonic();model.train();loss_sum=0.;seen=0
        for step,(image,pose,box,target,_) in enumerate(loader,1):
            optimizer.zero_grad(set_to_none=True)
            logits=model(image.to(device),pose.to(device),box.to(device))
            loss=spatial_loss(logits,target.to(device))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite ownership loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=loss.item()*len(image);seen+=len(image)
            if step==1 or step%50==0 or step==len(loader):
                print(f"{'OWNERSHIP SANITY' if sanity else 'TRAIN'} epoch {epoch}/{epochs} batch {step}/{len(loader)} loss={loss.item():.4f}",flush=True)
        scheduler.step()
        metrics,pred=evaluate(model,validation,device,batch,all_records=val_records)
        error=metrics['mean_error_detected_only'];key=(metrics['pck20_including_misses'],metrics['pck50_including_misses'],-(error if error is not None else float('inf')))
        history.append(dict(epoch=epoch,loss=loss_sum/seen,seconds=time.monotonic()-start,**metrics))
        write_json(output/'history.json',history)
        print(f"EPOCH {epoch}: PCK20={metrics['pck20_including_misses']:.3f} PCK50={metrics['pck50_including_misses']:.3f} missing={metrics['missing']} mean_px={error} {time.monotonic()-start:.1f}s",flush=True)
        if key>best:
            best=key;best_epoch=epoch;stale=0
            torch.save(checkpoint(model,manifest),output/'best.pt');write_json(output/'best_metrics.json',metrics);write_json(output/'best_predictions.json',pred)
        else:stale+=1
        torch.save(checkpoint(model,manifest),output/'last.pt')
        if sanity and metrics['pck20_including_misses']>=.95:break
        if not sanity and stale>=15:
            print(f'Early stop after15 epochs without improvement; best epoch {best_epoch}',flush=True);break
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'])
    return json.loads((output/'best_metrics.json').read_text()),best_epoch


def previews(bundle,manifest,predictions,output):
    output=Path(output);(output/'previews').mkdir(exist_ok=True)
    lookup={r['id']:r for r in manifest['samples']};pred={r['id']:r for r in predictions}
    ranked=sorted(predictions,key=lambda r:r['error_original_pixels'] if r['error_original_pixels'] is not None else float('inf'))
    for label,selected in [('best',ranked[:16]),('worst',list(reversed(ranked[-16:])) )]:
        tiles=[]
        for row in selected:
            r=lookup[row['id']]
            if not r['image']:continue
            image=cv2.imread(str(bundle/r['image']));size=image.shape[0]
            b=normalized_box(r)*size
            cv2.rectangle(image,tuple(np.rint(b[:2]).astype(int)),tuple(np.rint(b[2:]).astype(int)),(0,255,255),2)
            cv2.circle(image,tuple(round(v*size) for v in r['target']),6,(0,255,0),2)
            if row['point_original']:
                point=[(row['point_original'][k]-r['crop'][k])/r['crop'][2] for k in range(2)]
                cv2.drawMarker(image,tuple(round(v*size) for v in point),(255,0,255),cv2.MARKER_CROSS,18,2)
            tile=cv2.resize(image,(320,320));cv2.rectangle(tile,(0,0),(320,32),(0,0,0),-1)
            error=row['error_original_pixels'];title=row['id']+(f' {error:.0f}px' if error is not None else ' MISS')
            cv2.putText(tile,title,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);tiles.append(tile)
        if tiles:
            tiles += [np.zeros_like(tiles[0]) for _ in range((-len(tiles))%4)]
            cv2.imwrite(str(output/'previews'/f'{label}_16.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    writer=None;combined=[]
    try:
        for demo in manifest['demos']:
            r=lookup[demo['sample_id']];p=pred[r['id']]
            image=cv2.imread(str(bundle/demo['full_image']));scale=image.shape[1]/r['original_size'][0]
            draw_body(image,r['body_keypoints'],scale);connect_racket(image,r['body_keypoints'],p['point_original'],scale)
            cv2.circle(image,tuple(round(v*scale) for v in r['target_original']),6,(0,255,0),2)
            box=np.rint(np.array(r['owner_box'])*scale).astype(int);cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
            cv2.rectangle(image,(0,0),(image.shape[1],60),(0,0,0),-1)
            cv2.putText(image,f'v002 {r["id"]} | selected player only',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            cv2.putText(image,'Yellow: selected body | Green: label | Magenta: predicted racket',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            if writer is None:
                writer=cv2.VideoWriter(str(output/'previews/near_side_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6.,(image.shape[1],image.shape[0]))
                if not writer.isOpened():raise RuntimeError('Cannot create preview video')
            writer.write(image)
            combined.append(dict(clip=r['clip'],frame=r['frame'],coordinate_system='original_pixels',
                bbox=r['owner_box'],keypoints=r['body_keypoints'],racket=dict(name='head_center',point=p['point_original']),
                ownership='selected near-side player; anonymous until tracker ID is supplied'))
    finally:
        if writer:writer.release()
    with (output/'combined_pose_racket.jsonl').open('w') as stream:
        for row in combined:stream.write(json.dumps(row)+'\n')


def run(bundle,output,device='cuda',epochs=50,batch=8,sanity_epochs=150,verify_only=False):
    bundle,output=Path(bundle).resolve(),Path(output).resolve();m=verify(bundle)
    train=[r for r in m['samples'] if r['split']=='train' and r['train_eligible']]
    val=[r for r in m['samples'] if r['split']=='val'];usable_val=[r for r in val if r['valid_input']]
    print(f'VERIFIED v002: {len(train)} natural training crops + {len(m["ownership_pairs"])*2} synthetic selections; {len(val)} validation centres',flush=True)
    if verify_only:return True
    if output.exists():raise FileExistsError('Use a fresh output directory')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: select a GPU PyTorch pod')
    device=torch.device(device);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4);cv2.setNumThreads(1)
    output.mkdir(parents=True)
    print('DEVICE',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    write_json(output/'run_config.json',dict(format=FORMAT,manifest_sha256=sha(bundle/'manifest.json'),epochs=epochs,
        batch=batch,sanity_epochs=sanity_epochs,warm_start='v001 best epoch7',no_test_clips=True,far_side_evaluated=False,
        checkpoint_selection='unrestricted validation PCK20 then PCK50 then mean error'))
    try:
        gate=pair_only_dataset(bundle,train,m['ownership_pairs'][:4],m['image_size'])
        model=SelectedPlayerFusion(bundle/'models/baseline_v001.pt').to(device)
        print('OWNERSHIP SANITY: four identical-image pairs, eight selected-player targets; unrestricted decoder',flush=True)
        metrics,_=fit(model,gate,gate,bundle,m,output/'ownership_sanity',device,sanity_epochs,batch,True)
        passed=metrics['pck20_including_misses']>=.95
        write_json(output/'ownership_gate.json',dict(passed=passed,metrics=metrics,pairs=m['ownership_pairs'][:4],
                   scope='training-only synthetic ownership/geometry check, not real-world accuracy'))
        if not passed:
            write_json(output/'status.json',dict(status='ownership_gate_failed_full_training_not_started'))
            print('OWNERSHIP GATE FAILED. Full training NOT started; download diagnostics.',flush=True);return False
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        torch.manual_seed(42)
        model=SelectedPlayerFusion(bundle/'models/baseline_v001.pt').to(device)
        training=OwnerSamples(bundle,train,m['ownership_pairs'],True,m['image_size'])
        validation=OwnerSamples(bundle,usable_val,image_size=m['image_size'])
        print(f'FULL TRAINING STARTING: {len(training)} training examples; {len(val)} independent validation centres',flush=True)
        metrics,epoch=fit(model,training,validation,bundle,m,output/'full',device,epochs,batch,val_records=val)
        metrics,pred=evaluate(model,validation,device,batch,all_records=val)
        guarded,guard_pred=evaluate(model,validation,device,batch,'expanded_box',all_records=val)
        ablated,_=evaluate(model,validation,device,batch,ablate=True,all_records=val)
        gate_metrics,_=evaluate(model,gate,device,batch)
        write_json(output/'metrics.json',metrics);write_json(output/'predictions.json',pred)
        write_json(output/'guarded_metrics.json',guarded);write_json(output/'guarded_predictions.json',guard_pred)
        write_json(output/'conditioning_zeroed_metrics.json',ablated);write_json(output/'ownership_probe_after_full.json',gate_metrics)
        torch.save(checkpoint(model,m),output/'racket_center_owner_v002.pt')
        previews(bundle,m,pred,output)
        write_json(output/'comparison_to_v001.json',dict(v001=m['baseline_reference'],v002=metrics,v002_guarded=guarded,
                  note='same natural validation cohort; v002 also changes conditioning, warm-start, LR and synthetic training; not an architecture-only controlled experiment'))
        write_json(output/'status.json',dict(status='complete',best_epoch=epoch,final_test_evaluated=False,
                  validation_target_met_each_match=all(r['pck20_including_misses']>=.95 for r in metrics['per_source_group'].values()),far_side_evaluated=False))
        (output/'RESULTS.md').write_text(f'# v002 ownership trial\n\nBest epoch {epoch}; unrestricted PCK20 {metrics["pck20_including_misses"]:.1%}; guarded {guarded["pck20_including_misses"]:.1%}.\n\nInspect comparison_to_v001.json, conditioning_zeroed_metrics.json, ownership_probe_after_full.json and previews/near_side_preview.mp4. Synthetic ownership success is not evidence of far-side accuracy. The body model and tracker interfaces remain unchanged.\n',encoding='utf-8')
        print('FINAL METRICS',json.dumps(metrics),flush=True)
        return True
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error)));raise
    finally:package_results(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=Path('racket_mediapipe_run_v002'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--sanity-epochs',type=int,default=150)
    p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if min(args.epochs,args.batch_size,args.sanity_epochs)<1:p.error('Positive epochs and batch size required')
    if run(args.bundle,args.output,args.device,args.epochs,args.batch_size,args.sanity_epochs,args.verify_only) is False:raise SystemExit(2)


if __name__=='__main__':main()
