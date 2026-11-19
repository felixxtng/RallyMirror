"""Portable near-side racket-centre trial with frozen MediaPipe features.

The .task body model is not fine-tuned. The trained .pt fuses image features
and cached body landmarks. Original-frame metrics include every located val
label, even detection/crop failures. No temporal or court module is imported.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

from .mediapipe_bundle import BODY_NAMES, VERSION, sha, to_original, write_json
from .mediapipe_model import RacketPoseFusion,augment,decode,image_tensor,spatial_loss


class Samples(Dataset):
    def __init__(self,bundle,records,training=False,seed=42):
        self.bundle=Path(bundle);self.records=records;self.training=training
        self.rng=np.random.default_rng(seed)

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        record=self.records[index]
        image=cv2.imread(str(self.bundle/record['image']))
        if image is None: raise ValueError(f"Unreadable image {record['image']}")
        pose=np.asarray(record['pose'],np.float32)
        target=np.asarray(record.get('target') or [.5,.5],np.float32)
        if self.training: image,pose,target=augment(image,pose,target,self.rng)
        return image_tensor(image),torch.from_numpy(pose.copy()),torch.from_numpy(target.copy()),index


def verify_bundle(bundle):
    bundle=Path(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    if manifest['version']!=VERSION or manifest['body_names']!=BODY_NAMES:
        raise ValueError('Unsupported model/dataset contract')
    for path,expected in manifest['file_hashes'].items():
        if not (bundle/path).is_file() or sha(bundle/path)!=expected:
            raise ValueError(f'Upload incomplete or changed file: {path}')
    groups=defaultdict(set)
    for sample in manifest['samples']:
        groups[sample['split']].add(sample['source_group'])
    if groups['train']&groups['val']: raise ValueError('Match leakage')
    if not groups['train'] or not groups['val']: raise ValueError('Missing train or validation')
    return manifest


def summarize(predictions):
    """Misses are included as failures, never silently removed from PCK."""
    valid=[p['error_original_pixels'] for p in predictions if p['error_original_pixels'] is not None]
    n=len(predictions)
    return dict(samples=n,missing=n-len(valid),within_20px=sum(e<=20 for e in valid),
                within_50px=sum(e<=50 for e in valid),pck20_including_misses=sum(e<=20 for e in valid)/n if n else 0.,
                pck50_including_misses=sum(e<=50 for e in valid)/n if n else 0.,
                mean_error_detected_only=float(np.mean(valid)) if valid else None,
                median_error_detected_only=float(np.median(valid)) if valid else None,
                coordinate_system='original_video_pixels',scope='located near-side centres; not both-racket recall')


@torch.inference_mode()
def evaluate(model,bundle,records,device,batch=8,pose_enabled=True):
    model.eval(); usable=[r for r in records if r['valid_input']]
    dataset=Samples(bundle,usable)
    loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0)
    mapped={}
    for images,pose,_,indices in loader:
        if not pose_enabled: pose.zero_()
        points,scores=decode(model(images.to(device),pose.to(device)))
        for point,score,index in zip(points.cpu().tolist(),scores.cpu().tolist(),indices.tolist()):
            sample=usable[index]; point=to_original(point,sample['crop'])
            w,h=sample['original_size']
            valid=all(math.isfinite(v) for v in point) and 0<=point[0]<w and 0<=point[1]<h
            if not sample['target_in_crop']: valid=False
            error=math.dist(point,sample['target_original']) if valid else None
            mapped[sample['id']]=dict(id=sample['id'],clip=sample['clip'],frame=sample['frame'],
                                     source_group=sample['source_group'],point_original=point if valid else None,
                                     target_original=sample['target_original'],error_original_pixels=error,
                                     concentration=score,reason='ok' if valid else 'crop_or_prediction_outside',
                                     association=sample['association'])
    predictions=[mapped.get(s['id'],dict(id=s['id'],clip=s['clip'],frame=s['frame'],source_group=s['source_group'],
                 point_original=None,target_original=s['target_original'],error_original_pixels=None,
                 concentration=None,reason='person_not_detected',association=s['association'])) for s in records]
    metrics=summarize(predictions)
    for key in ['source_group','clip']:
        grouped=defaultdict(list)
        for row in predictions:grouped[row[key]].append(row)
        metrics['per_'+key]={group:summarize(rows) for group,rows in grouped.items()}
    return metrics,predictions


def checkpoint(model,manifest):
    return dict(format=VERSION,state_dict=model.state_dict(),image_size=manifest['image_size'],body_names=BODY_NAMES,
                output='head_center',coordinate_system='crop_normalized_xy',mediapipe_finetuned=False,
                body_model='pose_landmarker_lite.task',normalization='imagenet_rgb',
                preprocessing='square_bounds and square_image from mediapipe_bundle.py',
                annotation_target='near_side_player',far_side_validated=False)


def train_one(model,bundle,records,device,epochs,batch,output,manifest,validation=None,sanity=False):
    output.mkdir(parents=True,exist_ok=True)
    trainset=Samples(bundle,records,training=not sanity,seed=42)
    loader=DataLoader(trainset,batch_size=batch,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    backbone=list(model.encoder.parameters());backbone_ids={id(p) for p in backbone}
    head=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':.0001 if not sanity else .0002},
                                 {'params':head,'lr':.001}],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(epochs,1),eta_min=.00001)
    best_key=(-1.,-1.,-float('inf'));best_epoch=0;history=[];bad_epochs=0
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0;start=time.monotonic()
        for step,(images,pose,target,_) in enumerate(loader,1):
            images,pose,target=images.to(device),pose.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images,pose);loss=spatial_loss(logits,target)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=loss.item()*len(images);count+=len(images)
            if step==1 or step%50==0 or step==len(loader):
                print(f"{'SANITY' if sanity else 'TRAIN'} epoch {epoch}/{epochs} batch {step}/{len(loader)} loss={loss.item():.4f}",flush=True)
        scheduler.step()
        metrics,preds=evaluate(model,bundle,validation if validation is not None else records,device,batch)
        error=metrics['mean_error_detected_only']
        key=(metrics['pck20_including_misses'],metrics['pck50_including_misses'],-(error if error is not None else float('inf')))
        improved=key>best_key
        row=dict(epoch=epoch,loss=total/count,seconds=time.monotonic()-start,**metrics)
        history.append(row);write_json(output/'history.json',history)
        print(f"EPOCH {epoch}: loss={total/count:.4f}; PCK20={metrics['pck20_including_misses']:.3f}; "
              f"PCK50={metrics['pck50_including_misses']:.3f}; missing={metrics['missing']}; "
              f"mean_px={error}; {row['seconds']:.1f}s",flush=True)
        if improved:
            best_key=key;best_epoch=epoch;bad_epochs=0
            torch.save(checkpoint(model,manifest),output/'best.pt')
            write_json(output/'best_metrics.json',metrics);write_json(output/'best_predictions.json',preds)
        else:bad_epochs+=1
        torch.save(checkpoint(model,manifest),output/'last.pt')
        if sanity and metrics['pck20_including_misses']>=.95:break
        if not sanity and bad_epochs>=15:
            print(f'Early stop: 15 epochs without PCK improvement; best epoch {best_epoch}',flush=True);break
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'])
    return json.loads((output/'best_metrics.json').read_text()),best_epoch


def draw_body(image,points,scale=1.):
    pairs=[('left_shoulder','right_shoulder'),('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
           ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),('left_shoulder','left_hip'),
           ('right_shoulder','right_hip'),('left_hip','right_hip'),('left_hip','left_knee'),
           ('right_hip','right_knee'),('left_knee','left_ankle'),('right_knee','right_ankle')]
    by_name={p['name']:p for p in points if p.get('visibility',0)*p.get('presence',0)>.2}
    for a,b in pairs:
        if a in by_name and b in by_name:
            xy=lambda name:tuple(round(by_name[name][k]*scale) for k in ['x','y'])
            cv2.line(image,xy(a),xy(b),(255,180,50),2,cv2.LINE_AA)


def connect_racket(image,points,point,scale=1.,color=(255,0,255)):
    if point is None:return
    xy=tuple(round(v*scale) for v in point)
    cv2.circle(image,xy,6,color,-1,cv2.LINE_AA)
    wrists=[p for p in points if p['name'] in ['left_wrist','right_wrist'] and p.get('visibility',0)*p.get('presence',0)>.2]
    if wrists:
        # Drawing choice only, not a learned handedness or racket ownership claim.
        wrist=min(wrists,key=lambda p:math.dist([p['x'],p['y']],point))
        cv2.line(image,tuple(round(wrist[k]*scale) for k in ['x','y']),xy,color,2,cv2.LINE_AA)


@torch.inference_mode()
def artifacts(model,bundle,manifest,predictions,output,device):
    output=Path(output);(output/'previews').mkdir(exist_ok=True)
    lookup={r['id']:r for r in manifest['samples']};pred={r['id']:r for r in predictions}
    # Rank near-side predictions to inspect errors instead of trusting confidence.
    ordered=sorted(predictions,key=lambda p:p['error_original_pixels'] if p['error_original_pixels'] is not None else float('inf'))
    for name,selected in [('best',ordered[:16]),('worst',list(reversed(ordered[-16:])) )]:
        tiles=[]
        for row in selected:
            sample=lookup[row['id']]
            if not sample.get('image'):continue
            image=cv2.imread(str(bundle/sample['image']));size=image.shape[0]
            draw_body(image,[dict(p,x=(p['x']-sample['crop'][0])*size/sample['crop'][2],
                                 y=(p['y']-sample['crop'][1])*size/sample['crop'][2]) for p in sample['body_keypoints']])
            gt=sample['target'];cv2.circle(image,tuple(round(v*size) for v in gt),6,(0,255,0),2)
            if row['point_original']:
                from .mediapipe_bundle import to_unit
                point=to_unit(row['point_original'],sample['crop'])
                cv2.drawMarker(image,tuple(round(v*size) for v in point),(255,0,255),cv2.MARKER_CROSS,18,2)
            tile=cv2.resize(image,(320,320))
            error=row['error_original_pixels']
            cv2.rectangle(tile,(0,0),(320,42),(0,0,0),-1)
            cv2.putText(tile,f"{row['id']} error {error:.0f}px" if error is not None else row['id']+' MISS',
                        (5,25),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            tiles.append(tile)
        if tiles:
            tiles += [np.zeros_like(tiles[0]) for _ in range((-len(tiles))%4)]
            sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
            cv2.imwrite(str(output/'previews'/f'{name}_16.jpg'),sheet)
    frames=[];combined=[]
    model.eval()
    for demo in manifest['demos']:
        near=lookup[demo['sample_id']];near_pred=pred[demo['sample_id']]
        image=cv2.imread(str(bundle/demo['full_image']));scale=image.shape[1]/demo['original_size'][0]
        draw_body(image,near['body_keypoints'],scale)
        connect_racket(image,near['body_keypoints'],near_pred['point_original'],scale)
        cv2.circle(image,tuple(round(v*scale) for v in near['target_original']),5,(0,255,0),2)
        result=dict(clip=near['clip'],frame=near['frame'],coordinate_system='original_pixels',
                    players=[dict(side='near',bbox=near['owner_box'],keypoints=near['body_keypoints'],
                                  racket=dict(name='head_center',point=near_pred['point_original']))])
        far=demo['far']
        if far:
            tensor=image_tensor(cv2.imread(str(bundle/far['image'])))[None].to(device)
            body=torch.tensor(far['pose'],dtype=torch.float32,device=device)[None]
            point,_=decode(model(tensor,body));point=to_original(point[0].cpu().tolist(),far['crop'])
            draw_body(image,far['body_keypoints'],scale);connect_racket(image,far['body_keypoints'],point,scale,(0,200,255))
            result['players'].append(dict(side='far_candidate_unverified',bbox=far['owner_box'],keypoints=far['body_keypoints'],
                                           racket=dict(name='head_center',point=point),accuracy='unmeasured_no_labels'))
        cv2.rectangle(image,(0,0),(image.shape[1],65),(0,0,0),-1)
        cv2.putText(image,f"{near['id']} | green: near label | magenta: near prediction",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(image,'orange: FAR CANDIDATE - no ground truth; side selection is heuristic',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,200,255),1)
        frames.append(image);combined.append(result)
    if frames:
        writer=cv2.VideoWriter(str(output/'previews/both_sides_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),6.,(frames[0].shape[1],frames[0].shape[0]))
        if not writer.isOpened():raise RuntimeError('Cannot create preview video')
        for frame in frames:writer.write(frame)
        writer.release()
        tiles=[cv2.resize(frames[i],(640,360)) for i in np.linspace(0,len(frames)-1,min(8,len(frames)),dtype=int)]
        tiles += [np.zeros_like(tiles[0]) for _ in range(len(tiles)%2)]
        cv2.imwrite(str(output/'previews/both_sides_contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    with (output/'combined_pose_racket.jsonl').open('w',encoding='utf-8') as stream:
        for row in combined:stream.write(json.dumps(row)+'\n')


def package_results(output):
    destination=output.parent/(output.name+'_results.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(output).as_posix())
    print(f'DOWNLOAD RESULTS: {destination.resolve()}',flush=True)
    return destination


def run(bundle,output,device='cuda',epochs=50,batch=8,sanity_epochs=100,verify_only=False):
    bundle,output=Path(bundle).resolve(),Path(output).resolve()
    manifest=verify_bundle(bundle)
    train=[r for r in manifest['samples'] if r['split']=='train' and r['train_eligible']]
    val=[r for r in manifest['samples'] if r['split']=='val']
    print(f"VERIFIED: train {len(train)} / {manifest['counts']['train_located']} located; val {len(val)} including crop failures; "
          f"MediaPipe cached; no test clips",flush=True)
    if verify_only:return
    if output.exists():raise FileExistsError('Use a fresh output directory; existing runs are preserved')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: choose a GPU pod')
    device=torch.device(device);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    print('Torch',torch.__version__,'DEVICE',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    output.mkdir(parents=True)
    write_json(output/'run_config.json',dict(version=VERSION,manifest_sha256=sha(bundle/'manifest.json'),
               epochs=epochs,batch=batch,sanity_epochs=sanity_epochs,device=str(device),torch=torch.__version__,
               selection='best validation PCK20, then PCK50, then mean detected error',
               body_model_frozen=True,far_side_accuracy_measured=False))
    try:
        if len(train)<8 or not val:raise ValueError('Insufficient eligible data')
        # One sample per training clip where possible: this gate never uses val.
        grouped=defaultdict(list)
        for sample in train:grouped[sample['clip']].append(sample)
        sanity=[rows[len(rows)//2] for rows in grouped.values()][:8]
        while len(sanity)<8:
            sanity.append(next(r for r in train if r['id'] not in {s['id'] for s in sanity}))
        model=RacketPoseFusion(bundle/'models/resnet18_imagenet.pt').to(device)
        print('SANITY: overfit 8 training crops only; gate >=95% within 20 original pixels',flush=True)
        metrics,_=train_one(model,bundle,sanity,device,sanity_epochs,min(batch,8),output/'sanity',manifest,sanity=True)
        passed=metrics['pck20_including_misses']>=.95
        write_json(output/'sanity_gate.json',dict(passed=passed,metrics=metrics,sample_ids=[r['id'] for r in sanity]))
        if not passed:
            write_json(output/'status.json',dict(status='sanity_failed_full_training_not_started'))
            print('SANITY FAILED: full training was NOT started. Download diagnostics.',flush=True)
            return
        # Start fresh: sanity is a learning/geometry check, not a validation seed.
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        torch.manual_seed(42)
        model=RacketPoseFusion(bundle/'models/resnet18_imagenet.pt').to(device)
        print(f'FULL TRAINING STARTING: {len(train)} train; {len(val)} validation labels',flush=True)
        metrics,best_epoch=train_one(model,bundle,train,device,epochs,batch,output/'full',manifest,validation=val)
        metrics,predictions=evaluate(model,bundle,val,device,batch)
        # Same checkpoint with zero pose: diagnostic of reliance on body input,
        # not a separately trained image-only comparison model.
        ablation,_=evaluate(model,bundle,val,device,batch,pose_enabled=False)
        write_json(output/'metrics.json',metrics);write_json(output/'predictions.json',predictions)
        write_json(output/'pose_input_ablation.json',ablation)
        torch.save(checkpoint(model,manifest),output/'racket_center_fusion.pt')
        artifacts(model,bundle,manifest,predictions,output,device)
        per_match=metrics['per_source_group']
        target_met=all(m['pck20_including_misses']>=.95 for m in per_match.values())
        write_json(output/'status.json',dict(status='complete',best_epoch=best_epoch,
                   validation_target_met_each_match=target_met,final_test_evaluated=False,far_side_accuracy_measured=False))
        (output/'RESULTS.md').write_text(
            '# MediaPipe plus racket trial\n\n'+
            f"Best epoch: {best_epoch}. PCK20: {metrics['pck20_including_misses']:.1%}. "
            f"PCK50: {metrics['pck50_including_misses']:.1%}. Missing: {metrics['missing']}/{len(val)}.\n\n"+
            'See metrics.json for each held-out match, best/worst previews, and both_sides_preview.mp4. '
            'Far-side orange predictions are qualitative only. Camera-side ownership is a heuristic. '
            'A wrist-to-racket line is only a display connection to the nearest visible wrist. '
            'MediaPipe body weights are frozen; the separate .pt learns the extra point. '
            'Combined JSONL is a separate trial output, not a replacement for teammate interfaces.\n',encoding='utf-8')
        print('FINAL METRICS',json.dumps(metrics),flush=True)
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error)))
        raise
    finally:package_results(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=Path('racket_mediapipe_run_v001'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--sanity-epochs',type=int,default=100)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.sanity_epochs)<1:parser.error('Epochs and batch size must be positive')
    run(args.bundle,args.output,args.device,args.epochs,args.batch_size,args.sanity_epochs,args.verify_only)


if __name__=='__main__':main()
