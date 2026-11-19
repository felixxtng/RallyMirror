"""Read-only export of near-side annotations for a frozen-MediaPipe racket trial.

Person selection never reads the target point. A selection is a camera-side
heuristic, not a verified athlete identity. Failed validation crops remain in
the denominator. Source annotations and shared detection/pose code are untouched.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

BODY_NAMES = 'nose left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_hip right_hip left_knee right_knee left_ankle right_ankle left_heel right_heel left_foot_index right_foot_index'.split()
VERSION = 'racket_mediapipe_v001'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def square_bounds(box):
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:
        raise ValueError('Invalid person box')
    side = int(math.ceil(max((y2-y1)*1.8, (x2-x1)*2.5)))
    return [math.floor((x1+x2-side)/2), math.floor((y1+y2-side)/2), side]


def to_unit(point, crop):
    return [(point[0]-crop[0])/crop[2], (point[1]-crop[1])/crop[2]]


def to_original(point, crop):
    return [point[0]*crop[2]+crop[0], point[1]*crop[2]+crop[1]]


def square_image(image, crop, size):
    x,y,side = crop
    # Warp the square directly, avoiding huge temporary arrays for edge players.
    matrix = np.float32([[size/side,0,-x*size/side], [0,size/side,-y*size/side]])
    return cv2.warpAffine(image,matrix,(size,size),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))


def select_people(detections, width, height):
    """Near-side proxy: plausible person, ranked by bottom edge, never by GT.

    Retain all candidate boxes in audit records. This is explicitly provisional
    until the team's court/tracker associations are available.
    """
    people=[]
    for person in detections:
        x1,y1,x2,y2=person['bbox']; w,h=x2-x1,y2-y1
        if person.get('confidence',0)<.3 or h<height*.08 or w<width*.015:
            continue
        if w*h<width*height*.002 or not .025*width<(x1+x2)/2<.975*width:
            continue
        people.append(person)
    return sorted(people,key=lambda p:p['bbox'][3],reverse=True)


def pose_array(points, crop):
    by_name={p['name']:p for p in points}
    result=[]
    for name in BODY_NAMES:
        p=by_name.get(name)
        if p is None:
            result.append([0.,0.,0.]); continue
        x,y=to_unit([p['x'],p['y']],crop)
        score=max(0.,min(1.,p.get('visibility',0)*p.get('presence',0)))
        if not all(math.isfinite(v) for v in (x,y,score)):
            result.append([0.,0.,0.]); continue
        result.append([x,y,score if 0<=x<1 and 0<=y<1 else 0.])
    return result


def export(root, output, size=512):
    from .schema import load_clip, approved, group_id
    from .source import FrameSource
    from .validate import validate
    from src.detection.detector import PlayerDetector
    from src.pose.estimator import PlayerPose
    root,output=Path(root),Path(output)
    if (output/'manifest.json').exists():
        raise FileExistsError('Completed bundle exists; use a fresh version')
    errors,_=validate(root)
    if errors: raise ValueError(errors)
    output.mkdir(parents=True,exist_ok=True)
    detector=PlayerDetector('yolo11n.pt')
    pose=PlayerPose()
    records=[]; demos=[]; counts=Counter(); sources=[]; selection_audit=[]
    started=time.monotonic()
    try:
        for clip in sorted((root/'clips').iterdir()):
            meta,ann=load_clip(clip)
            if meta['split'] not in ('train','val'): continue
            frames=[i for i in range(meta['frame_count']) if str(i) in ann['frames'] and
                    approved(ann['frames'][str(i)],meta,ann['frames']) and
                    any(p['v'] in (1,2) for p in ann['frames'][str(i)]['keypoints'])]
            if not frames: continue
            print(f"EXPORT {clip.name} {meta['split']}: {len(frames)} centres",flush=True)
            source=FrameSource(meta)
            sources.append(dict(clip=clip.name,split=meta['split'],source_group=group_id(meta['source_group']),
                                annotation_sha256=sha(clip/'annotations.json'),meta_sha256=sha(clip/'meta.json'),
                                frames=meta['frame_count'],located=len(frames),source_hash=meta['source_hash']))
            # Validation demo images are spread through the labelled span.
            demo_frames=set(frames[i] for i in np.linspace(0,len(frames)-1,min(32,len(frames)),dtype=int)) if meta['split']=='val' else set()
            try:
                for step,i in enumerate(frames,1):
                    image=source.read(i); row=ann['frames'][str(i)]; p=row['keypoints'][0]
                    truth=[p['x'],p['y']]
                    detections=detector.detect_people(image)
                    people=select_people(detections,meta['width'],meta['height'])
                    owner=people[0] if people else None
                    # Existing explicitly verified boxes take precedence, never
                    # pick a new box because it contains a ground-truth point.
                    verified=row.get('association_status')=='verified' and row.get('player_bbox') is not None
                    if verified: owner={'bbox':row['player_bbox'],'confidence':1.}
                    record=dict(id=f'{clip.name}_{i:05d}',clip=clip.name,frame=i,split=meta['split'],
                                source_group=group_id(meta['source_group']),original_size=[meta['width'],meta['height']],
                                target_original=truth,owner_box=owner['bbox'] if owner else None,
                                association='existing_verified' if verified else 'near_side_heuristic',
                                candidates=people,body_names=BODY_NAMES,valid_input=owner is not None)
                    if owner:
                        crop=square_bounds(owner['bbox']); target=to_unit(truth,crop)
                        points=pose.extract(image,owner)
                        record.update(crop=crop,target=target,pose=pose_array(points,crop),
                                      body_keypoints=points,pose_available=bool(points),
                                      target_in_crop=all(0<=x<1 for x in target),
                                      image=f"images/{record['id']}.jpg")
                        (output/'images').mkdir(exist_ok=True)
                        if not cv2.imwrite(str(output/record['image']),square_image(image,crop,size),[cv2.IMWRITE_JPEG_QUALITY,96]):
                            raise IOError('Cannot write crop')
                        counts['pose_found' if points else 'pose_missing']+=1
                    else:
                        record.update(crop=None,target=None,pose=None,body_keypoints=[],pose_available=False,
                                      target_in_crop=False,image=None)
                        counts['detector_missing']+=1
                    record['train_eligible']=record['valid_input'] and record['target_in_crop']
                    counts[meta['split']+'_located']+=1
                    counts[meta['split']+'_usable']+=int(record['train_eligible'])
                    if not record['target_in_crop']: counts[meta['split']+'_outside_or_missing']+=1
                    records.append(record)
                    selection_audit.append(dict(clip=clip.name,frame=i,split=meta['split'],
                                               association=record['association'],pose_available=record['pose_available'],
                                               target_in_crop=record['target_in_crop'],box=json.dumps(record['owner_box'])))
                    if i in demo_frames:
                        # Full-frame evidence plus an unscored far-side crop.
                        full_path=f"demo_frames/{record['id']}.jpg"
                        (output/'demo_frames').mkdir(exist_ok=True)
                        preview=cv2.resize(image,(1280,round(meta['height']*1280/meta['width'])))
                        cv2.imwrite(str(output/full_path),preview,[cv2.IMWRITE_JPEG_QUALITY,92])
                        near_box=owner['bbox'] if owner else None
                        candidates=[person for person in people if near_box is None or
                                    person['bbox'][3]<near_box[3]-.06*meta['height']]
                        # Largest remaining on-court-looking person is only a
                        # qualitative far-side candidate; never a scored label.
                        far=max(candidates,key=lambda person:(person['bbox'][2]-person['bbox'][0])*(person['bbox'][3]-person['bbox'][1]),default=None)
                        demo=dict(sample_id=record['id'],full_image=full_path,original_size=record['original_size'],far=None)
                        if far:
                            bounds=square_bounds(far['bbox']); far_points=pose.extract(image,far)
                            path=f"images/{record['id']}_far.jpg"
                            cv2.imwrite(str(output/path),square_image(image,bounds,size),[cv2.IMWRITE_JPEG_QUALITY,96])
                            demo['far']=dict(image=path,crop=bounds,pose=pose_array(far_points,bounds),
                                             owner_box=far['bbox'],body_keypoints=far_points,
                                             scope='qualitative_only_no_far_side_ground_truth')
                        demos.append(demo)
                    if step==1 or step%100==0 or step==len(frames):
                        print(f"  {step}/{len(frames)} | elapsed {time.monotonic()-started:.0f}s",flush=True)
                write_json(output/'preparation_progress.json',dict(last_clip=clip.name,counts=dict(counts),samples=len(records)))
            finally: source.close()
    finally: pose.close()
    groups={split:{s['source_group'] for s in records if s['split']==split} for split in ['train','val']}
    if groups['train']&groups['val']: raise ValueError('Source-match leakage')
    # Hash every runtime input so an incomplete upload fails before training.
    paths=[p for p in output.rglob('*') if p.is_file() and p.suffix in ('.jpg','.pt','.task')]
    manifest=dict(version=VERSION,image_size=size,body_names=BODY_NAMES,annotation_target='near_side_player',
                  model='imagenet_resnet18_image_encoder_plus_frozen_mediapipe_pose_heatmaps',
                  mediapipe_finetuned=False,split_unit='whole_match',sources=sources,samples=records,demos=demos,
                  counts=dict(counts),file_hashes={p.relative_to(output).as_posix():sha(p) for p in paths},
                  validation_scope='located near-side racket centres only; all 578 labels retained in denominator',
                  far_side_scope='qualitative predictions only; no ground truth and no accuracy claim',
                  owner_policy='existing explicitly verified body box, otherwise bottom-edge near-side heuristic independent of GT')
    write_json(output/'manifest.json',manifest)
    with (output/'association_audit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selection_audit[0]));writer.writeheader();writer.writerows(selection_audit)
    print('COMPLETE',dict(counts),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/racket_manual'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=512)
    args=parser.parse_args()
    export(args.root,args.output,args.size)


if __name__=='__main__': main()
