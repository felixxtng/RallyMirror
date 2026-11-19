"""Portable real-ownership training bundle; originals and old runs stay intact."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from .mediapipe_bundle import BODY_NAMES,sha,square_image,write_json
from .ownership import iou
from .real_owner_data import FORMAT,SIDES,scene_bounds,selected_pose
from .schema import load_clip
from .source import FrameSource


def build_features(repo,pairs,output,warm,legacy,pose_extractor=None,size=640,context_padding=0.,warm_filename='baseline_v002.pt'):
    repo,pairs,output,warm,legacy=map(Path,[repo,pairs,output,warm,legacy])
    if output.exists():raise FileExistsError('Use a fresh feature bundle')
    exported=json.loads((pairs/'export.json').read_text());records=[]
    for split in ['train','val']:
        for line in (pairs/(split+'.jsonl')).read_text().splitlines():
            r=json.loads(line)
            if r['split']!=split or r['format']!='racket_real_ownership_v004':raise ValueError('Invalid real ownership export')
            records.append(r)
    groups={split:{r['source_group'] for r in records if r['split']==split} for split in ['train','val']}
    if not all(groups.values()) or groups['train']&groups['val']:raise ValueError('Whole-match split violation')
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate source frame')
    own_pose=pose_extractor is None
    if own_pose:
        from src.pose.estimator import PlayerPose
        pose_extractor=PlayerPose(repo/'models/pose_landmarker_lite.task')
    output.mkdir(parents=True);(output/'images').mkdir();(output/'demo_frames').mkdir();(output/'models').mkdir()
    if Path(warm_filename).name!=warm_filename:raise ValueError('Warm checkpoint must be a filename')
    shutil.copyfile(warm,output/'models'/warm_filename)
    shutil.copyfile(repo/'models/pose_landmarker_lite.task',output/'models/pose_landmarker_lite.task')
    frames=[];source=None;current=None;poses=Counter()
    try:
        for i,r in enumerate(records):
            if current!=r['clip']:
                if source:source.close()
                meta,_=load_clip(repo/'data/racket_manual/clips'/r['clip'])
                if sha(Path(meta['source']))!=r['source_hash']:raise ValueError('Video changed: '+r['clip'])
                source=FrameSource(meta);current=r['clip']
            image=source.read(r['frame']);boxes=[r['players'][s]['bbox'] for s in SIDES]
            crop=scene_bounds(boxes,context_padding)
            entry=dict(id=r['id'],clip=r['clip'],frame=r['frame'],split=r['split'],source_group=r['source_group'],
                original_size=r['original_size'],crop=crop,image='images/'+r['id']+'.jpg',players={},
                body_box_iou=iou(*boxes),locked_owner=r['locked_owner'],locked_reference_sha256=r['locked_reference_sha256'])
            for j,side in enumerate(SIDES):
                player=r['players'][side]
                if not player['ownership_verified']:raise ValueError('Unverified owner: '+r['id'])
                box=player['bbox'];body=pose_extractor.extract(image,dict(bbox=box))
                pose,status=selected_pose(body,box,[boxes[1-j]],crop);poses[status]+=1
                head=player['head_center'];point=[head['x'],head['y']]
                target=[(point[k]-crop[k])/crop[2] for k in range(2)]
                entry['players'][side]=dict(owner_box=box,target_original=point,target=target,pose=pose.tolist(),
                    body_keypoints=body,pose_status=status,valid_target=all(math.isfinite(v) and 0<=v<1 for v in target),
                    point_origin=player['point_origin'])
            entry['train_eligible']=all(p['valid_target'] for p in entry['players'].values())
            entry['head_separation_original_pixels']=math.dist(*[entry['players'][s]['target_original'] for s in SIDES])
            if not cv2.imwrite(str(output/entry['image']),square_image(image,crop,size),[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('Cannot write scene image')
            if r['split']=='val':
                entry['full_image']='demo_frames/'+r['id']+'.jpg'
                h=round(image.shape[0]*1280/image.shape[1]);h+=h%2
                cv2.imwrite(str(output/entry['full_image']),cv2.resize(image,(1280,h)),[cv2.IMWRITE_JPEG_QUALITY,94])
            frames.append(entry)
            if i==0 or (i+1)%25==0 or i+1==len(records):print(f'Real ownership features {i+1}/{len(records)}; MediaPipe selections {sum(poses.values())}',flush=True)
    finally:
        if source:source.close()
        if own_pose:pose_extractor.close()
    old=json.loads((legacy/'manifest.json').read_text());legacy_records=[]
    (output/'legacy_images').mkdir()
    for original in old['samples']:
        if original['split']!='val':continue
        r=copy.deepcopy(original)
        if r['image']:
            if sha(legacy/r['image'])!=old['file_hashes'][r['image']]:raise ValueError('Legacy JPEG changed')
            dest='legacy_images/'+r['id']+'.jpg';shutil.copyfile(legacy/r['image'],output/dest);r['image']=dest
        legacy_records.append(r)
    manifest=dict(version=FORMAT,status='features_ready',image_size=size,body_names=BODY_NAMES,frames=frames,
        legacy_validation=legacy_records,legacy_manifest_sha256=sha(legacy/'manifest.json'),
        initial_checkpoint_sha256=sha(warm),initial_checkpoint='models/'+warm_filename,context_padding=context_padding,
        split_unit='whole_source_match',coordinate_system='original_video_pixels',
        context='common union of two expanded body squares; independent of racket targets',
        counts=dict(train_frames=sum(r['split']=='train' for r in frames),val_frames=sum(r['split']=='val' for r in frames),
            train_eligible_frames=sum(r['split']=='train' and r['train_eligible'] for r in frames),
            legacy_near_validation_queries=len(legacy_records),pose_status=dict(poses)),
        ownership_export=exported,final_test_evaluated=False,mediapipe_finetuned=False,
        file_hashes={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'manifest.json',manifest)
    return manifest


def apply_training_review(output,exclusions):
    """Record visual-review exclusions in a derived bundle, never source labels.

    Validation cannot be removed with this helper. Out-of-context targets are
    already excluded by build_features and retained in this audit as well.
    """
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    lookup={r['id']:r for r in manifest['frames']}
    if any(identity not in lookup or lookup[identity]['split']!='train' or not reason
           for identity,reason in exclusions.items()):raise ValueError('Only identified TRAIN frames may be excluded')
    for identity,reason in exclusions.items():
        lookup[identity]['train_eligible']=False;lookup[identity]['review_exclusion_reason']=reason
    audit=[dict(id=r['id'],reason=r.get('review_exclusion_reason','One or both targets outside body-derived scene context'))
        for r in manifest['frames'] if r['split']=='train' and not r['train_eligible']]
    manifest['counts']['train_eligible_frames']=sum(r['split']=='train' and r['train_eligible'] for r in manifest['frames'])
    manifest['counts']['excluded_train_frames']=len(audit)
    manifest['training_review_exclusions']=audit
    write_json(output/'training_review_exclusions.json',audit)
    manifest['file_hashes']['training_review_exclusions.json']=sha(output/'training_review_exclusions.json')
    write_json(output/'manifest.json',manifest)
    return audit


def finalize(repo,output,destination,start_here=None):
    repo,output,destination=map(Path,[repo,output,destination])
    if destination.exists():raise FileExistsError('Preserve the existing package')
    for folder in ['src','src/racket','src/pose']:
        (output/folder).mkdir(parents=True,exist_ok=True);(output/folder/'__init__.py').write_text('')
    modules=['mediapipe_bundle','mediapipe_model','mediapipe_train','owner_model','owner_data','owner_train',
             'player_constraint','real_owner_data','real_owner_train','real_owner_infer']
    for module in modules:shutil.copyfile(repo/f'src/racket/{module}.py',output/f'src/racket/{module}.py')
    shutil.copyfile(repo/'src/pose/estimator.py',output/'src/pose/estimator.py')
    (output/'run_racket_ownership.py').write_text('from src.racket.real_owner_train import main\nif __name__=="__main__": main()\n')
    setup=(repo/'scripts/setup_racket_runpod.py').read_text().replace('run_racket_mediapipe.py','run_racket_ownership.py')
    (output/'setup_runpod.py').write_text(setup)
    shutil.copyfile(start_here or repo/'docs/racket_real_ownership_trial.md',output/'START_HERE.md')
    manifest=json.loads((output/'manifest.json').read_text());manifest['status']='ready'
    manifest['file_hashes']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*')
        if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    write_json(output/'manifest.json',manifest)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(output).as_posix())
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC check failed')
    report=dict(package=str(destination.resolve()),bytes=destination.stat().st_size,sha256=sha(destination),
        manifest_sha256=sha(output/'manifest.json'),counts=manifest['counts'],cloud_training_started=False)
    write_json(destination.with_suffix('.json'),report)
    print(json.dumps(report,indent=2),flush=True)
    return report
