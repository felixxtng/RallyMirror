"""Inventory shuttle data and prepare a conservative, reproducible group split.

No model inference/training or original metadata edits. Match IDs remain declared
IDs; a proposed test split is conditional on independent source verification.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np

from .dataset import ShuttleDataset, letterbox_transform, read_labels

EXPECTED = {**{f'rally{i:02d}': group for group, start, end in [
    ('match01',1,4), ('match03',5,8), ('match04',9,10), ('match05',11,13),
    ('match06',14,16), ('match07',17,21), ('match08',22,24), ('match09',25,28),
    ('match10',29,32), ('match11',33,36), ('match12',37,40), ('match13',41,44),
    ('match14',45,48)] for i in range(start,end+1)}}
EVALUATED = {'rally18','rally24','rally29','rally35','rally44'}
TEST_CANDIDATES = {'match12','match14'}
LINKED = {'match02': 'match02+match09', 'match09': 'match02+match09'}
MANIFEST_FIELDS = ['rally_id','scope','source_group','leakage_block','video_path','clip_path',
    'meta_path','labels_path','frame_count','visible_labels','invisible_labels','unknown_labels',
    'labelled_frames','width','height','fps','current_split','historical_video_path_exists',
    'video_sha256','meta_sha256','labels_sha256','v2_training_clip','v2_validation_clip',
    'historical_exploratory_evaluation','group_has_exploratory_evaluation','mapping_status',
    'test_eligibility','observed_source_evidence']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    with Path(path).open('x') as file:
        json.dump(value,file,indent=2)
        file.write('\n')


def select_split(group_counts, forced_train, test_candidates, forced_val=()):
    """Exhaustive minimum squared frame-fraction error; deterministic tie-break.

    Atomic units are declared groups; validation01 is outside the 48-rally target.
    Test is allowed only on unused candidate groups, never on trained/developed
    groups. Source-overlap blocks must be validated separately before staging.
    """
    names=sorted(group_counts)
    total=sum(group_counts.values())
    if total<=0 or not set(forced_train)<=set(names) or not set(test_candidates)<=set(names):
        raise ValueError('Invalid groups or no labelled frames')
    if not set(forced_val)<=set(names) or set(forced_train)&set(forced_val):
        raise ValueError('Invalid forced validation groups')
    if (set(forced_train)|set(forced_val))&set(test_candidates):
        raise ValueError('Previously exposed group cannot be a test candidate')
    free=[g for g in names if g not in set(forced_train)|set(forced_val)]
    options=[('train','val','test') if g in test_candidates else ('train','val') for g in free]
    best=None
    for values in product(*options):
        assignment=dict.fromkeys(sorted(forced_train),'train')
        assignment.update(dict.fromkeys(sorted(forced_val),'val'))
        assignment.update(zip(free,values))
        totals={s:sum(group_counts[g] for g in names if assignment[g]==s) for s in ('train','val','test')}
        if any(n==0 for n in totals.values()):
            continue
        objective=sum((totals[s]/total-target)**2 for s,target in zip(('train','val','test'),(.7,.15,.15)))
        key=(objective,tuple(assignment[g] for g in names))
        if best is None or key<best[0]:
            best=(key,assignment,totals)
    if best is None:
        raise ValueError('No nonempty group-respecting train/val/test split is feasible')
    return best[1],best[2],best[0][0]


def validate_assignments(rows, assignments):
    blocks={}
    for row in rows:
        if row['scope']!='primary_48':
            continue
        split=assignments[row['source_group']]
        if split not in ('train','val','test'):
            raise ValueError('Invalid split')
        old=blocks.setdefault(row['leakage_block'],split)
        if old!=split:
            raise ValueError('Linked source groups cross splits')
        if row['v2_training_clip'] and split!='train':
            raise ValueError('V2 training data cannot become a V2 holdout')
        if split=='test' and row['test_eligibility']!='candidate_pending_source_verification':
            raise ValueError('Test includes historical training/development data')


def inventory(root, evidence):
    run=json.loads(Path('outputs/shuttle_v2/run.json').read_text())
    train_ids={Path(p.replace('\\','/')).name for p in run['train_clips']}
    val_ids={Path(p.replace('\\','/')).name for p in run['val_clips']}
    groups_evaluated={EXPECTED[n] for n in EVALUATED}
    rows=[]
    issues=[]
    for name in list(sorted(EXPECTED))+['validation01']:
        clip=root/'clips'/name
        meta_path=clip/'meta.json';labels_path=clip/'labels.csv';video=root/'videos'/f'{name}.mp4'
        for path in (meta_path,labels_path,video):
            if not path.is_file():raise FileNotFoundError(path)
        m=json.loads(meta_path.read_text())
        group=m.get('source_group')
        if not isinstance(group,str) or not re.fullmatch(r'match\d{2}',group):
            raise ValueError(f'{name}: invalid source_group')
        if name in EXPECTED and EXPECTED[name]!=group:
            raise ValueError(f'{name}: source_group contradicts supplied mapping')
        with labels_path.open(newline='') as file:
            indices=[int(r['frame']) for r in csv.DictReader(file)]
        if len(indices)!=len(set(indices)):
            raise ValueError(f'{name}: duplicate labels')
        labels=read_labels(labels_path)
        count=m['frame_count']
        if set(labels)!=set(range(count)):
            raise ValueError(f'{name}: label indices do not cover the clip')
        for r in labels.values():
            if r['visible']==1 and not (0<=r['x']<m['original_width'] and 0<=r['y']<m['original_height']):
                raise ValueError(f'{name}: label outside image')
        counts={state:sum(r['visible']==state for r in labels.values()) for state in (0,1,None)}
        if counts[0]+counts[1]+counts[None]!=count:raise ValueError('Label count mismatch')
        t=m['transform'];expected=letterbox_transform(m['original_width'],m['original_height'],t['width'],t['height'])
        if any(not math.isclose(t[k],v,abs_tol=1e-9) for k,v in expected.items()):
            raise ValueError(f'{name}: transform mismatch')
        frames=[clip/'frames'/f'{i:06d}.jpg' for i in range(count)]
        if any(not f.is_file() for f in frames):
            raise ValueError(f'{name}: missing canonical prepared frames')
        extras=sorted({p.name for p in (clip/'frames').glob('*.jpg')}-{p.name for p in frames})
        if extras:
            issues.append(dict(rally_id=name,kind='unreferenced_prepared_backups',files=extras,
                               action='preserved; excluded from canonical frame inventory and hashes'))
        cap=cv2.VideoCapture(str(video))
        values=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if values!=(m['original_width'],m['original_height'],count) or not math.isclose(fps,m['fps'],rel_tol=1e-5):
            raise ValueError(f'{name}: video metadata disagree')
        if m.get('start_frame',0)!=0:raise ValueError(f'{name}: nonzero frame origin requires review')
        # Read boundary/middle images to check dimensions, without loading every JPEG.
        for i in sorted({0,count//2,count-1}):
            image=cv2.imread(str(frames[i]))
            if image is None or image.shape[:2]!=(t['height'],t['width']):
                raise ValueError(f'{name}: prepared image dimensions disagree')
        historic=m.get('video','')
        path_exists=not ('\\' in historic) and Path(historic).is_file()
        mixed=group in {'match09','match10'}
        row=dict(rally_id=name,scope='primary_48' if name in EXPECTED else 'historical_validation_auxiliary',
            source_group=group,leakage_block=LINKED.get(group,group),video_path=video.as_posix(),clip_path=clip.as_posix(),
            meta_path=meta_path.as_posix(),labels_path=labels_path.as_posix(),frame_count=count,
            visible_labels=counts[1],invisible_labels=counts[0],unknown_labels=counts[None],
            labelled_frames=counts[1]+counts[0],width=values[0],height=values[1],fps=fps,
            current_split=m['split'],historical_video_path_exists=int(path_exists),video_sha256=sha256(video),
            meta_sha256=sha256(meta_path),labels_sha256=sha256(labels_path),
            v2_training_clip=int(name in train_ids),v2_validation_clip=int(name in val_ids),
            historical_exploratory_evaluation=int(name in EVALUATED),
            group_has_exploratory_evaluation=int(group in groups_evaluated),
            mapping_status='mixed_match_content' if mixed else 'visually_consistent_canonical_source_unverified',
            test_eligibility='candidate_pending_source_verification' if group in TEST_CANDIDATES else 'historical_training_or_development_or_linked_source',
            observed_source_evidence=evidence[name])
        rows.append(row)
        print(f'Inventory {name}: {group}, {count} frames; visible={counts[1]}, invisible={counts[0]}, unknown={counts[None]}',flush=True)
    return rows,run,issues


def content_audit(rows, output):
    videos=defaultdict(list);jpeg_hashes={};jpeg_duplicates=[];samples=[]
    for row in rows:
        videos[row['video_sha256']].append(row['rally_id'])
        clip=Path(row['clip_path']);m=json.loads(Path(row['meta_path']).read_text());t=m['transform']
        step=max(1,round(row['fps']))
        for frame in range(row['frame_count']):
            path=clip/'frames'/f'{frame:06d}.jpg'
            digest=sha256(path)
            old=jpeg_hashes.get(digest)
            if old and old[0]!=row['rally_id']:
                jpeg_duplicates.append(dict(first=list(old),second=[row['rally_id'],frame],sha256=digest))
            else:jpeg_hashes[digest]=(row['rally_id'],frame)
            if frame%step==0:
                image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                image=image[t['pad_y']:t['pad_y']+t['resized_height'],t['pad_x']:t['pad_x']+t['resized_width']]
                tiny=cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
                dct=cv2.dct(tiny)[:8,:8].flatten();median=np.median(dct[1:])
                phash=sum(int(v>median)<<i for i,v in enumerate(dct))
                samples.append(dict(rally_id=row['rally_id'],source_group=row['source_group'],frame=frame,
                                    seconds=frame/row['fps'],phash=f'{phash:016x}'))
        print(f'Content audit {row["rally_id"]}: JPEG hashes and 1 Hz perceptual samples complete',flush=True)
    pair_best={}
    for i,a in enumerate(samples):
        ah=int(a['phash'],16)
        for b in samples[i+1:]:
            if a['source_group']==b['source_group']:continue
            key=tuple(sorted((a['rally_id'],b['rally_id'])))
            distance=(ah^int(b['phash'],16)).bit_count()
            if key not in pair_best or distance<pair_best[key]['hamming_distance']:
                pair_best[key]=dict(first=a,second=b,hamming_distance=distance)
    near=sorted(pair_best.values(),key=lambda r:(r['hamming_distance'],r['first']['rally_id'],r['second']['rally_id']))
    result=dict(exact_duplicate_videos=[ids for ids in videos.values() if len(ids)>1],
                exact_duplicate_prepared_jpegs=jpeg_duplicates,perceptual_samples=len(samples),
                cross_group_nearest_pairs=near[:30],
                caution='Perceptual similarity is a review lead, not proof of duplicate rallies; negative results do not exclude recropping or shared matches.')
    write_json(output/'content_audit.json',result)
    write_json(output/'perceptual_fingerprints.json',samples)
    return result


def stage(rows,assignments,directory):
    directory.mkdir(parents=True,exist_ok=False)
    for row in rows:
        if row['scope']!='primary_48' or assignments[row['source_group']]=='test':continue
        target=directory/row['rally_id'];target.mkdir()
        meta=json.loads(Path(row['meta_path']).read_text())
        meta['video']=row['video_path']
        meta['split']=assignments[row['source_group']]
        write_json(target/'meta.json',meta)
        for name in ('labels.csv','frames'):
            original=Path(row['clip_path'])/name
            (target/name).symlink_to(os.path.relpath(original.resolve(),target.resolve()),target_is_directory=name=='frames')
    train=ShuttleDataset(directory,'train');val=ShuttleDataset(directory,'val')
    if train.groups&val.groups:raise ValueError('Staged group overlap')
    return dict(train_labels=len(train),val_labels=len(val),train_groups=sorted(train.groups),
                val_groups=sorted(val.groups),test_clips_excluded=True)


def validate_manifests(root):
    """Read-only revalidation; does not regenerate or replace audit results."""
    with (root/'dataset_manifest.csv').open(newline='') as file:
        rows=list(csv.DictReader(file))
    with (root/'split_manifest.csv').open(newline='') as file:
        split_rows=list(csv.DictReader(file))
    if len(rows)!=49 or {r['rally_id'] for r in rows}!=set(EXPECTED)|{'validation01'}:
        raise ValueError('Dataset manifest membership/duplicates disagree')
    if len(split_rows)!=48 or {r['rally_id'] for r in split_rows}!=set(EXPECTED):
        raise ValueError('Split manifest membership/duplicates disagree')
    numeric=('frame_count','visible_labels','invisible_labels','unknown_labels','labelled_frames',
             'v2_training_clip','v2_validation_clip')
    by_id={}
    for row in rows:
        for key in numeric:row[key]=int(row[key])
        for key in ('video_path','clip_path','meta_path','labels_path'):
            if Path(row[key]).is_absolute() or '..' in Path(row[key]).parts:
                raise ValueError('Manifest paths must be repository-relative')
        meta=json.loads(Path(row['meta_path']).read_text())
        labels=read_labels(row['labels_path'])
        if row['source_group']!=meta['source_group'] or row['frame_count']!=meta['frame_count']:
            raise ValueError('Manifest metadata has drifted')
        counts=[sum(r['visible']==state for r in labels.values()) for state in (1,0,None)]
        if counts!=[row[k] for k in ('visible_labels','invisible_labels','unknown_labels')]:
            raise ValueError('Manifest visibility counts have drifted')
        if sum(counts)!=row['frame_count'] or counts[0]+counts[1]!=row['labelled_frames']:
            raise ValueError('Labelled/unknown frame denominators disagree')
        for path_key,hash_key in [('video_path','video_sha256'),('meta_path','meta_sha256'),('labels_path','labels_sha256')]:
            if sha256(row[path_key])!=row[hash_key]:raise ValueError(f'Input hash changed: {row[path_key]}')
        by_id[row['rally_id']]=row
    assignment={}
    counts=defaultdict(int)
    for row in split_rows:
        original=by_id[row['rally_id']]
        for key in ('source_group','leakage_block','clip_path','video_path'):
            if row[key]!=original[key]:raise ValueError('Split/dataset manifests disagree')
        if int(row['labelled_frames'])!=original['labelled_frames'] or int(row['unknown_labels'])!=original['unknown_labels']:
            raise ValueError('Split visibility counts disagree')
        if row['test_independence_verified']!='0':raise ValueError('Current policy does not certify independent testing')
        group=row['source_group'];split=row['proposed_split']
        if assignment.setdefault(group,split)!=split:raise ValueError('A group crosses splits')
        counts[group]+=original['labelled_frames']
    validate_assignments(rows,assignment)
    forced={r['source_group'] for r in rows if r['v2_training_clip']}
    recomputed,totals,_=select_split(counts,forced,TEST_CANDIDATES,forced_val={'match09'})
    if assignment!=recomputed:raise ValueError('Manifest does not reproduce the frozen split policy')
    return dict(dataset_rows=len(rows),primary_rallies=len(split_rows),split_labelled_frames=totals,
                hashes_match=True,group_disjoint=True,independent_test_certified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--root',type=Path,default=Path('data/shuttle'))
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--stage',action='store_true')
    parser.add_argument('--validate-manifests',action='store_true')
    args=parser.parse_args()
    if args.validate_manifests:
        print(json.dumps(validate_manifests(args.root),indent=2))
        return
    if args.output is None or args.evidence is None:
        parser.error('--output and --evidence are required for a new audit')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'audit.json').exists():raise FileExistsError('Choose a fresh audit output')
    evidence=json.loads(args.evidence.read_text())
    rows,run,issues=inventory(args.root,evidence)
    protected={str(Path(r[k])):r[h] for r in rows for k,h in [('meta_path','meta_sha256'),('labels_path','labels_sha256'),('video_path','video_sha256')]}
    protected['outputs/shuttle_v2/best.pt']=sha256('outputs/shuttle_v2/best.pt')
    write_json(args.output/'protected_inputs_before.json',protected)
    content=content_audit(rows,args.output)
    group_counts=defaultdict(int)
    for r in rows:
        if r['scope']=='primary_48':group_counts[r['source_group']]+=r['labelled_frames']
    forced={r['source_group'] for r in rows if r['v2_training_clip']}
    assignments,totals,objective=select_split(group_counts,forced,TEST_CANDIDATES,forced_val={'match09'})
    validate_assignments(rows,assignments)
    # The historical validation group and its linked mixed source remain validation.
    if assignments['match09']!='val':
        raise ValueError('Unresolved historical-validation source link requires match09 in validation')
    for path in (args.root/'dataset_manifest.csv',args.root/'split_manifest.csv'):
        if path.exists():raise FileExistsError(path)
    with (args.root/'dataset_manifest.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=MANIFEST_FIELDS);writer.writeheader();writer.writerows(rows)
    split_rows=[]
    for r in rows:
        if r['scope']!='primary_48':continue
        split=assignments[r['source_group']]
        split_rows.append(dict(rally_id=r['rally_id'],source_group=r['source_group'],leakage_block=r['leakage_block'],
                               proposed_split=split,labelled_frames=r['labelled_frames'],unknown_labels=r['unknown_labels'],
                               clip_path=r['clip_path'],video_path=r['video_path'],
                               status='reserved_candidate_test_pending_source_verification' if split=='test' else 'proposed_development_split',
                               test_independence_verified=0,split_policy='v2_continuation_group_frame_balance_v1'))
    with (args.root/'split_manifest.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(split_rows[0]));writer.writeheader();writer.writerows(split_rows)
    staged=stage(rows,assignments,args.output/'staged_clips') if args.stage else None
    after={path:sha256(path) for path in protected}
    write_json(args.output/'protected_inputs_after.json',after)
    if after!=protected:raise ValueError('Protected input changed')
    result=dict(rows=rows,group_labelled_frames=dict(group_counts),assignments=assignments,split_labelled_frames=totals,
                target_ratios=dict(train=.7,val=.15,test=.15),objective='sum of squared labelled-frame fraction errors',
                objective_value=objective,forced_train_groups=sorted(forced),forced_validation_groups=['match09'],test_candidate_groups=sorted(TEST_CANDIDATES),
                source_independence_verified=False,originals_unchanged=True,training_run=run,
                duplicate_summary={k:v for k,v in content.items() if k!='cross_group_nearest_pairs'},staged_dataset=staged,issues=issues)
    write_json(args.output/'audit.json',result)
    print(json.dumps(dict(assignments=assignments,split_labelled_frames=totals,staged_dataset=staged),indent=2))


if __name__=='__main__':main()
