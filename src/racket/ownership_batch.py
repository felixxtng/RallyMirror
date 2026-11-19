"""Curated, training-only ownership queues; existing annotations stay untouched."""
import argparse
from collections import Counter,defaultdict
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from .ownership import OwnershipStore,iou
from .schema import digest,file_hash,read_json,save_json
from .mediapipe_bundle import square_bounds

FORMAT='racket_ownership_curated_batch_v001'


def usable_pair(task,row):
    """Require both approved targets inside the body's common training context.

    Mirrors real_owner_data.scene_bounds without importing a training runtime
    into the manual editor. Labels never determine the context's bounds.
    """
    if row['decision']!='approved':return False
    bounds=[square_bounds(p['bbox']) for p in row['players'].values()]
    left=min(b[0] for b in bounds);top=min(b[1] for b in bounds)
    right=max(b[0]+b[2] for b in bounds);bottom=max(b[1]+b[2] for b in bounds)
    side=math.ceil(max(right-left,bottom-top))
    x=math.floor((left+right-side)/2);y=math.floor((top+bottom-side)/2)
    point=task['locked_reference']['point']
    return all(x<=p[0]<x+side and y<=p[1]<y+side for p in [[point['x'],point['y']],row['other_point']])


def load_batch(store,path):
    return validate_batch(store,read_json(Path(path)))


def validate_batch(store,batch):
    if batch.get('format')!=FORMAT:raise ValueError('Unsupported curated ownership batch')
    if Path(batch['dataset_root']).resolve()!=store.root:raise ValueError('Batch belongs to a different ownership dataset')
    if file_hash(store.root/'manifest.json')!=batch['dataset_manifest_sha256']:raise ValueError('Curated source manifest changed')
    entries=batch['entries'];ids=[e['id'] for e in entries]
    if len(set(ids))!=len(ids) or not ids:raise ValueError('Empty or duplicated batch frame IDs')
    for identity,expected in batch['protected_existing_rows'].items():
        if identity not in store.tasks or digest(store.row(identity))!=expected:
            raise ValueError('Existing reviewed ownership label changed: '+identity)
    for entry in entries:
        task=store.tasks.get(entry['id'])
        if not task or task['split']!='train' or task['source_group'] not in batch['training_groups']:
            raise ValueError('Only registered TRAIN frames may enter this batch')
        if digest(task)!=entry['task_sha256']:raise ValueError('Frozen task changed: '+entry['id'])
        if entry['id'] in batch['protected_existing_rows'] or entry['decision_at_selection']!='pending':
            raise ValueError('Batch cannot reopen previously reviewed labels')
        if entry['stage'] not in ('primary','reserve'):raise ValueError('Unknown batch stage')
    return batch


def batch_status(store,batch):
    stages={s:Counter() for s in ['primary','reserve']};groups=defaultdict(Counter);clips=defaultdict(Counter)
    for entry in batch['entries']:
        t=store.tasks[entry['id']];row=store.labels[t['clip']]['frames'].get(str(t['frame']))
        decision=row['decision'] if row else 'pending'
        good=usable_pair(t,row) if row else False
        for counter in [stages[entry['stage']],groups[t['source_group']],clips[t['clip']]]:
            counter['total']+=1;counter[decision]+=1;counter['usable']+=int(good)
            counter['approved_outside_context']+=int(decision=='approved' and not good)
    usable=sum(v['usable'] for v in stages.values());primary_pending=stages['primary']['pending']
    deficits={g:max(0,batch['minimum_usable_per_group']-groups[g]['usable']) for g in batch['training_groups']}
    ready=not primary_pending and usable>=batch['target_usable_pairs'] and not any(deficits.values())
    return dict(batch=batch['name'],new_usable_pairs=usable,target_usable_pairs=batch['target_usable_pairs'],
        primary=dict(stages['primary']),reserve=dict(stages['reserve']),per_group={g:dict(v) for g,v in sorted(groups.items())},
        per_clip={c:dict(v) for c,v in sorted(clips.items())},group_deficits=deficits,
        ready_for_package=ready,existing_reviewed_rows_protected=len(batch['protected_existing_rows']),
        validation_included=False,test_included=False,
        note='Ready means annotation coverage only. Training/export quality checks and held-out accuracy must still pass.')


def pending_order(store,batch,queue=None):
    allowed=set(queue) if queue is not None else {e['id'] for e in batch['entries']}
    pending=[e for e in batch['entries'] if e['id'] in allowed and store.row(e['id'])['decision']=='pending']
    primary=[e['id'] for e in pending if e['stage']=='primary']
    if primary:return primary
    status=batch_status(store,batch)
    if status['ready_for_package']:return []
    reserve=[e for e in pending if e['stage']=='reserve']
    return [e['id'] for e in sorted(reserve,key=lambda e:(status['group_deficits'][store.tasks[e['id']]['source_group']]==0,
                                                         store.tasks[e['id']]['clip'],store.tasks[e['id']]['frame']))]


def _rank(values):
    values=np.asarray(values,float)
    # Equal-valued proxies must not acquire an arbitrary frame-order ranking.
    unique,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    mid=np.cumsum(counts)-counts/2
    return (mid[inverse]-.5)/max(1,len(values)-1)


def descriptors(store,cache):
    result=[]
    for clip,info in sorted(store.clips.items()):
        if info['split']!='train':continue
        tasks=sorted([t for t in store.tasks.values() if t['clip']==clip],key=lambda t:t['frame'])
        if len(tasks)<2:continue
        coords=np.array([[t['locked_reference']['point'][k] for k in ['x','y']] for t in tasks],float)
        times=np.array([t['frame']/info['meta']['fps'] for t in tasks]);vel=np.gradient(coords,times,axis=0)
        accel=np.gradient(vel,times,axis=0)
        for index,t in enumerate(tasks):
            # Exclude completed work and even partial stored drafts from a NEW batch.
            if str(t['frame']) in store.labels[clip]['frames']:continue
            if store.row(t['id'])['decision']!='pending':continue
            near,far=t['proposed_near_box'],t['proposed_far_box'];record=cache.get(t['id'],{})
            height=near[3]-near[1] if near else info['meta']['height']*.25
            entry=dict(id=t['id'],clip=clip,frame=t['frame'],source_group=t['source_group'],
                both_proposals=near is not None and far is not None,
                motion=float(np.linalg.norm(vel[index])/max(height,1)),
                acceleration=float(np.linalg.norm(accel[index])/max(height,1)),
                overlap_iou=iou(near,far) if near and far else 0.,
                far_height_fraction=(far[3]-far[1])/info['meta']['height'] if far else 1.,
                far_detail=None,cache_image=record.get('image'),far_roi_coverage=0.)
            if far and record.get('image') and record.get('crop'):
                im=cv2.imread(str(Path(cache['_root'])/record['image']),cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    x,y,side=record['crop'];scale=im.shape[0]/side;h=far[3]-far[1]
                    wanted=np.array([far[0]-.2*h,far[1]-.15*h,far[2]+.2*h,far[3]+.1*h])
                    unit=(wanted-np.array([x,y,x,y]))*scale
                    lo=np.maximum(np.floor(unit[:2]).astype(int),0);hi=np.minimum(np.ceil(unit[2:]).astype(int),im.shape[0])
                    if np.all(hi>lo):
                        entry['far_roi_coverage']=float(np.prod(hi-lo)/max(1,np.prod(unit[2:]-unit[:2])))
                        patch=im[lo[1]:hi[1],lo[0]:hi[0]]
                        if patch.size>=256 and entry['far_roi_coverage']>=.6:
                            entry['far_detail']=float(cv2.Laplacian(patch,cv2.CV_32F).var())
            result.append(entry)
    for clip in sorted({r['clip'] for r in result}):
        rows=[r for r in result if r['clip']==clip]
        motion=_rank([r['motion'] for r in rows]);accel=_rank([r['acceleration'] for r in rows])
        small=_rank([-r['far_height_fraction'] for r in rows])
        known=[r['far_detail'] for r in rows if r['far_detail'] is not None];default=float(np.median(known)) if known else 0.
        detail=_rank([-(r['far_detail'] if r['far_detail'] is not None else default) for r in rows])
        for i,r in enumerate(rows):
            r['motion_score']=float(.8*motion[i]+.2*accel[i]);r['detail_score']=float(.6*small[i]+.4*detail[i])
    return result


def _choose(rows,count,previous=(),stage='primary'):
    if len(rows)<count:raise ValueError('Not enough untouched candidate frames')
    selected=[];used=set();span=max(r['frame'] for r in rows)-min(r['frame'] for r in rows)+1
    centre=(min(r['frame'] for r in rows)+max(r['frame'] for r in rows))/2
    wanted_gap=max(2,span/max(count,1));prior=list(previous)
    def take(pool,n,field,category):
        for _ in range(n):
            remaining=[r for r in pool if r['id'] not in used]
            if not remaining:break
            def value(r):
                distance=min([abs(r['frame']-p) for p in prior]+[span/2])
                novelty=min(1.,distance/wanted_gap) if field else distance/span
                return ((r.get(field,0.) if field else 0.)+.75*novelty,-abs(r['frame']-centre),-r['frame'])
            pick=max(remaining,key=value);used.add(pick['id']);prior.append(pick['frame'])
            selected.append(dict(pick,category=category,stage=stage))
    overlap=[r for r in rows if r['overlap_iou']>=.03]
    take(overlap,min(round(count*.2),len(overlap)),'overlap_iou','overlap_proxy')
    take(rows,round(count*.25),'motion_score','fast_motion_proxy')
    take(rows,round(count*.2),'detail_score','small_or_low_detail_far_proxy')
    take(rows,count-len(selected),None,'temporal_coverage')
    return selected


def curate(root,cache_root,output,primary=700,reserve=140,target=550,minimum_per_group=50,
           baseline_bundle=Path('outputs/racket_ownership_v004/bundle')):
    store=OwnershipStore(root);output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing curation; choose a fresh batch filename')
    if min(primary,target)<1 or reserve<0 or target>primary+reserve:raise ValueError('Invalid frame counts')
    protected={t['id']:digest(store.row(t['id'])) for t in store.tasks.values()
        if str(t['frame']) in store.labels[t['clip']]['frames']}
    sidecars={str(p):file_hash(p) for p in (store.root/'clips').glob('*/ownership.json')}
    cached=read_json(Path(cache_root)/'manifest.json');cache={r['id']:r for r in cached['samples']};cache['_root']=str(cache_root)
    candidates=descriptors(store,cache);groups=sorted({r['source_group'] for r in candidates})
    if not groups:raise ValueError('No untouched training frames')
    if minimum_per_group<1 or minimum_per_group*len(groups)>target:raise ValueError('Per-match minimum exceeds the total usable target')
    selected=[];quotas={g:primary//len(groups)+(i<primary%len(groups)) for i,g in enumerate(groups)}
    for g in groups:
        clips=sorted({r['clip'] for r in candidates if r['source_group']==g})
        for i,clip in enumerate(clips):
            count=quotas[g]//len(clips)+(i<quotas[g]%len(clips))
            pool=[r for r in candidates if r['clip']==clip and r['both_proposals']]
            selected.extend(_choose(pool,count))
    primary_ids={r['id'] for r in selected};reserve_rows=[]
    # Balance reserves by match, respecting smaller available pools.
    for _ in range(reserve):
        eligible=[r for r in candidates if r['both_proposals'] and r['id'] not in primary_ids and r['id'] not in {x['id'] for x in reserve_rows}]
        if not eligible:raise ValueError('Not enough reserve frames with both body proposals')
        counts=Counter(r['source_group'] for r in reserve_rows)
        g=min({r['source_group'] for r in eligible},key=lambda group:(counts[group],group))
        pool=[r for r in eligible if r['source_group']==g]
        # Greatest temporal novelty relative to the selected frames in its clip.
        def novelty(r):
            used=[x['frame'] for x in selected+reserve_rows if x['clip']==r['clip']]
            return (min(abs(r['frame']-f) for f in used),r['motion_score']+r['detail_score'],-r['frame'])
        reserve_rows.append(dict(max(pool,key=novelty),stage='reserve',category='reserve_temporal_coverage'))
    selected=sorted(selected,key=lambda r:(r['clip'],r['frame']))+sorted(reserve_rows,key=lambda r:(r['clip'],r['frame']))
    for r in selected:
        r.update(task_sha256=digest(store.tasks[r['id']]),decision_at_selection='pending')
    exclusion_path=Path(baseline_bundle)/'training_review_exclusions.json'
    known_exclusions=read_json(exclusion_path) if exclusion_path.is_file() else []
    excluded_ids={r['id'] for r in known_exclusions}
    baseline={g:0 for g in groups}
    for t in store.tasks.values():
        if t['split']=='train' and t['id'] not in excluded_ids and usable_pair(t,store.row(t['id'])):baseline[t['source_group']]+=1
    batch=dict(format=FORMAT,name=output.stem,dataset_root=str(store.root),dataset_manifest_sha256=file_hash(store.root/'manifest.json'),
        protected_existing_rows=protected,training_groups=groups,target_usable_pairs=target,minimum_usable_per_group=minimum_per_group,
        primary_count=primary,reserve_count=reserve,entries=selected,baseline_usable_train_pairs=sum(baseline.values()),
        baseline_usable_per_group=baseline,known_prior_training_exclusions=known_exclusions,
        selection='Match balanced; proposal overlap, source-point motion, far size/detail proxies and temporal coverage',
        quality_caution='Proposals and image-detail scores are curation proxies, not verified ownership, blur or visibility labels. Human confirmation required.',
        includes_validation=False,includes_test=False,interpolation=False,model_proposed_racket_points=False,
        next_trial_target='95% within20 original-video pixels on held-out paired validation; measured, not guaranteed')
    # Validate every protection/split/task reference before publishing the batch.
    store.verify_sources();validate_batch(store,batch)
    if sidecars!={str(p):file_hash(p) for p in (store.root/'clips').glob('*/ownership.json')}:
        raise ValueError('Ownership labels changed during curation; no batch was published')
    output.parent.mkdir(parents=True,exist_ok=True);save_json(output,batch)
    fields=['id','stage','clip','frame','source_group','category','overlap_iou','motion','far_height_fraction','far_detail']
    with output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(selected)
    report=batch_status(store,batch)
    report.update(candidate_pool=len(candidates),primary_categories=dict(Counter(r['category'] for r in selected if r['stage']=='primary')),
        primary_overlap_proxy_005=sum(r['stage']=='primary' and r['overlap_iou']>=.05 for r in selected),
        primary_overlap_proxy_020=sum(r['stage']=='primary' and r['overlap_iou']>=.2 for r in selected),
        baseline_usable_train_pairs=sum(baseline.values()),projected_minimum_train_pairs=sum(baseline.values())+target)
    save_json(output.with_name(output.stem+'_report.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('curate');create.add_argument('--root',type=Path,default=Path('data/racket_ownership_v004'))
    create.add_argument('--cache',type=Path,default=Path('outputs/racket_mediapipe/v002/bundle'));create.add_argument('--output',type=Path,required=True)
    create.add_argument('--primary',type=int,default=700);create.add_argument('--reserve',type=int,default=140);create.add_argument('--target',type=int,default=550)
    create.add_argument('--minimum-per-group',type=int,default=50)
    args=parser.parse_args();print(__import__('json').dumps(curate(args.root,args.cache,args.output,args.primary,args.reserve,args.target,args.minimum_per_group),indent=2))


if __name__=='__main__':main()
