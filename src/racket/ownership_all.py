"""One ownership pass over all approved located centres, including partial clips."""
import argparse
from collections import Counter
import copy
import csv
from pathlib import Path
import shutil

from .ownership import OwnershipStore,prepare,status,export,separate,DEFAULT_ROOT
from .schema import digest,file_hash,load_clip,read_json,save_json,approved

ALL_ROOT=Path('data/racket_ownership_all_v005')


def all_status(root=ALL_ROOT):
    store=OwnershipStore(root);report=status(root)
    if store.manifest.get('workflow')!='all_existing_centres':raise ValueError('Use the consolidated all-existing ownership dataset')
    queue=store.queue();split_pending=Counter(store.tasks[i]['split'] for i in queue if store.row(i)['decision']=='pending')
    locked=store.manifest['locked_imported_reviews'];exclusions={r['id'] for r in store.manifest.get('carried_training_exclusions',[])}
    exportable=Counter(t['split'] for t in store.tasks.values() if store.row(t['id'])['decision']=='approved' and t['id'] not in exclusions)
    source_overview=[]
    for clip,info in store.clips.items():
        meta,ann=load_clip(info['source_clip']);reviewed=sum(approved(r,meta,ann['frames']) for r in ann['frames'].values())
        tasks=[t for t in store.tasks.values() if t['clip']==clip];decisions=Counter(store.row(t['id'])['decision'] for t in tasks)
        source_overview.append(dict(clip=clip,split=info['split'],source_group=info['source_group'],
            video_frames=meta['frame_count'],reviewed_source_frames=reviewed,located_source_centres=len(tasks),
            partial_source_clip=reviewed<meta['frame_count'],preserved_reviewed_ownership=sum(t['id'] in locked for t in tasks),
            approved_pairs=decisions['approved'],unlocatable=decisions['unlocatable'],quarantined=decisions['quarantined'],
            pending=decisions['pending']))
    report.update(workflow='all_existing_centres',imported_completed_reviews_locked=len(locked),
        imported_pending_drafts=store.manifest['consolidation']['imported_decisions'].get('pending',0),
        source_clips=len(store.clips),located_source_centres=len(store.tasks),
        pending_by_split={s:split_pending[s] for s in ['train','val']},queue_members=len(queue),
        remaining=sum(split_pending.values()),current_exportable_pairs=dict(exportable),
        prior_training_quality_exclusions=len(exclusions),per_clip_overview=source_overview,
        all_existing_reviews_complete=not sum(split_pending.values()),final_test_included=False,
        note='All existing approved located points are included. Original unknown/unreviewed frames remain unchanged; no points are invented. Ready/completed counts do not guarantee model accuracy.')
    return report


def consolidate(source=Path('data/racket_manual'),previous=DEFAULT_ROOT,output=ALL_ROOT,
                bundle=Path('outputs/racket_mediapipe/v002/bundle'),
                prior_review=Path('outputs/racket_ownership_v004/bundle/training_review_exclusions.json')):
    source,output=separate(source,output);previous=Path(previous).resolve()
    separate(previous,output)
    if output.exists():raise FileExistsError('Consolidated pass already exists; resume it without replacing work')
    old=OwnershipStore(previous)
    if Path(old.manifest['source_root']).resolve()!=source:raise ValueError('Previous ownership pass uses a different source annotation root')
    protected={str(p):file_hash(p) for folder in [source,previous] for p in folder.rglob('*.json')}
    old_rows={t['id']:copy.deepcopy(old.row(t['id'])) for t in old.tasks.values()
              if str(t['frame']) in old.labels[t['clip']]['frames']}
    prepare(source,output,bundle,per_clip=50,include_partial=True)
    manifest=read_json(output/'manifest.json');lookup={t['id']:t for t in manifest['tasks']}
    if not set(old.tasks)<=set(lookup):raise ValueError('A previously eligible source point disappeared from the all-existing pass')
    for identity,task in old.tasks.items():
        fresh=lookup[identity]
        if any(fresh[k]!=task[k] for k in ['clip','frame','split','source_group','locked_reference']):
            raise ValueError('Original ownership task identity changed: '+identity)
        # Keep exactly the old task/proposals for imported records and drafts.
        lookup[identity]=copy.deepcopy(task)
    manifest['tasks']=sorted(lookup.values(),key=lambda t:(t['split']!='train',t['source_group'],t['clip'],t['frame']))
    copied=[]
    for clip in old.clips:
        src=previous/'clips'/clip/'ownership.json'
        if src.is_file():
            dst=output/'clips'/clip/'ownership.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            if file_hash(src)!=file_hash(dst):raise ValueError('Ownership sidecar copy mismatch')
            copied.append(dict(clip=clip,source=str(src),destination=str(dst),sha256=file_hash(src)))
    locked={identity:digest(row) for identity,row in old_rows.items() if row['decision']!='pending'}
    carried=[]
    if Path(prior_review).is_file():
        for row in read_json(prior_review):
            identity=row['id']
            if identity not in old_rows or old.tasks[identity]['split']!='train' or old_rows[identity]['decision']!='approved':
                raise ValueError('Prior training exclusion no longer matches imported ownership work: '+identity)
            carried.append(dict(id=identity,reason=row['reason'],ownership_row_sha256=digest(old_rows[identity])))
    manifest.update(workflow='all_existing_centres',locked_imported_reviews=locked,carried_training_exclusions=carried,
        fixed_original_paired_validation_ids=old.manifest.get('fixed_original_paired_validation_ids',
            sorted(identity for identity,row in old_rows.items() if old.tasks[identity]['split']=='val' and row['decision']=='approved')),
        consolidation=dict(previous_dataset=str(previous),previous_manifest_sha256=file_hash(previous/'manifest.json'),
            imported_decisions=dict(Counter(r['decision'] for r in old_rows.values())),copied_sidecars=copied,
            includes_partial_clips=True,new_source_clips=sorted(set(c['clip'] for c in manifest['clips'])-set(old.clips)),
            original_sources_unchanged=True,previous_dataset_unchanged=True,
            supersedes_curated_round2_queue=True),
        policy='ALL approved located centres from complete or partial clips; completed imported ownership reviews locked; pending work continues; whole-match splits preserved; final test excluded')
    if any(file_hash(p)!=h for p,h in protected.items()):raise ValueError('Source/previous labels changed during consolidation; original files were not overwritten')
    save_json(output/'manifest.json',manifest)
    imported=OwnershipStore(output)
    if any(imported.row(identity)!=row for identity,row in old_rows.items()):raise ValueError('Imported ownership row changed')
    report=all_status(output)
    save_json(output/'preparation_report.json',report)
    save_json(output/'consolidation_report.json',report)
    save_json(output/'import_provenance.json',dict(protected_original_files=protected,copied_sidecars=copied))
    fields=list(report['per_clip_overview'][0])
    with (output/'overview.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(report['per_clip_overview'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--root',type=Path,default=ALL_ROOT)
    prep.add_argument('--source',type=Path,default=Path('data/racket_manual'));prep.add_argument('--previous',type=Path,default=DEFAULT_ROOT)
    prep.add_argument('--bundle',type=Path,default=Path('outputs/racket_mediapipe/v002/bundle'))
    label=sub.add_parser('label');label.add_argument('--root',type=Path,default=ALL_ROOT)
    label.add_argument('--clip');label.add_argument('--frame',type=int,help='Clip-local source frame within the unfinished-at-import queue')
    label.add_argument('--review',type=Path,help='Review only the saved training frames in an explicit correction queue')
    stat=sub.add_parser('status');stat.add_argument('--root',type=Path,default=ALL_ROOT)
    exp=sub.add_parser('export');exp.add_argument('--root',type=Path,default=ALL_ROOT);exp.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=consolidate(args.source,args.previous,args.root,args.bundle)
    elif args.command=='status':result=all_status(args.root)
    elif args.command=='export':result=export(args.root,args.output)
    else:
        from .ownership_labeler import OwnershipLabeler
        try:
            if OwnershipStore(args.root).manifest.get('workflow')!='all_existing_centres':
                raise ValueError('Use the consolidated all-existing root; the old ownership dataset is preserved separately')
            OwnershipLabeler(args.root,clip=args.clip,frame=args.frame,review=args.review).run()
        except ValueError as error:parser.error(str(error))
        return
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
