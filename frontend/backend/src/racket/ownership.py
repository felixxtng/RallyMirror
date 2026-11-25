"""Two-player ownership sidecars with immutable existing source annotations.

This workflow never writes the single-player annotation root. The new editor
only has setters for the other point and per-player ownership boxes. Existing
coordinates are hash-checked references, never editable fields in its output.
"""
from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from .schema import approved, digest, file_hash, group_id, load_clip, read_json, save_json

FORMAT = 'racket_real_ownership_v004'
DEFAULT_ROOT = Path('data/racket_ownership_v004')


def separate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Ownership output must be separate from the near-side annotation root')
    return source, output


def box_valid(box, width, height):
    return (isinstance(box, list) and len(box) == 4 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap / area if area else 0


def candidate_boxes(record, width, height):
    boxes = []
    for box in [record.get('owner_box')] + [p['bbox'] for p in record.get('candidates', [])]:
        if box_valid(box, width, height) and not any(iou(box, other) > .98 for other in boxes):
            boxes.append(copy.deepcopy(box))
    return boxes


def far_candidate(boxes, near_box):
    if not near_box or not boxes:
        return None
    h = near_box[3]-near_box[1]
    # Proposal ranking only. Every accepted pair requires explicit human box
    # confirmation; these heuristics do not establish athlete identity.
    choices = [box for box in boxes if iou(box, near_box) < .75 and box[3] < near_box[3]-.1*h
               and box[3]-box[1] >= .3*h]
    return min(choices, key=lambda box: abs((box[0]+box[2]-near_box[0]-near_box[2])/2)/h
               + abs((box[3]-near_box[3])/h) + .2*abs(math.log((box[3]-box[1])/h)), default=None)


def prepare(source=Path('data/racket_manual'), output=DEFAULT_ROOT,
            bundle=Path('outputs/racket_mediapipe/v002/bundle'), per_clip=50,include_partial=False):
    source, output = separate(source, output)
    if output.exists():
        raise FileExistsError('Ownership pass already exists; use label/status to resume, never re-prepare over it')
    if per_clip < 1:
        raise ValueError('per-clip must be positive')
    cache = {}
    bundle = Path(bundle)
    if (bundle/'manifest.json').is_file():
        cached = read_json(bundle/'manifest.json')
        cache = {r['id']: r for r in cached['samples']}
    clips = []; tasks = []; protected = {}; skipped = Counter()
    for path in sorted((source/'clips').glob('*/meta.json')):
        meta, ann = load_clip(path.parent)
        rows = ann['frames']
        reviewed = [f for f in range(meta['frame_count']) if approved(rows.get(str(f), {}), meta, rows)]
        if meta['split'] not in ('train', 'val'):
            skipped['test_or_other_clips'] += 1
            continue
        if len(reviewed) != meta['frame_count'] and not include_partial:
            skipped['incomplete_clips'] += 1
            continue
        if meta.get('keypoint_names') != ['head_center'] or meta.get('schema_version') != 1:
            raise ValueError('Expected existing single-centre source: '+path.parent.name)
        frames = [f for f in reviewed if rows[str(f)]['keypoints'][0]['v'] in (1, 2)]
        skipped['source_unknown_frames'] += len(reviewed)-len(frames)
        if not frames:
            continue
        if len(reviewed)!=meta['frame_count']:skipped['partial_clips_included']+=1
        pilot = {frames[i] for i in np.linspace(0, len(frames)-1, min(per_clip, len(frames)), dtype=int)}
        clips.append(dict(clip=path.parent.name, source_clip=str(path.parent), split=meta['split'],
            source_group=group_id(meta['source_group']), meta=copy.deepcopy(meta),
            eligible=len(frames), pilot=len(pilot)))
        for name in ['meta.json', 'annotations.json']:
            protected[str(path.parent/name)] = file_hash(path.parent/name)
        for frame in frames:
            row = rows[str(frame)];identity = f'{path.parent.name}_{frame:05d}'
            cached_row = cache.get(identity, {})
            # Stale detector proposals must not be attached to changed labels.
            if cached_row and (cached_row['frame'] != frame or
                    cached_row['target_original'] != [row['keypoints'][0]['x'], row['keypoints'][0]['y']]):
                cached_row = {}
            boxes = candidate_boxes(cached_row, meta['width'], meta['height'])
            # A legacy point's verified owner is not necessarily the near-side
            # person. Retain that box as a proposal, never equate it to NEAR.
            if box_valid(row.get('player_bbox'), meta['width'], meta['height']) and row['player_bbox'] not in boxes:
                boxes.append(copy.deepcopy(row['player_bbox']))
            near_box = copy.deepcopy(cached_row.get('owner_box'))
            if not box_valid(near_box, meta['width'], meta['height']):
                near_box = None
            tasks.append(dict(id=identity, clip=path.parent.name, frame=frame, split=meta['split'],
                source_group=group_id(meta['source_group']), pilot=frame in pilot,
                locked_reference=dict(point=copy.deepcopy(row['keypoints'][0]), row_sha256=digest(row),
                    provenance='locked existing annotation; not re-verified by preparation'),
                candidates=boxes, proposed_near_box=near_box, proposed_far_box=far_candidate(boxes, near_box)))
    groups = {split: {c['source_group'] for c in clips if c['split'] == split} for split in ('train', 'val')}
    if not all(groups.values()) or groups['train'] & groups['val']:
        raise ValueError('Need complete train and validation clips from disjoint source matches')
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError('Near-side source changed while preparing; nothing was written')
    output.mkdir(parents=True)
    manifest = dict(format=FORMAT, source_root=str(source), source_hashes=protected,
        coordinate_system='original_pixels', frame_indexing='clip_local_zero_based', split_unit='whole_source_match',
        identities='near_side/far_side within each source frame; optional tracker IDs can be attached later',
        source_labels_read_only=True, interpolation=False, clips=clips, tasks=tasks,
        skipped=dict(skipped), policy=('Approved located centres from complete AND partial clips' if include_partial else
            'Complete clips with located existing centres')+'; owner must be assigned explicitly per frame; final test excluded')
    save_json(output/'manifest.json', manifest)
    result = status(output)
    save_json(output/'preparation_report.json', result)
    return result


class OwnershipStore:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve();self.manifest = read_json(self.root/'manifest.json')
        if self.manifest.get('format') != FORMAT or not self.manifest.get('source_labels_read_only'):
            raise ValueError('Use the prepared locked-near ownership dataset')
        separate(self.manifest['source_root'], self.root)
        self.verify_sources()
        self.tasks = {r['id']: r for r in self.manifest['tasks']}
        if len(self.tasks) != len(self.manifest['tasks']):
            raise ValueError('Duplicate frame IDs')
        self.clips = {r['clip']: r for r in self.manifest['clips']};self.labels = {}
        groups = {split:{r['source_group'] for r in self.clips.values() if r['split']==split} for split in ['train','val']}
        if groups['train'] & groups['val']:
            raise ValueError('Source matches overlap between training and validation')
        originals = {clip:load_clip(r['source_clip']) for clip,r in self.clips.items()}
        for task in self.tasks.values():
            meta,ann = originals[task['clip']]
            row = ann['frames'][str(task['frame'])]
            if (task['locked_reference']['row_sha256'] != digest(row) or
                    task['locked_reference']['point'] != row['keypoints'][0] or
                    not approved(row,meta,ann['frames']) or row['keypoints'][0]['v'] not in (1,2) or
                    task['split'] != meta['split'] or task['source_group'] != group_id(meta['source_group']) or
                    self.clips[task['clip']]['meta'] != meta):
                raise ValueError('Frozen existing reference or source identity mismatch: '+task['id'])
        for clip in self.clips:
            path = self.root/'clips'/clip/'ownership.json'
            self.labels[clip] = read_json(path) if path.is_file() else dict(format=FORMAT, clip=clip, frames={})
            for key, row in self.labels[clip]['frames'].items():
                identity = f'{clip}_{int(key):05d}'
                if identity not in self.tasks:
                    raise ValueError('Unknown frame in far-side sidecar: '+identity)
                self.validate(self.tasks[identity], row)
        for identity,expected in self.manifest.get('locked_imported_reviews',{}).items():
            if identity not in self.tasks or self.row(identity)['decision']=='pending' or digest(self.row(identity))!=expected:
                raise ValueError('Completed imported ownership review changed: '+identity)
        for exclusion in self.manifest.get('carried_training_exclusions',[]):
            identity=exclusion['id']
            if (identity not in self.tasks or self.tasks[identity]['split']!='train' or
                    digest(self.row(identity))!=exclusion['ownership_row_sha256']):
                raise ValueError('Carried training exclusion does not match the frozen ownership row: '+identity)

    def verify_sources(self, clip=None):
        for path, expected in self.manifest['source_hashes'].items():
            if clip is None or Path(path).parent.name == clip:
                if file_hash(path) != expected:
                    raise ValueError('LOCKED existing source changed externally: '+path+
                                     '. Stop and reconcile using a new dataset version; this editor will not overwrite it.')

    def row(self, identity):
        t = self.tasks[identity]
        existing = self.labels[t['clip']]['frames'].get(str(t['frame']))
        if existing:
            return copy.deepcopy(existing)
        return dict(frame=t['frame'], locked_reference_sha256=t['locked_reference']['row_sha256'], decision='pending',
            locked_owner=None, other_point=None,
            players={side:dict(bbox=copy.deepcopy(t['proposed_'+name+'_box']), ownership_verified=False, track_id=None)
                     for side,name in [('near_side','near'),('far_side','far')]}, locked_reference_issue=None)

    def validate(self, task, row):
        if set(row) != {'frame','locked_reference_sha256','decision','locked_owner','other_point','players','locked_reference_issue'}:
            raise ValueError('Only the other point and ownership associations can be saved; existing coordinates are locked')
        if row['frame'] != task['frame'] or row['locked_reference_sha256'] != task['locked_reference']['row_sha256']:
            raise ValueError('Near reference/frame cannot be replaced')
        if row['decision'] not in ('pending','approved','unlocatable','quarantined'):
            raise ValueError('Unknown review decision')
        if row['locked_owner'] not in (None,'near_side','far_side'):
            raise ValueError('Assign the existing point owner explicitly: N=near, F=far')
        if set(row['players']) != {'near_side','far_side'}:
            raise ValueError('Expected both player ownership records')
        meta = self.clips[task['clip']]['meta'];w,h = meta['width'],meta['height']
        for key in ('near_side','far_side'):
            player = row['players'][key]
            allowed = {'bbox','ownership_verified','track_id'}
            if set(player) != allowed or not isinstance(player['ownership_verified'],bool):
                raise ValueError('Invalid per-player ownership record')
            if player['bbox'] is not None and not box_valid(player['bbox'],w,h):
                raise ValueError('Player box must be within the original image')
            track = player['track_id']
            if track is not None and (isinstance(track,bool) or not isinstance(track,(int,str)) or track == ''):
                raise ValueError('Tracker ID must be null, integer or nonempty string')
        point = row['other_point']
        if point is not None and (not isinstance(point,list) or len(point) != 2 or
                not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in point) or
                not (0 <= point[0] < w and 0 <= point[1] < h)):
            raise ValueError('Other string-bed centre must be a finite point in original pixels')
        if point is not None and row['locked_owner'] is None:
            raise ValueError('Assign the locked point owner with N or F before labelling the other racket')
        if row['decision'] == 'approved':
            if row['locked_owner'] is None or point is None or not all(p['bbox'] and p['ownership_verified'] for p in row['players'].values()):
                raise ValueError('Assign locked owner (N/F), click OTHER centre and verify BOTH boxes before accepting')
            if iou(row['players']['far_side']['bbox'],row['players']['near_side']['bbox']) > .95:
                raise ValueError('Both boxes identify almost the same region; draw distinct player body boxes')
            ref=task['locked_reference']['point']
            if math.dist(point,[ref['x'],ref['y']]) < 3:
                raise ValueError('Both centres coincide: click the OTHER racket, or exclude an ambiguous pair')
        if row['decision'] in ('unlocatable','quarantined') and point is not None:
            raise ValueError('Excluded/unknown samples must not contain an accepted far-side point')

    def save(self, identity, row):
        task = self.tasks[identity]
        self.verify_sources(task['clip'])
        self.validate(task,row)
        expected=self.manifest.get('locked_imported_reviews',{}).get(identity)
        if expected:
            if digest(row)!=expected:raise ValueError('Completed imported ownership review is locked; this queue only edits unfinished work')
            return
        clip = task['clip'];data = copy.deepcopy(self.labels[clip])
        data['frames'][str(task['frame'])] = copy.deepcopy(row)
        save_json(self.root/'clips'/clip/'ownership.json',data)
        self.labels[clip] = data

    def queue(self, pilot=False, clip=None):
        if clip and clip not in self.clips:
            raise ValueError('Clip is not in this completed-clip ownership pass: '+clip)
        locked=self.manifest.get('locked_imported_reviews',{})
        return [t['id'] for t in self.manifest['tasks'] if t['id'] not in locked and
                (not pilot or t['pilot']) and (clip is None or t['clip'] == clip)]


def status(root=DEFAULT_ROOT):
    store = OwnershipStore(root);report = dict(format=FORMAT, source_hashes_verified=True, source_labels_read_only=True,
        all=Counter(),pilot=Counter(),per_clip={},source_root=store.manifest['source_root'],skipped=store.manifest['skipped'])
    for t in store.tasks.values():
        decision = store.row(t['id'])['decision'];clip = t['clip']
        counts = report['per_clip'].setdefault(clip,dict(split=t['split'],source_group=t['source_group'],all=Counter(),pilot=Counter()))
        for key,enabled in [('all',True),('pilot',t['pilot'])]:
            if enabled:
                for counter in [report[key],counts[key]]:
                    counter['total'] += 1;counter[decision] += 1
                report[key][t['split']] += 1
    return report


def import_v003(legacy_root, root=DEFAULT_ROOT):
    """Retain old work without carrying its forced-near ownership assumption.

    Coordinates/drafts are archived verbatim; accepted or excluded v003 rows
    reopen pending because their roles require a new explicit decision. Existing
    v004 work is never overwritten. No source or v003 files are changed.
    """
    store=OwnershipStore(root);legacy_root=Path(legacy_root)
    old=read_json(legacy_root/'manifest.json')
    if old['format']!='racket_real_ownership_v003' or Path(old['source_root']).resolve()!=Path(store.manifest['source_root']).resolve():
        raise ValueError('Legacy ownership pass does not refer to the same source dataset')
    backup=store.root/'legacy_v003_import.json'
    if backup.exists():raise FileExistsError('Legacy import already recorded; do not overwrite it')
    originals=[];imported=0
    for path in sorted((legacy_root/'clips').glob('*/far_side.json')):
        data=read_json(path);originals.append(dict(path=str(path.resolve()),sha256=file_hash(path),data=data))
    # Save complete original draft coordinates before creating new pending rows.
    save_json(backup,dict(legacy_root=str(legacy_root.resolve()),originals=originals,
        note='Verbatim legacy work retained. Old forced-near role is unverified. Reassign locked owner and label the other racket.'))
    for entry in originals:
        clip=entry['data']['clip']
        for frame,old_row in entry['data']['frames'].items():
            identity=f'{clip}_{int(frame):05d}'
            if identity not in store.tasks or str(frame) in store.labels[clip]['frames']:continue
            row=store.row(identity)
            for side,field in [('near_side','near_association'),('far_side','far')]:
                row['players'][side].update(bbox=copy.deepcopy(old_row[field]['bbox']),track_id=old_row[field]['track_id'])
            # Do not treat the old far click as the other racket before knowing
            # whether the locked point itself already belongs to FAR.
            store.save(identity,row);imported+=1
    if any(file_hash(e['path'])!=e['sha256'] for e in originals):raise ValueError('Legacy source changed during import')
    return dict(imported_pending=imported,verbatim_legacy_backup=str(backup))


def export(root, output):
    store = OwnershipStore(root);output = Path(output).resolve()
    source, _ = separate(store.manifest['source_root'], output)
    if output == store.root or store.root in output.parents or output in store.root.parents:
        raise ValueError('Export to a separate sibling/output folder')
    if output.exists():
        raise FileExistsError('Preserve prior exports; choose a fresh directory')
    rows = [];excluded = Counter();exclusion_rows=[]
    groups = {'train':set(),'val':set()}
    carried={r['id']:r for r in store.manifest.get('carried_training_exclusions',[])}
    for t in store.tasks.values():
        row = store.row(t['id'])
        if t['id'] in carried:
            excluded['prior_training_quality_review']+=1
            exclusion_rows.append(dict(id=t['id'],clip=t['clip'],frame=t['frame'],split=t['split'],
                source_group=t['source_group'],reason='prior_training_quality_review',detail=carried[t['id']]['reason'],
                exclude_original_source_row_from_future_training=False))
            continue
        if row['decision'] != 'approved':
            excluded[row['decision']] += 1
            exclusion_rows.append(dict(id=t['id'],clip=t['clip'],frame=t['frame'],split=t['split'],
                source_group=t['source_group'],reason=row['decision'],locked_reference_issue=row['locked_reference_issue'],
                exclude_original_source_row_from_future_training=row['decision']=='quarantined'))
            continue
        store.validate(t,row);meta=store.clips[t['clip']]['meta']
        players={}
        for side in ['near_side','far_side']:
            is_locked=side==row['locked_owner']
            point=copy.deepcopy(t['locked_reference']['point']) if is_locked else dict(name='head_center',x=row['other_point'][0],y=row['other_point'][1],v=2)
            players[side]=dict(head_center=point,**row['players'][side],point_origin='locked existing manual annotation' if is_locked else 'new manual other-racket annotation')
        rows.append(dict(format=FORMAT,id=t['id'],clip=t['clip'],frame=t['frame'],split=t['split'],source_group=t['source_group'],
            source=meta['source'],source_hash=meta['source_hash'],start_frame=meta.get('start_frame',0),
            original_size=[meta['width'],meta['height']],coordinate_system='original_pixels',
            locked_reference_sha256=t['locked_reference']['row_sha256'],
            locked_owner=row['locked_owner'],players=players))
        groups[t['split']].add(t['source_group'])
    if not rows:
        raise ValueError('No verified, located two-racket pairs to export yet')
    if groups['train'] & groups['val']:
        raise ValueError('A source match appears in both splits')
    store.verify_sources();output.mkdir(parents=True)
    for split in ['train','val']:
        with (output/(split+'.jsonl')).open('w',encoding='utf-8') as f:
            for row in rows:
                if row['split'] == split:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (output/'exclusions.jsonl').open('w',encoding='utf-8') as f:
        for row in exclusion_rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(format=FORMAT,counts=dict(Counter(r['split'] for r in rows)),excluded=dict(excluded),
        purpose='Verified real ownership pairs; recompute MediaPipe separately for each verified body box in the next training package',
        source_hashes_verified=True,unknown_other_racket_is_not_absence=True,test_included=False,
        not_directly_a_v002_training_bundle=True)
    save_json(output/'export.json',report)
    return report
