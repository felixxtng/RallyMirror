"""Diverse sampled ownership annotations; originals and historical benchmarks stay frozen."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .ownership import OwnershipStore, box_valid, iou, separate
from .schema import digest, file_hash, group_id, load_clip, read_json, save_json
from .source import FrameSource

FORMAT = 'racket_sampled_ownership_v006'
ROOT = Path('data/racket_ownership_sampled_v006')
SIDES = ('near_side', 'far_side')
KNOWN = [
    ('rally34_00186', 'far_side', 'Wrong person: seated spectator, not far player.'),
    ('rally43_00074', 'far_side', 'Wrong person: umpire, not far player.'),
    ('rally34_00214', 'far_side', 'Wrong person: duplicate near-player detection assigned to far.'),
    ('rally09_00064', 'far_side', 'Previously excluded: seated official, not far player.'),
    ('rally25_00296', 'far_side', 'Previously excluded: umpire, not far player.'),
    ('rally31_00004', 'far_side', 'Previously excluded: point outside body-derived context; inspect body/point, X if ambiguous.'),
]


def initial_row(task):
    row = dict(decision='pending', players={}, actions=[])
    for side in SIDES:
        ref = task.get('references', {}).get(side)
        row['players'][side] = dict(
            point=copy.deepcopy(ref), racket_state='located' if ref else 'pending',
            bbox=copy.deepcopy(task['proposals'].get(side)),
            body_state='verified' if side in task.get('retained_verified_sides', []) else 'pending')
    return row


def steps(row):
    """Unknown rackets are not negatives; bodies are checked unless both rackets are H."""
    if row['decision'] == 'excluded':
        return []
    result = [(s, 'racket') for s in SIDES if row['players'][s]['racket_state'] == 'pending']
    result += [(s, 'body') for s in SIDES if row['players'][s]['body_state'] == 'pending']
    return result


class SampleStore:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        if self.manifest.get('format') != FORMAT:
            raise ValueError('Not a sampled ownership queue')
        self.tasks = {t['id']: t for t in self.manifest['tasks']}
        if len(self.tasks) != len(self.manifest['tasks']):
            raise ValueError('Repeated source frame')
        self.clips = self.manifest['clips']
        groups = {}
        for task in self.tasks.values():
            c = self.clips[task['clip']]
            if (task['split'] != c['meta']['split'] or task['source_group'] != group_id(c['meta']['source_group'])
                    or task['split'] not in ('train', 'val')
                    or (task['split'] == 'val' and task['kind'] != 'correction')):
                raise ValueError('Split/identity changed')
            groups.setdefault(task['source_group'], set()).add(task['split'])
        if any(len(v) > 1 for v in groups.values()):
            raise ValueError('Source match crosses splits')
        self.verify()
        self.rows = {}
        for clip in self.clips:
            path = self.root/'labels'/f'{clip}.json'
            if path.exists():
                for identity, row in read_json(path)['rows'].items():
                    if identity not in self.tasks or self.tasks[identity]['clip'] != clip:
                        raise ValueError('Unknown saved sample')
                    self.validate(self.tasks[identity], row)
                    self.rows[identity] = row

    def verify(self):
        for path, expected in self.manifest['protected_sources'].items():
            if file_hash(path) != expected:
                raise ValueError('Protected source changed; reconcile in a new version: '+path)

    def row(self, identity):
        return copy.deepcopy(self.rows.get(identity, initial_row(self.tasks[identity])))

    def validate(self, task, row):
        if row['decision'] not in ('pending', 'complete', 'excluded') or set(row['players']) != set(SIDES):
            raise ValueError('Invalid decision/players')
        m = self.clips[task['clip']]['meta']
        for side in SIDES:
            p = row['players'][side]
            if p['racket_state'] not in ('pending', 'located', 'unlocatable'):
                raise ValueError('Invalid racket state')
            if p['body_state'] not in ('pending', 'verified', 'unlocatable', 'not_required'):
                raise ValueError('Invalid body state')
            point = p['point']
            if p['racket_state'] == 'located':
                if (not isinstance(point, list) or len(point) != 2 or
                        not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point)
                        or not (0 <= point[0] < m['width'] and 0 <= point[1] < m['height'])):
                    raise ValueError('Point must be inside original image')
            elif point is not None:
                raise ValueError('Unknown racket must not contain an invented point')
            if p['bbox'] is not None and not box_valid(p['bbox'], m['width'], m['height']):
                raise ValueError('Body box must be valid original pixels')
            if p['body_state'] == 'verified' and p['bbox'] is None:
                raise ValueError('Verified body requires a box')
            if side in task.get('references', {}):
                override = p.get('reference_override')
                if override is not None and override != dict(
                        original_point=task['references'][side], provenance='explicit_manual_correction',
                        original_reference_sha256=digest(dict(id=task['id'], side=side, point=task['references'][side]))):
                    raise ValueError('Manual correction must preserve the exact original reference provenance')
                if (point != task['references'][side] or p['racket_state'] != 'located') and override is None:
                    raise ValueError('Existing point is locked; X excludes without editing it')
            elif p.get('reference_override') is not None:
                raise ValueError('Cannot attach an original reference to an unrelated new point')
        if row['decision'] == 'complete':
            if steps(row):
                raise ValueError('Unfinished actions cannot be complete')
            a, b = [row['players'][s] for s in SIDES]
            if all(p['body_state'] == 'verified' for p in (a, b)) and iou(a['bbox'], b['bbox']) > .95:
                raise ValueError('Boxes select the same person. Correct one box or H for an unlocatable body.')
            if all(p['racket_state'] == 'located' for p in (a, b)) and math.dist(a['point'], b['point']) < 3:
                raise ValueError('Both points select the same racket; Backspace undoes the last action.')

    def save(self, identity, row):
        task = self.tasks[identity]
        self.validate(task, row)
        # Only new sidecars are written; no legacy rows are ever modified.
        clip = task['clip']
        data = {k: v for k, v in self.rows.items() if self.tasks[k]['clip'] == clip}
        data[identity] = copy.deepcopy(row)
        save_json(self.root/'labels'/f'{clip}.json', dict(format=FORMAT, rows=data))
        self.rows[identity] = copy.deepcopy(row)

    def status(self):
        result = dict(total=len(self.tasks), complete=0, excluded=0, pending=0, per_clip={}, positive_queries=0,
                      paired_frames=0, one_sided_frames=0, unknown_rackets=0, validation_corrections=0,
                      target=dict(metric='PCK20 including misses', near_at_least=.85, far_at_least=.85,
                                  scope='held-out source matches, each side separately; not full-video visibility recall'))
        for t in self.tasks.values():
            row = self.row(t['id'])
            result[row['decision']] += 1
            c = result['per_clip'].setdefault(t['clip'], dict(split=t['split'], total=0, complete=0, excluded=0, pending=0))
            c['total'] += 1
            c[row['decision']] += 1
            if row['decision'] != 'complete':
                continue
            n = sum(p['racket_state'] == 'located' and p['body_state'] == 'verified' for p in row['players'].values())
            result['positive_queries'] += n
            result['paired_frames'] += n == 2
            result['one_sided_frames'] += n == 1
            result['unknown_rackets'] += sum(p['racket_state'] == 'unlocatable' for p in row['players'].values())
            result['validation_corrections'] += t['split'] == 'val'
        result['milestones'] = {}
        for limit in (500, 1000):
            ids = list(self.tasks)[:limit]
            result['milestones'][str(limit)] = dict(
                frames=len(ids), reviewed=sum(self.row(k)['decision'] != 'pending' for k in ids),
                training_rallies=len({self.tasks[k]['clip'] for k in ids if self.tasks[k]['split'] == 'train'}))
        return result


def farthest_samples(frames, features, count, frame_count):
    """Deterministic image + temporal diversity, with an anti-adjacency guard."""
    if count > len(frames):
        raise ValueError('Not enough distinct candidate frames')
    x = np.asarray(features, np.float32)
    scale = np.maximum(x.std(0), .025)
    x = np.clip((x-x.mean(0))/scale, -3, 3) / math.sqrt(x.shape[1])
    time = np.asarray(frames, float)/max(1, frame_count-1)
    x = np.c_[x*.7, time*2.]
    chosen = [int(np.argmin(abs(time-.5)))]
    distance = np.full(len(frames), np.inf)
    gap = max(2, min(8, frame_count//max(1, count*3)))
    while len(chosen) < count:
        distance = np.minimum(distance, ((x-x[chosen[-1]])**2).sum(1))
        scores = distance.copy()
        for i in chosen:
            scores[abs(np.asarray(frames)-frames[i]) < gap] = -1
        scores[chosen] = -np.inf
        if scores.max() < 0:
            scores = distance.copy()
            scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
    return [frames[i] for i in chosen]


def image_feature(image):
    h, w = image.shape[:2]
    # Court/action area, avoiding most scoreboard and audience pixels.
    crop = image[round(.22*h):round(.94*h), round(.08*w):round(.92*w)]
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (16, 12)).astype(np.float32)/255.
    return gray.flatten().tolist()


def court_coverage(image):
    """Broad floor coverage on red/green/grey courts; not an owner classifier."""
    thumb = cv2.resize(image, (160, 90))
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)[45:87, 12:148]
    hue, sat, val = cv2.split(hsv)
    color = (((hue <= 12) | (hue >= 170)) | ((hue >= 32) & (hue <= 90)))
    floor = (color & (sat > 65) & (val > 40)) | ((sat < 40) & (val > 50) & (val < 210))
    return float(floor.mean())


def read_selected(meta, frames):
    """Decode a video once, retrieving pixels only for selected source indices."""
    wanted = set(frames)
    stream = FrameSource(meta)
    try:
        if stream.capture is None:
            for frame in sorted(wanted):
                yield frame, stream.read(frame)
            return
        start = meta.get('start_frame', 0)
        if start:
            stream.capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame in range(max(wanted)+1):
            if not stream.capture.grab():
                raise ValueError(f'Cannot decode source frame {frame}')
            if frame in wanted:
                ok, image = stream.capture.retrieve()
                if not ok or image.shape[:2] != (meta['height'], meta['width']):
                    raise ValueError(f'Cannot retrieve source frame {frame}')
                yield frame, image
    finally:
        stream.close()


def proposed_bodies(candidates, w, h):
    # Proposals only: nobody is verified by this heuristic.
    plausible = [d for d in candidates if d['confidence'] >= .25
                 and .035*h < d['bbox'][3]-d['bbox'][1] < .9*h
                 and .12*w < (d['bbox'][0]+d['bbox'][2])/2 < .88*w
                 and .18*h < d['bbox'][3] < .99*h]
    plausible.sort(key=lambda d: d['bbox'][3], reverse=True)
    near = plausible[0]['bbox'] if plausible else None
    remaining = [d['bbox'] for d in plausible[1:] if iou(near, d['bbox']) < .7]
    far = remaining[0] if remaining else None
    return dict(near_side=near, far_side=far)


def prepare(output=ROOT, total=1000, source=Path('data/racket_manual'),
            previous=Path('data/racket_ownership_all_v005'), weights=Path('yolo11n.pt')):
    source, output = separate(source, output)
    previous, _ = separate(previous, output)
    if output.exists():
        raise FileExistsError('Queue already exists; resume with label, do not replace it')
    if not 500 <= total <= 1000:
        raise ValueError('Choose 500 to 1000 distinct frames')
    old = OwnershipStore(previous)
    cv2.setNumThreads(1)
    protected = {str(p.resolve()): file_hash(p) for folder in (source, previous) for p in folder.rglob('*.json')}
    clips = {}
    for path in sorted((source/'clips').glob('*/meta.json')):
        meta, _ = load_clip(path.parent)
        if meta['split'] in ('train', 'val'):
            clips[path.parent.name] = dict(meta=meta, source_clip=str(path.parent.resolve()))
    train = {c: v for c, v in clips.items() if v['meta']['split'] == 'train'}
    val_groups = {group_id(c['meta']['source_group']) for c in clips.values() if c['meta']['split'] == 'val'}
    if any(group_id(c['meta']['source_group']) in val_groups for c in train.values()):
        raise ValueError('Match leakage')
    tasks = []
    for identity, side, reason in KNOWN:
        if identity not in old.tasks:
            continue
        t = old.tasks[identity]
        row = old.row(identity)
        ref = t['locked_reference']['point']
        refs = {row['locked_owner']: [ref['x'], ref['y']]}
        if row['other_point'] is not None:
            refs[SIDES[1] if row['locked_owner'] == SIDES[0] else SIDES[0]] = row['other_point']
        tasks.append(dict(id=identity, clip=t['clip'], frame=t['frame'], split=t['split'], source_group=t['source_group'],
            kind='correction', reason=reason, references=refs, candidates=copy.deepcopy(t['candidates']),
            proposals={s: copy.deepcopy(row['players'][s]['bbox']) for s in SIDES},
            retained_verified_sides=[s for s in SIDES if s != side and row['players'][s]['ownership_verified']],
            previous_row_sha256=digest(row), correction_side=side))
    fixed_ids = {t['id'] for t in tasks}
    available = {c: [f for f in range(v['meta']['frame_count'])
                     if f'{c}_{f:05d}' not in old.tasks and f'{c}_{f:05d}' not in fixed_ids]
                 for c, v in train.items()}
    # A floor per rally, then allocate remaining slots toward previously untouched footage.
    wanted = total-len(tasks)
    floor = min(10, wanted//len(train))
    quotas = {c: min(floor, len(v)) for c, v in available.items()}
    while sum(quotas.values()) < wanted:
        candidates = [c for c in train if quotas[c] < len(available[c])]
        if not candidates:
            raise ValueError('Not enough novel source frames; never repeat completed labels silently')
        c = max(candidates, key=lambda k: (len(available[k])**.5/(quotas[k]+1), k))
        quotas[c] += 1
    descriptors = {}
    pools = {}
    cache = Path('outputs/racket_sampled_v006_candidates').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for clip, info in train.items():
        meta = info['meta']
        pool = available[clip]
        indices = np.unique(np.linspace(0, len(pool)-1, min(len(pool), quotas[clip]*6)).round().astype(int))
        wanted_frames = {pool[i] for i in indices}
        cache_path = cache/(clip+'.json')
        signature = digest(dict(meta=meta, wanted=sorted(wanted_frames), preprocessing='floor_v3'))
        cached = read_json(cache_path) if cache_path.exists() else {}
        if cached.get('signature') != signature:
            scanned = []
            for frame, image in read_selected(meta, wanted_frames):
                thumb = cv2.resize(image, (640, round(image.shape[0]*640/image.shape[1])))
                path = cache/f'{clip}_{frame:05d}.jpg'
                cv2.imwrite(str(path), thumb)
                scanned.append(dict(frame=frame, coverage=court_coverage(image), feature=image_feature(image),
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    thumbnail_sha256=hashlib.sha256(cv2.resize(image, (320, 180)).tobytes()).hexdigest(),
                    thumbnail=str(path)))
            cached = dict(signature=signature, records=scanned)
            save_json(cache_path, cached)
        frames, features, hashes, rejected = [], [], set(), 0
        for record in cached['records']:
            if record['coverage'] < .5:
                rejected += 1
                continue
            image_hash = record['thumbnail_sha256']
            if image_hash in hashes:
                continue
            hashes.add(image_hash)
            frames.append(record['frame']);features.append(record['feature'])
        pools[clip] = dict(frames=frames, features=features, records={r['frame']: r for r in cached['records']})
        quotas[clip] = min(quotas[clip], len(frames))
        descriptors[clip] = dict(pool=len(frames), rejected_cutaways=rejected)
        print(f'Court-view candidate pool {clip}: {len(frames)}; rejected {rejected} cutaways', flush=True)
    while sum(quotas.values()) < wanted:
        candidates = [c for c in train if quotas[c] < len(pools[c]['frames'])]
        if not candidates:
            raise ValueError('Too few novel court views for this target. Expand candidate pools before proceeding.')
        c = max(candidates, key=lambda k: len(pools[k]['frames'])**.5/(quotas[k]+1))
        quotas[c] += 1
    for clip, info in train.items():
        frames, features = pools[clip]['frames'], pools[clip]['features']
        if not frames:
            raise ValueError('No novel court views in '+clip+'; do not fill its quota with spectators')
        picked = farthest_samples(frames, features, quotas[clip], info['meta']['frame_count'])
        features_by_frame = dict(zip(frames, features))
        descriptors[clip]['selected'] = len(picked)
        for frame in picked:
            tasks.append(dict(id=f'{clip}_{frame:05d}', clip=clip, frame=frame, split='train',
                source_group=group_id(info['meta']['source_group']), kind='new', references={}, proposals={s: None for s in SIDES},
                candidates=[], reason='New source frame: visual and temporal diversity within a rally quota.',
                selection_feature_sha256=digest(features_by_frame[frame]),
                image_sha256=pools[clip]['records'][frame]['image_sha256'],
                cached_thumbnail=pools[clip]['records'][frame]['thumbnail']))
        print(f'Curated {clip}: {len(picked)} new court-view frames', flush=True)
    # All novel frames are annotated manually. Only body boxes are offered as unverified proposals.
    from ultralytics import YOLO
    model = YOLO(str(weights.resolve()))
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    for clip in sorted({t['clip'] for t in tasks}):
        selected = sorted([t for t in tasks if t['clip'] == clip], key=lambda t: t['frame'])
        for t in selected:
                meta = clips[clip]['meta']
                if t['kind'] == 'new':
                    image = cv2.imread(t['cached_thumbnail'])
                else:
                    stream = FrameSource(meta)
                    try:
                        original = stream.read(t['frame'])
                        t['image_sha256'] = hashlib.sha256(original.tobytes()).hexdigest()
                        image = cv2.resize(original, (640, round(meta['height']*640/meta['width'])))
                    finally:
                        stream.close()
                if image is None:
                    raise ValueError('Unreadable cached candidate')
                detected = model(image, classes=[0], conf=.2, imgsz=640, verbose=False, device='cpu')[0].boxes
                proposals = []
                for b in detected:
                    box = [float(v)*meta['width' if k%2 == 0 else 'height']/image.shape[1 if k%2 == 0 else 0]
                           for k, v in enumerate(b.xyxy[0].tolist())]
                    if box_valid(box, meta['width'], meta['height']):
                        proposals.append(dict(bbox=box, confidence=float(b.conf[0])))
                # Preserve all old box candidates as well as new YOLO candidates for corrections.
                boxes = [d['bbox'] for d in proposals]
                for b in t['candidates']:
                    if all(iou(b, a) < .98 for a in boxes):
                        boxes.append(b)
                t['candidates'] = boxes
                if t['kind'] == 'new':
                    t['proposals'] = proposed_bodies(proposals, meta['width'], meta['height'])
                thumb = cv2.resize(image, (640, round(image.shape[0]*640/image.shape[1])))
                thumb_path = output/'thumbnails'/f'{t["id"]}.jpg'
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(thumb_path), thumb)
        print(f'Cached unverified player-box proposals: {clip} ({len(selected)} frames)', flush=True)
    corrections = [t for t in tasks if t['kind'] == 'correction']
    fresh = {c: sorted([t for t in tasks if t['kind'] == 'new' and t['clip'] == c],
                       key=lambda t: t['frame']) for c in train}
    # Interleave rallies, and visit temporal quantiles instead of consecutive frames.
    ordered = list(corrections)
    for turn in range(max(quotas.values())):
        for c in sorted(train):
            rows = fresh[c]
            if turn < len(rows):
                # Bit-reversal ordering spreads the early part of the pass over the whole clip.
                order = sorted(range(len(rows)), key=lambda k: int(f'{k:08b}'[::-1], 2))
                ordered.append(rows[order[turn]])
    used_clips = {t['clip']: clips[t['clip']] for t in ordered}
    ids = [t['id'] for t in ordered]
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError('Queue must contain exactly the requested distinct source frames')
    # Also exclude exact repeated images, not just repeated frame IDs.
    if len({t['image_sha256'] for t in ordered}) != total:
        raise ValueError('Exact duplicate source images found; reconcile before labelling')
    manifest = dict(format=FORMAT, tasks=ordered, clips=used_clips, protected_sources=protected,
        previous_root=str(previous), previous_manifest_sha256=file_hash(previous/'manifest.json'),
        coordinate_system='original_pixels', frame_indexing='clip_local_zero_based',
        interpolation=False, source_labels_read_only=True, proposed_boxes_verified=False,
        selection=dict(total=total, new_frames=wanted, corrections=len(corrections),
            training_rallies=len(train), descriptors=descriptors, quotas=quotas,
            strategy='Novel source frames only; per-rally floor; farthest image/time selection; rally interleaving.',
            excluded_registered_validation_rallies='Only explicit box corrections, never training',
            final_test_excluded=True, unregistered_rallies='17-24 need source-match confirmation'),
        acceptance=dict(metric='PCK20 including misses', near_at_least=.85, far_at_least=.85,
                        separately=True, scope='held-out located queries; not a guarantee of full-video recall'),
        unknown_policy='H = unlocatable, never a negative presence label; located positives are retained independently.',
        model_weights_sha256=file_hash(weights))
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError('An original annotation changed during curation')
    save_json(output/'manifest.json', manifest)
    store = SampleStore(output)
    report = store.status()
    report.update(selection=manifest['selection'], original_sources_unchanged=True,
                  training_frames=sum(t['split'] == 'train' for t in ordered),
                  validation_frames=sum(t['split'] == 'val' for t in ordered))
    save_json(output/'preparation_report.json', report)
    with (output/'queue.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'clip', 'frame', 'split', 'source_group', 'kind', 'reason'])
        writer.writeheader()
        writer.writerows({k: t[k] for k in writer.fieldnames} for t in ordered)
    return report


def export_labels(root, output):
    """Export positives AND unknown masks, retaining every original approved pair."""
    store = SampleStore(root)
    output = Path(output).resolve()
    separate(store.root, output)
    separate(store.manifest['previous_root'], output)
    if output.exists():
        raise FileExistsError('Use a fresh export directory')
    old = OwnershipStore(store.manifest['previous_root'])
    from .ownership import export as legacy_export
    output.mkdir(parents=True)
    legacy_export(old.root, output/'historical_frozen')
    records = {}
    for split in ('train', 'val'):
        for line in (output/'historical_frozen'/f'{split}.jsonl').read_text().splitlines():
            r = json.loads(line)
            r['annotation_version'] = 'historical_v005'
            records[r['id']] = r
    partial = []
    unresolved_validation = []
    for task in store.tasks.values():
        row = store.row(task['id'])
        if row['decision'] == 'pending':
            continue
        if row['decision'] == 'excluded':
            if task['split'] == 'val':
                unresolved_validation.append(task['id'])
            else:
                records.pop(task['id'], None)
            continue
        players = {}
        for side in SIDES:
            p = row['players'][side]
            usable = p['racket_state'] == 'located' and p['body_state'] == 'verified'
            players[side] = dict(bbox=p['bbox'], ownership_verified=p['body_state'] == 'verified', track_id=None,
                head_center=dict(name='head_center', x=p['point'][0], y=p['point'][1], v=2) if usable else None,
                racket_state=p['racket_state'], body_state=p['body_state'], target_valid=usable,
                point_origin=('manual correction; original reference preserved' if p.get('reference_override') else
                              'locked previous annotation' if side in task['references'] else 'new manual'),
                reference_override=copy.deepcopy(p.get('reference_override')))
        m = store.clips[task['clip']]['meta']
        r = dict(format='racket_real_ownership_v004', id=task['id'], clip=task['clip'], frame=task['frame'],
            split=task['split'], source_group=task['source_group'], source=m['source'], source_hash=m['source_hash'],
            start_frame=m.get('start_frame', 0), original_size=[m['width'], m['height']],
            coordinate_system='original_pixels', players=players, annotation_version='sampled_v006',
            locked_owner=next(iter(task['references']), None),
            locked_reference_sha256=digest(task['references']))
        partial.append(r)
        if all(p['target_valid'] for p in players.values()):
            # A complete verified correction replaces, never duplicates, the frame.
            records[r['id']] = r
        elif task['split'] == 'val':
            # Do not improve validation scores by silently dropping a hard case.
            # Its historical paired input remains until an explicit benchmark revision.
            unresolved_validation.append(task['id'])
        else:
            records.pop(task['id'], None)
    for split in ('train', 'val'):
        rows = sorted([r for r in records.values() if r['split'] == split], key=lambda r: r['id'])
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        masked = [r for r in partial if r['split'] == split]
        (output/f'{split}_with_masks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in masked), encoding='utf-8')
    store.verify()
    report = dict(paired_counts=dict(Counter(r['split'] for r in records.values())),
        sampled_records_with_masks=len(partial), pending=store.status()['pending'],
        historical_benchmark_preserved_in='historical_frozen/val.jsonl',
        corrected_validation_is_new_version=True, no_validation_in_training=True,
        partial_queries_require_mask_aware_trainer=True,
        unresolved_validation_corrections=unresolved_validation,
        unresolved_validation_retains_historical_pairs=True,
        note='Do not run the old pair-only trainer directly on *_with_masks.jsonl. Next package must consume valid-query masks.',
        acceptance=store.manifest['acceptance'])
    save_json(output/'export.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--total', type=int, default=1000)
    for command in (prep, sub.add_parser('label'), sub.add_parser('fix'), sub.add_parser('boxes'), sub.add_parser('status'), sub.add_parser('export')):
        command.add_argument('--root', type=Path, default=ROOT)
        if command.prog.endswith(' label'):
            command.add_argument('--id', help='Open any frame in the full editable review queue')
        if command.prog.endswith(' export'):
            command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        report = prepare(args.root, args.total)
    elif args.command == 'status':
        report = SampleStore(args.root).status()
    elif args.command == 'export':
        report = export_labels(args.root, args.output)
    elif args.command == 'fix':
        from .ownership_fix_labeler import CorrectionLabeler
        CorrectionLabeler(args.root).run()
        return
    elif args.command == 'boxes':
        from .ownership_box_review import SampledBoxLabeler
        SampledBoxLabeler(args.root).run()
        return
    else:
        from .ownership_full_review import FullReviewLabeler
        FullReviewLabeler(args.root, args.id).run()
        return
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
