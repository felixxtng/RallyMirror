"""Build v006 features from frozen old pairs and current per-query masks."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from .mediapipe_bundle import BODY_NAMES, square_image
from .ownership import iou
from .ownership_fast import SampleStore, export_labels, read_selected
from .real_owner_data import SIDES, scene_bounds, selected_pose
from .sampled_owner_data import FORMAT
from .schema import digest, file_hash, read_json, save_json


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def merged_records(export):
    """Explicit reviewed overrides replace old frames; never duplicate a frame."""
    export = Path(export)
    records = {}
    for split in ('train', 'val'):
        for row in read_records(export/(split+'.jsonl')):
            records[row['id']] = row
        for row in read_records(export/(split+'_with_masks.jsonl')):
            # Validation cannot be shrunk by an unknown target.
            if split == 'val' and not all(p['target_valid'] for p in row['players'].values()):
                raise ValueError('Validation correction unresolved: '+row['id'])
            records[row['id']] = row
    return sorted(records.values(), key=lambda r: (r['clip'], r['frame']))


def build(repo, root, output, pose_extractor=None):
    repo, root, output = map(Path, (repo, root, output))
    store = SampleStore(root)
    status = store.status()
    if status['pending']:
        raise ValueError('Unfinished annotation rows remain')
    progress = read_json(root/'full_review_progress.json')['reviewed']
    if any(progress.get(k) != digest(store.row(k)) for k in store.tasks):
        raise ValueError('Some edited frames have not been confirmed')
    if output.exists():
        raise FileExistsError('Preserve existing builds; use a new output directory')
    protected = {str(p): file_hash(p) for folder in
        ('data/racket_manual', 'data/racket_ownership_v004', 'data/racket_ownership_all_v005',
         'data/racket_ownership_sampled_v006', 'src/pose', 'src/detection', 'src/temporal', 'src/court', 'models')
        for p in (repo/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    old_bundle = repo/'outputs/racket_ownership_v005/bundle'
    old = read_json(old_bundle/'manifest.json')
    old_by_id = {r['id']: r for r in old['frames']}
    returned = repo/'Runpod/racket_ownership_v005/racket_ownership_run_v005'
    warm = returned/'racket_center_real_owner_v005.pt'
    protected[str(warm)] = file_hash(warm)
    output.mkdir(parents=True)
    save_json(output/'protected_snapshot.json', protected)
    exports = output/'annotations'
    export_report = export_labels(root, exports)
    raw = merged_records(exports)
    bundle = output/'bundle'
    bundle.mkdir()
    for folder in ('images', 'demo_frames', 'models', 'native_v005', 'legacy_images'):
        (bundle/folder).mkdir()
    shutil.copyfile(warm, bundle/'models/baseline_v005.pt')
    shutil.copyfile(repo/'models/pose_landmarker_lite.task', bundle/'models/pose_landmarker_lite.task')
    save_json(bundle/'annotation_export.json', export_report)
    own_pose = pose_extractor is None
    if own_pose:
        from src.pose.estimator import PlayerPose
        pose_extractor = PlayerPose(repo/'models/pose_landmarker_lite.task')
    size, padding = 960, .05
    frames, exclusions, query_audit = [], [], []
    pose_counts = Counter()
    verified_sources = {}
    try:
        for clip in sorted({r['clip'] for r in raw}):
            selected = [r for r in raw if r['clip'] == clip]
            meta = read_json(repo/'data/racket_manual/clips'/clip/'meta.json')
            if file_hash(meta['source']) != meta['source_hash']:
                raise ValueError('Source video changed: '+clip)
            verified_sources[clip] = meta['source_hash']
            lookup = {r['frame']: r for r in selected}
            print(f'FEATURES {clip}: {len(selected)} frames, sequential source decode', flush=True)
            for frame, image in read_selected(meta, lookup):
                r = lookup[frame]
                if r['source_hash'] != meta['source_hash']:
                    raise ValueError('Source identity mismatch: '+r['id'])
                if r['id'] in store.tasks and store.tasks[r['id']]['kind'] == 'new':
                    if hashlib.sha256(image.tobytes()).hexdigest() != store.tasks[r['id']]['image_sha256']:
                        raise ValueError('Prepared source frame mismatch: '+r['id'])
                boxes = {s: copy.deepcopy(p['bbox']) if p.get('ownership_verified') else None
                         for s, p in r['players'].items()}
                has_point = lambda p: p.get('head_center') is not None and p.get('target_valid', p.get('ownership_verified', False))
                usable = {s: bool(boxes[s] and has_point(r['players'][s])) for s in SIDES}
                if not any(usable.values()):
                    exclusions.append(dict(id=r['id'], reason='No located racket with verified body; unknowns are not negatives'))
                    continue
                crop = scene_bounds([b for b in boxes.values() if b], padding)
                entry = dict(id=r['id'], clip=clip, frame=frame, split=r['split'], source_group=r['source_group'],
                    original_size=r['original_size'], source_hash=r['source_hash'], crop=crop,
                    image='images/'+r['id']+'.jpg', players={}, locked_owner=r.get('locked_owner'),
                    annotation_version=r.get('annotation_version', 'historical_v005'),
                    body_box_iou=iou(boxes[SIDES[0]], boxes[SIDES[1]]) if all(boxes.values()) else 0.)
                cached = old_by_id.get(r['id'])
                for side in SIDES:
                    player = r['players'][side]
                    head = player.get('head_center')
                    point = [head['x'], head['y']] if head else None
                    target = [(point[k]-crop[k])/crop[2] for k in range(2)] if point else None
                    valid = usable[side] and all(math.isfinite(v) and 0 <= v < 1 for v in target)
                    body = []
                    pose = np.zeros((17, 3), np.float32)
                    pose_status = 'masked_unknown_or_unverified'
                    if valid:
                        old_player = cached['players'][side] if cached else None
                        if old_player and old_player['owner_box'] == boxes[side]:
                            body = copy.deepcopy(old_player['body_keypoints'])
                            pose_counts['reused_frozen_body_features'] += 1
                        else:
                            body = pose_extractor.extract(image, dict(bbox=boxes[side]))
                            pose_counts['recomputed_body_features'] += 1
                        pose, pose_status = selected_pose(body, boxes[side],
                            [b for s, b in boxes.items() if s != side and b], crop)
                    elif usable[side]:
                        query_audit.append(dict(id=r['id'], side=side, split=r['split'],
                                                reason='Located point outside body-only context; query masked'))
                    pose_counts[pose_status] += 1
                    entry['players'][side] = dict(owner_box=boxes[side], target_original=point,
                        target=target, pose=pose.tolist(), body_keypoints=body, pose_status=pose_status,
                        valid_target=bool(valid), racket_state=player.get('racket_state', 'located' if head else 'unlocatable'),
                        point_origin=player['point_origin'])
                entry['train_eligible'] = any(p['valid_target'] for p in entry['players'].values())
                if r['split'] == 'val' and not all(p['valid_target'] for p in entry['players'].values()):
                    raise ValueError('Validation query cannot be dropped: '+r['id'])
                if not entry['train_eligible']:
                    exclusions.append(dict(id=r['id'], reason='All located queries outside body-only context'))
                    continue
                entry['head_separation_original_pixels'] = (math.dist(*[entry['players'][s]['target_original'] for s in SIDES])
                    if all(p['valid_target'] for p in entry['players'].values()) else 0.)
                if not cv2.imwrite(str(bundle/entry['image']), square_image(image, crop, size), [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError('Cannot write scene crop')
                if r['split'] == 'val':
                    entry['full_image'] = 'demo_frames/'+r['id']+'.jpg'
                    height = round(image.shape[0]*1280/image.shape[1])
                    height += height % 2
                    cv2.imwrite(str(bundle/entry['full_image']), cv2.resize(image, (1280, height)), [cv2.IMWRITE_JPEG_QUALITY, 94])
                frames.append(entry)
                if len(frames) % 100 == 0:
                    print(f'FEATURES saved {len(frames)}/{len(raw)}; poses {dict(pose_counts)}', flush=True)
    finally:
        if own_pose:
            pose_extractor.close()
    native = []
    for original in old['frames']:
        if original['split'] != 'val':
            continue
        record = copy.deepcopy(original)
        source = old_bundle/record['image']
        if file_hash(source) != old['file_hashes'][record['image']]:
            raise ValueError('Frozen validation image changed')
        record['image'] = 'native_v005/'+record['id']+'.jpg'
        shutil.copyfile(source, bundle/record['image'])
        native.append(record)
    legacy = []
    for original in old['legacy_validation']:
        record = copy.deepcopy(original)
        if record['image']:
            source = old_bundle/record['image']
            if file_hash(source) != old['file_hashes'][record['image']]:
                raise ValueError('Frozen legacy image changed')
            record['image'] = 'legacy_images/'+record['id']+'.jpg'
            shutil.copyfile(source, bundle/record['image'])
        legacy.append(record)
    fixed = sorted(r['id'] for r in old['frames'] if r['split'] == 'val')
    if sorted(r['id'] for r in frames if r['split'] == 'val') != fixed:
        raise ValueError('Frozen validation frame membership changed')
    counts = dict(train_frames=sum(r['split'] == 'train' for r in frames),
                  val_frames=sum(r['split'] == 'val' for r in frames),
                  per_side_train_queries={s: sum(r['split'] == 'train' and r['players'][s]['valid_target'] for r in frames) for s in SIDES},
                  paired_train_frames=sum(r['split'] == 'train' and all(p['valid_target'] for p in r['players'].values()) for r in frames),
                  partial_train_frames=sum(r['split'] == 'train' and sum(p['valid_target'] for p in r['players'].values()) == 1 for r in frames),
                  train_rallies=len({r['clip'] for r in frames if r['split'] == 'train'}),
                  pose_status=dict(pose_counts), masked_context_queries=len(query_audit))
    save_json(bundle/'training_query_audit.json', query_audit)
    save_json(bundle/'unusable_frames.json', exclusions)
    manifest = dict(version=FORMAT, run_version='v006', baseline_label='v005', status='features_ready',
        frames=frames, image_size=768, stored_image_size=960, context_padding=padding, body_names=BODY_NAMES,
        fixed_validation_ids=fixed, native_v005_validation=native,
        native_v005_expected_metrics=read_json(returned/'paired_metrics.json'),
        legacy_validation=legacy, counts=counts, source_hashes=verified_sources,
        split_unit='whole_source_match', initial_checkpoint='models/baseline_v005.pt',
        initial_checkpoint_sha256=file_hash(warm), annotation_manifest_sha256=file_hash(root/'manifest.json'),
        annotation_snapshot_sha256=digest({p: h for p, h in protected.items() if 'data' in Path(p).parts}),
        annotation_status=status, annotation_export=export_report,
        query_masks='Located point plus verified body required. H/unverified/out-of-context queries never contribute loss.',
        excluded_historical_unknowns='Unlocatable old rows lack verified ownership boxes; not invented as training examples.',
        acceptance=dict(near_at_least=.80, far_at_least=.80, metric='PCK20 including misses'),
        mediapipe_finetuned=False, final_test_evaluated=False,
        validation_note='Same 444 frame IDs; 3 explicit corrections from completed review. Old 444 inputs/targets retained as native_v005.')
    manifest['file_hashes'] = {p.relative_to(bundle).as_posix(): file_hash(p) for p in bundle.rglob('*') if p.is_file()}
    save_json(bundle/'manifest.json', manifest)
    if any(file_hash(p) != expected for p, expected in protected.items()):
        raise ValueError('A protected original changed during feature build')
    save_json(output/'verification.json', dict(counts=counts, protected_files_unchanged=True,
        all_1000_reviews_confirmed=True, final_test_evaluated=False, learning_accuracy_not_measured=True))
    return bundle


def finalize(repo, bundle, destination, guide):
    repo, bundle, destination = map(Path, (repo, bundle, destination))
    if destination.exists():
        raise FileExistsError('Preserve existing package')
    for folder in ('src', 'src/racket', 'src/pose'):
        (bundle/folder).mkdir(parents=True, exist_ok=True)
        (bundle/folder/'__init__.py').write_text('', encoding='utf-8')
    modules = ('mediapipe_bundle', 'mediapipe_model', 'mediapipe_train', 'owner_model', 'owner_data',
               'owner_train', 'player_constraint', 'real_owner_data', 'real_owner_train', 'real_owner_infer',
               'sampled_owner_data', 'sampled_owner_train')
    for module in modules:
        shutil.copyfile(repo/f'src/racket/{module}.py', bundle/f'src/racket/{module}.py')
    shutil.copyfile(repo/'src/pose/estimator.py', bundle/'src/pose/estimator.py')
    (bundle/'run_racket_ownership.py').write_text(
        'from src.racket.sampled_owner_train import main\nif __name__ == "__main__": main()\n', encoding='utf-8')
    setup = (repo/'scripts/setup_racket_runpod.py').read_text().replace('run_racket_mediapipe.py', 'run_racket_ownership.py')
    (bundle/'setup_runpod.py').write_text(setup, encoding='utf-8')
    shutil.copyfile(guide, bundle/'START_HERE.md')
    m = read_json(bundle/'manifest.json')
    m['status'] = 'ready'
    m['file_hashes'] = {p.relative_to(bundle).as_posix(): file_hash(p) for p in bundle.rglob('*')
                        if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    save_json(bundle/'manifest.json', m)
    from .sampled_owner_train import verify
    verify(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise RuntimeError('ZIP CRC check failed')
    report = dict(package=str(destination.resolve()), bytes=destination.stat().st_size,
                  sha256=file_hash(destination), manifest_sha256=file_hash(bundle/'manifest.json'),
                  counts=m['counts'], acceptance=m['acceptance'], cloud_training_started=False)
    save_json(destination.with_suffix('.json'), report)
    print(json.dumps(report, indent=2), flush=True)
    return report
