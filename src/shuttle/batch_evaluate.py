"""Frozen-parameter multi-rally evaluation using one shared inference per frame.

Writes evaluator-compatible JSONL for original/temporal/offline modes, alignment
checks, provenance hashes and pooled counts. Does not render videos or train.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np

from .dataset import frame_path, letterbox, read_labels
from .demo import ShuttleDetector
from .evaluate import evaluate, validate_alignment
from .offline_tracker import OfflineConfig, OfflineTracker
from .tracker import TemporalTracker, TrackerConfig

MODES = ('original', 'temporal', 'offline')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def pooled_counts(metrics):
    """Micro metrics from integer totals, never from averaged percentages."""
    counts = {key: sum(row[key] for row in metrics) for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
    ratio = lambda a, b: a/b if b else None
    return dict(counts, precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn))


def preflight(rally, root):
    video_path = root/'videos'/f'{rally}.mp4'
    clip = root/'clips'/rally
    for path in (video_path, clip/'labels.csv', clip/'meta.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads((clip/'meta.json').read_text())
    with (clip/'labels.csv').open(newline='') as file:
        indices = [int(row['frame']) for row in csv.DictReader(file)]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate label frame indices')
    labels = read_labels(clip/'labels.csv')
    # Existing alignment utility verifies dimensions/FPS/offsets/ranges and label bounds.
    placeholders = {frame: dict(status='missing') for frame in range(meta['frame_count'])}
    alignment = validate_alignment(labels, placeholders, meta, video_path,
                                   confirm_original_coordinates=True)
    if set(labels) != set(placeholders):
        raise ValueError('Labels must explicitly cover every clip frame, including unknowns')
    visibility = dict(visible=sum(r['visible'] == 1 for r in labels.values()),
                      invisible=sum(r['visible'] == 0 for r in labels.values()),
                      unknown=sum(r['visible'] is None for r in labels.values()))
    return dict(video=video_path, clip=clip, meta=meta, labels=labels,
                alignment=alignment, visibility=visibility)


def collect(detector, info, rally, output_dir):
    meta = info['meta']
    count = meta['frame_count']
    sample_frames = sorted({0, count//4, count//2, 3*count//4, count-1})
    video = cv2.VideoCapture(str(info['video']))
    frames, samples = [], []
    start = time.perf_counter()
    try:
        ok, current = video.read()
        if not ok:
            raise ValueError('Cannot decode first frame')
        previous = current
        has_next, following = video.read()
        with (output_dir/'candidates.jsonl').open('x') as file:
            while True:
                frame = len(frames)
                if current.shape[:2] != (meta['original_height'], meta['original_width']):
                    raise ValueError(f'Frame {frame} decoded at wrong dimensions')
                if frame in sample_frames:
                    prepared_path = frame_path(info['clip'], frame)
                    prepared = cv2.imread(str(prepared_path))
                    if prepared is None:
                        raise FileNotFoundError(prepared_path)
                    expected = letterbox(current, meta['transform'])
                    if prepared.shape != expected.shape:
                        raise ValueError(f'Prepared image dimensions disagree at frame {frame}')
                    mae = float(np.abs(prepared.astype(np.float32)-expected.astype(np.float32)).mean())
                    samples.append(dict(frame=frame, prepared_video_mean_absolute_pixel_difference=mae))
                    if mae > 8:
                        raise ValueError(f'Prepared/video frame alignment suspect at {frame}: MAE={mae:.2f}')
                candidates = detector.predict_candidates(previous, current, following if has_next else current)
                frames.append(candidates)
                file.write(json.dumps(dict(frame=frame, time_seconds=frame/meta['fps'], candidates=candidates))+'\n')
                if frame % 50 == 0:
                    elapsed = time.perf_counter()-start
                    print(f'{rally}: inferred {frame+1}/{count} frames ({elapsed:.1f}s)', flush=True)
                if not has_next:
                    break
                previous, current = current, following
                has_next, following = video.read()
    finally:
        video.release()
    if len(frames) != count:
        raise ValueError(f'Decoded {len(frames)} frames, metadata declares {count}')
    return frames, samples, time.perf_counter()-start


def save_predictions(output, mode, frames, detections, info, provenance):
    directory = output/mode
    directory.mkdir()
    meta = info['meta']
    metadata = dict(video=str(info['video'].resolve()), width=meta['original_width'],
                    height=meta['original_height'], fps=meta['fps'], frame_count=len(frames),
                    frame_origin=0, coordinate_system='original_pixels',
                    candidate_threshold=.5, temporal_filter=mode == 'temporal', offline_refine=mode == 'offline',
                    provenance=provenance)
    if mode == 'temporal':
        metadata['tracker_config'] = asdict(TrackerConfig())
    elif mode == 'offline':
        metadata['offline_config'] = asdict(OfflineConfig())
    write_json(directory/'shuttle_metadata.json', metadata)
    rows = {}
    with (directory/'shuttle_tracking.jsonl').open('x') as file:
        for frame, (candidates, detection) in enumerate(zip(frames, detections)):
            row = dict(frame=frame, time_seconds=frame/meta['fps'],
                       status='detected' if detection else 'missing', x=None, y=None, score=None)
            if detection:
                row.update(detection)
            if mode != 'original':
                raw = candidates[0] if candidates else None
                row.update(raw_detection={k: raw[k] for k in ('x', 'y', 'score')} if raw else None,
                           candidates=candidates)
            rows[frame] = row
            file.write(json.dumps(row, allow_nan=False)+'\n')
    alignment = validate_alignment(info['labels'], rows, meta, info['video'], metadata)
    metrics = evaluate(info['labels'], rows, tolerance=20, jump_threshold=200)
    result = dict(alignment=alignment, metrics=metrics)
    write_json(directory/'evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rallies', nargs='+', default=['rally18', 'rally24', 'rally35', 'rally44'])
    parser.add_argument('--root', type=Path, default=Path('data/shuttle'))
    parser.add_argument('--weights', type=Path, default=Path('outputs/shuttle_v2/best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    if len(set(args.rallies)) != len(args.rallies):
        parser.error('Duplicate rally names')
    args.output.mkdir(parents=True, exist_ok=False)
    temporal_reference = json.loads(Path('outputs/shuttle_v2_filtered/rally29/shuttle_metadata.json').read_text())
    offline_reference = json.loads(Path('outputs/shuttle_v2_offline/rally29_review/shuttle_metadata.json').read_text())
    if asdict(TrackerConfig()) != temporal_reference['tracker_config']:
        raise ValueError('Temporal defaults differ from rally29')
    if asdict(OfflineConfig()) != offline_reference['offline_config']:
        raise ValueError('Offline defaults differ from rally29')
    protected = [args.weights]
    for name in ('demo.py', 'dataset.py', 'model.py', 'tracker.py', 'offline_tracker.py', 'offline_demo.py', 'evaluate.py'):
        protected.append(Path('src/shuttle')/name)
    for rally in args.rallies:
        protected += [args.root/'videos'/f'{rally}.mp4', args.root/'clips'/rally/'labels.csv',
                      args.root/'clips'/rally/'meta.json']
    before = {str(path): sha256(path) for path in protected if path.is_file()}
    write_json(args.output/'protected_inputs_before.json', before)
    summary = dict(command=' '.join(sys.argv), rallies={}, failures={},
                   temporal_config=asdict(TrackerConfig()), offline_config=asdict(OfflineConfig()),
                   threshold=.5, tolerance=20, jump_threshold=200, exploratory=True)
    infos = {}
    for rally in args.rallies:
        try:
            infos[rally] = preflight(rally, args.root)
            print(f'{rally}: preflight OK {infos[rally]["alignment"]}, {infos[rally]["visibility"]}', flush=True)
        except Exception:
            summary['failures'][rally] = traceback.format_exc()
    detector = ShuttleDetector(args.weights, .5, args.device) if infos else None
    for rally, info in infos.items():
        output = args.output/rally
        output.mkdir()
        try:
            frames, samples, elapsed = collect(detector, info, rally, output)
            temporal_tracker = TemporalTracker(info['meta']['original_width'], info['meta']['original_height'], info['meta']['fps'])
            temporal = [temporal_tracker.update(candidates, frame) for frame, candidates in enumerate(frames)]
            offline_tracker = OfflineTracker(info['meta']['original_width'], info['meta']['original_height'], info['meta']['fps'])
            offline = offline_tracker.refine(frames)
            original = [{k: candidates[0][k] for k in ('x', 'y', 'score')} if candidates else None for candidates in frames]
            provenance = dict(weights=str(args.weights.resolve()), weights_sha256=before[str(args.weights)],
                              device=str(detector.device), shared_candidate_cache=str((output/'candidates.jsonl').resolve()))
            results = {mode: save_predictions(output, mode, frames, decisions, info, provenance)
                       for mode, decisions in zip(MODES, (original, temporal, offline))}
            write_json(output/'shuttle_offline_diagnostics.json', offline_tracker.diagnostics())
            summary['rallies'][rally] = dict(visibility=info['visibility'], alignment=info['alignment'],
                                           prepared_frame_checks=samples, inference_seconds=elapsed,
                                           candidate_count=sum(map(len, frames)), modes=results)
            write_json(output/'summary.json', summary['rallies'][rally])
            print(f'{rally}: COMPLETE '+json.dumps({mode: r['metrics'] for mode, r in results.items()}), flush=True)
        except Exception:
            summary['failures'][rally] = traceback.format_exc()
            print(f'{rally}: FAILED {summary["failures"][rally]}', flush=True)
        write_json(args.output/f'progress_{rally}.json', summary)
    summary['micro_aggregate'] = {mode: pooled_counts([r['modes'][mode]['metrics'] for r in summary['rallies'].values()]) for mode in MODES}
    after = {str(path): sha256(path) for path in protected if path.is_file()}
    write_json(args.output/'protected_inputs_after.json', after)
    summary['protected_inputs_unchanged'] = before == after
    write_json(args.output/'summary.json', summary)
    print('MICRO '+json.dumps(summary['micro_aggregate']), flush=True)
    if summary['failures'] or not summary['protected_inputs_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
