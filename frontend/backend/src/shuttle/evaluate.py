"""Evaluate clip-local, zero-based original-pixel predictions against labels.

Legacy JSONL requires --confirm-original-coordinates: the caller explicitly
attests its provenance, since bare coordinates cannot prove their own scale.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import cv2

from .dataset import read_labels


def load_predictions(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        frame = row['frame']
        if type(frame) is not int or frame < 0 or frame in rows:
            raise ValueError('Prediction frames must be unique nonnegative integers')
        if row['status'] not in ('detected', 'missing', 'uncertain'):
            raise ValueError('Unknown prediction status')
        if row['status'] == 'detected' and not all(
                isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in ('x', 'y')):
            raise ValueError('Detected prediction requires finite coordinates')
        rows[frame] = row
    return rows


def validate_alignment(labels, predictions, meta, video_path, prediction_meta=None,
                       confirm_original_coordinates=False):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open alignment video')
        width, height, count = [int(cap.get(p)) for p in
                               (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                cv2.CAP_PROP_FRAME_COUNT)]
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if (width, height, count) != (meta['original_width'], meta['original_height'], meta['frame_count']):
        raise ValueError('Video/label metadata dimensions or frame count disagree')
    if meta.get('start_frame', 0) != 0 or not math.isclose(fps, meta['fps'], rel_tol=1e-4):
        raise ValueError('Video/labels have an offset or FPS mismatch; supply the exact clip video')
    # Complete prediction coverage prevents silently treating a shifted/truncated file as misses.
    if set(predictions) != set(range(count)):
        raise ValueError('Predictions must cover exactly the zero-based video frame range')
    if any(type(f) is not int or not 0 <= f < count for f in labels):
        raise ValueError('Label frame index outside clip')
    if prediction_meta:
        expected = dict(width=width, height=height, frame_count=count,
                        frame_origin=0, coordinate_system='original_pixels')
        if any(prediction_meta.get(k) != v for k, v in expected.items()):
            raise ValueError('Prediction coordinate/frame metadata disagree')
        if not math.isclose(prediction_meta['fps'], fps, rel_tol=1e-4):
            raise ValueError('Prediction FPS mismatch')
        if Path(prediction_meta['video']).name != Path(video_path).name:
            raise ValueError('Prediction video identity mismatch')
    elif not confirm_original_coordinates:
        raise ValueError('Legacy predictions lack coordinate metadata; verify provenance and use '
                         '--confirm-original-coordinates')
    for frame, row in predictions.items():
        if 'time_seconds' in row and not math.isclose(row['time_seconds'], frame/fps, abs_tol=1e-5):
            raise ValueError('Prediction time/frame alignment mismatch')
        if row['status'] == 'detected' and not (0 <= row['x'] < width and 0 <= row['y'] < height):
            raise ValueError('Prediction outside original image')
    for row in labels.values():
        if row['visible'] == 1 and not (0 <= row['x'] < width and 0 <= row['y'] < height):
            raise ValueError('Label outside original image')
    return dict(width=width, height=height, frames=count, fps=fps,
                coordinate_system='original_pixels', frame_origin=0,
                provenance='metadata' if prediction_meta else 'caller-confirmed legacy provenance')


def evaluate(labels, predictions, tolerance=20, jump_threshold=200):
    if tolerance <= 0 or jump_threshold <= 0:
        raise ValueError('Tolerance and jump threshold must be positive')
    tp = fp = fn = tn = invisible_fp = invisible = accepted = jumps = 0
    errors = []
    previous = None
    labelled = 0
    for frame, truth in sorted(labels.items()):
        if truth['visible'] is None:
            previous = None
            continue
        labelled += 1
        pred = predictions.get(frame)
        detected = pred is not None and pred['status'] == 'detected'
        if detected:
            accepted += 1
            if previous and frame == previous[0]+1:
                jumps += math.hypot(pred['x']-previous[1], pred['y']-previous[2]) > jump_threshold
            previous = (frame, pred['x'], pred['y'])
        else:
            previous = None
        if truth['visible'] == 0:
            invisible += 1
            if detected:
                fp += 1
                invisible_fp += 1
            else:
                tn += 1
        elif detected:
            error = math.hypot(pred['x']-truth['x'], pred['y']-truth['y'])
            if error <= tolerance:
                tp += 1
                errors.append(error)
            else:
                fp += 1
                fn += 1
        else:
            fn += 1
    ratio = lambda a, b: a/b if b else None
    precision, recall = ratio(tp, tp+fp), ratio(tp, tp+fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=ratio(2*tp, 2*tp+fp+fn), mean_error=statistics.mean(errors) if errors else None,
                median_error=statistics.median(errors) if errors else None,
                invisible_false_positives=invisible_fp, invisible_frames=invisible,
                invisible_fp_rate=ratio(invisible_fp, invisible), large_jumps=jumps,
                accepted_percent=100*accepted/labelled if labelled else None,
                labelled_frames=labelled, skipped_unknown=len(labels)-labelled,
                tolerance_pixels=tolerance, jump_threshold_pixels=jump_threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path, nargs='+')
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--meta', type=Path, help='Defaults to labels sibling meta.json')
    parser.add_argument('--tolerance', type=float, default=20)
    parser.add_argument('--jump-threshold', type=float, default=200)
    parser.add_argument('--confirm-original-coordinates', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Detect duplicate CSV indices before read_labels can collapse them.
    with args.labels.open(newline='') as file:
        indices = [int(row['frame']) for row in csv.DictReader(file)]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate label frame indices')
    labels = read_labels(args.labels)
    meta = json.loads((args.meta or args.labels.with_name('meta.json')).read_text())
    results = {}
    for path in args.predictions:
        predictions = load_predictions(path)
        sidecar = path.with_name('shuttle_metadata.json')
        alignment = validate_alignment(labels, predictions, meta, args.video,
                                       json.loads(sidecar.read_text()) if sidecar.exists() else None,
                                       args.confirm_original_coordinates)
        results[str(path)] = dict(alignment=alignment, metrics=evaluate(
            labels, predictions, args.tolerance, args.jump_threshold))
    output = json.dumps(results, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x') as file:
            file.write(output+'\n')
    print(output)


if __name__ == '__main__':
    main()
