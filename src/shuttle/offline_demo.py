"""Two-pass video inference/refinement/rendering without retaining decoded frames."""
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .evaluate import load_predictions, validate_alignment
from .offline_tracker import OfflineTracker
from .tracker import TemporalTracker


def _annotate(image, detection, trace, colour):
    canvas = image.copy()
    point = (round(detection['x']), round(detection['y'])) if detection else None
    trace.append(point)
    for first, second in zip(trace, list(trace)[1:]):
        if first is not None and second is not None:
            cv2.line(canvas, first, second, colour, 2)
    if point is not None:
        cv2.circle(canvas, point, 12, colour, 3)
    return canvas


def render_refined_video(video_path, output_video, frames, refined, fps, width, height,
                         comparison_path=None):
    """Render refined coordinates; optional comparison uses the SAME candidates."""
    output_video = Path(output_video)
    if output_video.exists() or (comparison_path and Path(comparison_path).exists()):
        raise FileExistsError('Video output already exists')
    video = cv2.VideoCapture(str(video_path))
    writers = []
    try:
        if not video.isOpened():
            raise ValueError(f'Cannot reopen {video_path}')
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        writers.append(writer)
        comparison = None
        if comparison_path:
            panel_width = min(640, width)//2*2
            panel_height = max(2, round(height*panel_width/width)//2*2)
            comparison = cv2.VideoWriter(str(comparison_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (panel_width*3, panel_height+64))
            writers.append(comparison)
        if any(not w.isOpened() for w in writers):
            raise OSError('Cannot create output video')
        traces = [deque(maxlen=max(2, round(fps))) for _ in range(3)]
        temporal = TemporalTracker(width, height, fps) if comparison else None
        for frame, (candidates, detection) in enumerate(zip(frames, refined)):
            ok, image = video.read()
            if not ok:
                raise ValueError(f'Render video ended before candidate frame {frame}')
            canvas = _annotate(image, detection, traces[2], (0, 255, 0))
            # Keep comparison labels outside the image so small panels stay readable.
            annotated = canvas.copy()
            for text, y in (('OFFLINE V2', height-52), ('DETECTED' if detection else 'MISSING', height-20)):
                cv2.putText(annotated, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 5)
                cv2.putText(annotated, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
            writer.write(annotated)
            if comparison:
                raw = max(candidates, key=lambda d: d['score'], default=None)
                causal = temporal.update(candidates, frame)
                panels = [_annotate(image, raw, traces[0], (0, 100, 255)),
                          _annotate(image, causal, traces[1], (0, 230, 255)), canvas]
                labelled = []
                for panel, title, selected, colour in zip(
                        panels, ('ORIGINAL V2', 'TEMPORAL V2', 'OFFLINE V2'),
                        (raw, causal, detection), ((0, 100, 255), (0, 230, 255), (0, 255, 0))):
                    panel = cv2.copyMakeBorder(cv2.resize(panel, (panel_width, panel_height)),
                                               64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    status = f"DETECTED score={selected['score']:.3f}" if selected else 'MISSING'
                    cv2.putText(panel, f'{title} | frame {frame}', (12, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, colour, 2)
                    cv2.putText(panel, status, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2)
                    labelled.append(panel)
                comparison.write(np.concatenate(labelled, axis=1))
            if frame % 100 == 0:
                print(f'Rendered {frame+1} refined frames', flush=True)
        if video.read()[0]:
            raise ValueError('Render video contains more frames than candidate data')
    finally:
        video.release()
        for writer in writers:
            writer.release()


def run_offline_demo(video_path, weights, output_dir, threshold=.5, device='auto',
                     config=None, candidate_cache=None, comparison_video=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ['shuttle_demo.mp4', 'shuttle_tracking.jsonl', 'shuttle_metadata.json',
             'shuttle_offline_diagnostics.json', 'shuttle_comparison.mp4']
    if any((output_dir/name).exists() for name in names):
        raise FileExistsError('Choose a fresh output directory to avoid replacing an earlier result')
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise ValueError(f'Cannot open {video_path}')
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(fps, width, height, count) <= 0:
            raise ValueError('Empty video or invalid video metadata')
        if candidate_cache:
            source = Path(candidate_cache)
            rows = load_predictions(source)
            sidecar_path = source.with_name('shuttle_metadata.json')
            if not sidecar_path.exists():
                raise ValueError('Candidate cache requires coordinate/video metadata sidecar')
            sidecar = json.loads(sidecar_path.read_text())
            meta = dict(original_width=width, original_height=height, frame_count=count, fps=fps)
            validate_alignment({}, rows, meta, video_path, sidecar)
            # The legacy temporal cache was generated at the old CLI default .5.
            cached_threshold = sidecar.get('candidate_threshold', .5)
            if threshold != cached_threshold:
                raise ValueError('Candidate cache threshold differs from requested threshold')
            if any('candidates' not in row for row in rows.values()):
                raise ValueError('Cache must contain ALL candidates for EVERY frame')
            frames = [rows[i]['candidates'] for i in range(count)]
            inference_source = dict(candidate_cache=str(source.resolve()),
                                    cache_metadata=sidecar,
                                    legacy_threshold_assumption='candidate_threshold' not in sidecar)
        else:
            # Local import avoids coupling the standalone tracker to Torch/demo imports.
            from .demo import ShuttleDetector
            detector = ShuttleDetector(weights, threshold, device)
            ok, current = video.read()
            if not ok:
                raise ValueError('Cannot decode first frame')
            previous = current
            has_next, following = video.read()
            frames = []
            while True:
                frames.append(detector.predict_candidates(previous, current, following if has_next else current))
                if len(frames) % 100 == 1:
                    print(f'Collected candidates for {len(frames)} frames', flush=True)
                if not has_next:
                    break
                previous, current = current, following
                has_next, following = video.read()
            if len(frames) != count:
                raise ValueError('Decoded frame count disagrees with video metadata')
            inference_source = dict(weights=str(Path(weights).resolve()), device=str(detector.device))
    finally:
        video.release()
    tracker = OfflineTracker(width, height, fps, config)
    refined = tracker.refine(frames)
    metadata = dict(video=str(Path(video_path).resolve()), width=width, height=height,
                    fps=fps, frame_count=len(frames), frame_origin=0,
                    coordinate_system='original_pixels', temporal_filter=False, offline_refine=True,
                    candidate_threshold=threshold, offline_config=asdict(tracker.config),
                    inference_source=inference_source)
    with (output_dir/'shuttle_metadata.json').open('x') as file:
        json.dump(metadata, file, indent=2)
    with (output_dir/'shuttle_offline_diagnostics.json').open('x') as file:
        json.dump(tracker.diagnostics(), file, indent=2)
    with (output_dir/'shuttle_tracking.jsonl').open('x') as file:
        for frame, (candidates, detection) in enumerate(zip(frames, refined)):
            raw = max(candidates, key=lambda d: d['score'], default=None)
            row = dict(frame=frame, time_seconds=frame/fps,
                       status='detected' if detection else 'missing', x=None, y=None, score=None,
                       raw_detection={k: raw[k] for k in ('x', 'y', 'score')} if raw else None,
                       candidates=candidates, tracking_reason='offline_selected' if detection else 'offline_missing')
            if detection:
                row.update(detection)
            file.write(json.dumps(row, allow_nan=False)+'\n')
    render_refined_video(video_path, output_dir/'shuttle_demo.mp4', frames, refined,
                         fps, width, height,
                         output_dir/'shuttle_comparison.mp4' if comparison_video else None)
    print(f'Saved offline outputs in {output_dir}', flush=True)
