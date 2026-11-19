"""Frozen v005/v006 on a fresh clip, with identical automatic player/pose inputs."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.racket_demo_utils import predict_batch
from src.detection.detector import PlayerDetector
from src.detection.target_selector import SelectionConfig, select_targets
from src.pose.estimator import PlayerPose
from src.racket.real_owner_infer import RealOwnerRacketPredictor
from src.racket.schema import file_hash, read_json, save_json
from src.racket.source import FrameSource

COLORS = ((255, 255, 0), (255, 0, 255))
CONNECTIONS = [('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
               ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'),
               ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
               ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
               ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
               ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')]


def overlay(frame, players, version, width, height):
    view = cv2.resize(frame, (width, height))
    sx, sy = width/frame.shape[1], height/frame.shape[0]
    at = lambda xy: (round(xy[0]*sx), round(xy[1]*sy))
    for player in players:
        color = COLORS[player['side'] == 'far_side']
        box = player['bbox']
        cv2.rectangle(view, at(box[:2]), at(box[2:]), color, 1)
        by_name = {p['name']: p for p in player['keypoints']
                   if p.get('visibility', 0)*p.get('presence', 0) > .2}
        for a, b in CONNECTIONS:
            if a in by_name and b in by_name:
                cv2.line(view, at([by_name[a]['x'], by_name[a]['y']]),
                         at([by_name[b]['x'], by_name[b]['y']]), (255, 160, 50), 1, cv2.LINE_AA)
        racket = player['racket_'+version]
        if racket['located']:
            xy = [racket['x'], racket['y']]
            cv2.circle(view, at(xy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(view, at(xy), 8, (0, 0, 0), 1, cv2.LINE_AA)
            wrists = [by_name[n] for n in ('left_wrist', 'right_wrist') if n in by_name]
            if wrists:
                wrist = min(wrists, key=lambda q: math.dist([q['x'], q['y']], xy))
                cv2.line(view, at([wrist['x'], wrist['y']]), at(xy), color, 2, cv2.LINE_AA)
        origin = at(box[:2])
        cv2.putText(view, player['side'].replace('_side', '').upper(),
                    (origin[0], max(15, origin[1]-6)), 0, .48, color, 1, cv2.LINE_AA)
    return view


def encoder(ffmpeg, path, size, fps, log):
    return subprocess.Popen([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size[0]}x{size[1]}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
        stdin=subprocess.PIPE, stderr=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', default='rally16')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-weights', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--batch-frames', type=int, default=2)
    parser.add_argument('--ffmpeg', type=Path, default=shutil.which('ffmpeg'))
    args = parser.parse_args()
    if min(args.stride, args.batch_frames) < 1:
        parser.error('Stride and batch size must be positive')
    if args.output.exists():
        raise FileExistsError('Preserve previous videos; use a new output folder')
    if not args.ffmpeg:
        raise FileNotFoundError('ffmpeg is required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    meta = read_json(REPO/'data/racket_manual/clips'/args.clip/'meta.json')
    if file_hash(meta['source']) != meta['source_hash']:
        raise ValueError('Source video changed')
    protected = {str(p): file_hash(p) for p in
                 [args.old_weights, args.weights, REPO/'yolo11n.pt', REPO/'models/pose_landmarker_lite.task']}
    protected.update({str(p): file_hash(p) for folder in ['src/detection', 'src/pose', 'src/temporal', 'src/court']
                      for p in (REPO/folder).rglob('*.py')})
    old = RealOwnerRacketPredictor(args.old_weights)
    new = RealOwnerRacketPredictor(args.weights)
    source = FrameSource(meta)
    detector = PlayerDetector(str(REPO/'yolo11n.pt'))
    body = PlayerPose(REPO/'models/pose_landmarker_lite.task')
    # Visible court geometry for this demonstration only; no racket labels.
    # This polygon is not the teammates' court-coordinate mapping.
    if args.clip != 'rally16':
        raise ValueError('Configure a visible court polygon for a different camera')
    polygon = [(x*meta['width'], y*meta['height']) for x, y in
               [(.36, .63), (.78, .63), (.98, .995), (.01, .995)]]
    config = SelectionConfig(max_targets=2, min_confidence=.3, min_bbox_height=meta['height']*.05,
                             min_aspect_ratio=.25, max_aspect_ratio=6., court_polygon=polygon)
    width = 1280
    height = round(meta['height']*width/meta['width'])
    height += height % 2
    fps = meta['fps']/args.stride
    main_path = args.output/f'v006_{args.clip}_in_action.mp4'
    comparison_path = args.output/f'v005_vs_v006_{args.clip}.mp4'
    logs = [(args.output/'encoding_main.log').open('w'), (args.output/'encoding_comparison.log').open('w')]
    writers = [encoder(args.ffmpeg, main_path, (width, height+132), fps, logs[0]),
               encoder(args.ffmpeg, comparison_path, (1920, 680), fps/2, logs[1])]
    count = missing = pose_missing = 0
    checked = set()
    started = time.monotonic()
    pending = []
    predictions = []
    try:
        for index in range(meta['frame_count']):
            frame = source.read(index)
            if index % args.stride:
                continue
            people = select_targets(detector.detect_people(frame), config)
            for j, player in enumerate(people):
                player['side'] = 'near_side' if j == 0 else 'far_side'
                player['keypoints'] = body.extract(frame, player)
            pending.append(dict(frame=frame, index=index, players=people))
            if len(pending) < args.batch_frames and index+args.stride < meta['frame_count']:
                continue
            for version, predictor in [('v005', old), ('v006', new)]:
                predict_batch(predictor, pending)
                for item in pending:
                    for player in item['players']:
                        player['racket_'+version] = copy.deepcopy(player['racket'])
                    if version not in checked and item['players']:
                        direct = predictor.predict_players(item['frame'], item['players'])
                        for a, b in zip(direct, item['players']):
                            p, q = a['racket'], b['racket']
                            if p['located'] != q['located'] or (p['located'] and math.dist(
                                    [p['x'], p['y']], [q['x'], q['y']]) > .1):
                                raise ValueError('Batch predictions differ from production adapter')
                        checked.add(version)
            for item in pending:
                frame, players, index = item['frame'], item['players'], item['index']
                if len(players) < 2:
                    missing += 1
                pose_missing += sum(not p['keypoints'] for p in players)
                canvas = np.zeros((height+132, width, 3), np.uint8)
                canvas[94:94+height] = overlay(frame, players, 'v006', width, height)
                for j, text in enumerate([
                    f'v006 | {args.clip} frame {index}/{meta["frame_count"]-1} | fresh test-match inference',
                    'Cyan: NEAR racket | Magenta: FAR racket | Blue: frozen MediaPipe pose',
                    f'Automatic person boxes | {fps:g} fps | No manual labels, smoothing or interpolated points']):
                    cv2.putText(canvas, text, (12, 25+j*28), 0, .60, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, 'Unlabelled footage: qualitative demonstration; accuracy is unscored here.',
                            (12, height+120), 0, .56, (255, 255, 255), 1, cv2.LINE_AA)
                writers[0].stdin.write(canvas.tobytes())
                comparison = np.zeros((680, 1920, 3), np.uint8)
                for j, version in enumerate(('v005', 'v006')):
                    x = j*960
                    comparison[80:640, x:x+960] = overlay(frame, players, version, 960, 560)
                    size = old.size if version == 'v005' else new.size
                    cv2.putText(comparison, f'{version} | {size}px | {args.clip} frame {index}',
                                (x+14, 31), 0, .75, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(comparison, 'Same automatic boxes and frozen MediaPipe body features',
                                (x+14, 61), 0, .55, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(comparison, '0.5x playback | Cyan NEAR, magenta FAR | No smoothing or annotation inputs | Unlabelled test clip: no accuracy percentage',
                            (12, 666), 0, .58, (255, 255, 255), 1, cv2.LINE_AA)
                writers[1].stdin.write(comparison.tobytes())
                predictions.append(dict(clip=args.clip, frame=index, source_group=meta['source_group'],
                                        players=players, coordinate_system='original_pixels'))
                count += 1
                if count == 1 or count % 30 == 0:
                    print(f'RENDER {count}/{math.ceil(meta["frame_count"]/args.stride)}; '
                          f'{time.monotonic()-started:.0f}s; fewer than two people: {missing}', flush=True)
                    cv2.imwrite(str(args.output/'latest_frame.jpg'), comparison)
                if index in [0, 100, 200, 300, 400]:
                    cv2.imwrite(str(args.output/f'comparison_{index:05d}.jpg'), comparison)
            pending = []
    finally:
        source.close()
        body.close()
        errors = []
        for writer, log in zip(writers, logs):
            writer.stdin.close()
            code = writer.wait()
            log.close()
            if code:
                errors.append(code)
        if errors:
            raise RuntimeError('Video encoder failed; see encoding logs')
    (args.output/'predictions.jsonl').write_text(
        ''.join(json.dumps(p, allow_nan=False)+'\n' for p in predictions), encoding='utf-8')
    videos = []
    for path in [main_path, comparison_path]:
        cap = cv2.VideoCapture(str(path))
        written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if written != count:
            raise ValueError('Incomplete video: '+str(path))
        for index in [0, count//2, count-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, _ = cap.read()
            if not ok:
                raise ValueError('Encoded video cannot be decoded')
        videos.append(dict(file=path.name, frames=written, fps=cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
    if any(file_hash(p) != expected for p, expected in protected.items()):
        raise ValueError('A protected model or shared module changed')
    save_json(args.output/'verification.json', dict(videos=videos, clip=args.clip, split=meta['split'],
        source_group=meta['source_group'], source_hash=meta['source_hash'], final_test_video_inspected=True,
        frames_with_fewer_than_two_people=missing, missing_body_poses=pose_missing,
        production_adapters_checked=sorted(checked), protected_files_unchanged=True,
        old_weights_sha256=file_hash(args.old_weights), new_weights_sha256=file_hash(args.weights),
        input_sizes=[old.size, new.size], annotation_inputs=False, smoothing=False,
        quantitative_test_accuracy_measured=False, inference_stride=args.stride,
        player_assignment='Court-ROI and bottom-edge heuristic; no stable athlete identity implied'))
    print('VIDEOS READY', main_path, comparison_path, flush=True)


if __name__ == '__main__':
    main()
