"""Synthetic behavior and end-to-end checks, independent of rally29 labels."""
from itertools import product
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.shuttle.demo import run_demo
from src.shuttle.offline_tracker import OfflineConfig, OfflineTracker


def point(x, y=100, score=.99):
    return dict(x=x, y=y, score=score)


class OfflineTests(unittest.TestCase):
    def test_empty_and_deterministic(self):
        tracker = OfflineTracker(1000, 500)
        self.assertEqual(tracker.refine([]), [])
        self.assertEqual(tracker.refine([[], [], []]), [None]*3)
        frames = [[point(100), point(100)], [], [point(200)]]
        self.assertEqual(tracker.refine(frames), tracker.refine(frames))
        self.assertEqual(tracker.refine(frames)[0]['offline_candidate_index'], 0)

    def test_stationary_and_short_apex(self):
        tracker = OfflineTracker(1000, 500)
        self.assertTrue(all(tracker.refine([[point(100)]]*6)))
        path = tracker.refine([[point(100)]]*100)
        self.assertTrue(all(p is None for p in path))
        self.assertTrue(tracker.diagnostics()['hotspots'])

    def test_intermittent_hotspots_and_genuine_crossing(self):
        frames = [[] for _ in range(121)]
        for f in range(0, 121, 20):
            frames[f] = [point(500), point(850)]
        tracker = OfflineTracker(1000, 500)
        self.assertTrue(all(p is None for p in tracker.refine(frames)))
        self.assertEqual(len(tracker.diagnostics()['hotspots']), 2)
        for f, x in enumerate([450, 475, 500, 525, 550], 58):
            frames[f] = [point(x)]
        path = tracker.refine(frames)
        self.assertEqual(path[60]['x'], 500)
        self.assertTrue(path[60]['offline_moving_support'])
        self.assertGreater(path[60]['offline_hotspot_cost'], 0)
        self.assertLess(path[60]['offline_hotspot_cost'], tracker.config.hotspot_penalty)

    def test_smash_direction_change_and_occlusion_recovery(self):
        positions = [100, 200, 400, 600, 400, 200]
        frames = [[point(x)] for x in positions]+[[]]*8+[[point(850)], [point(800)], [point(750)]]
        path = OfflineTracker(1000, 500).refine(frames)
        self.assertEqual([p['x'] for p in path[:6]], positions)
        self.assertEqual(path[6:14], [None]*8)
        self.assertEqual(path[14]['x'], 850)

    def test_competing_candidates(self):
        frames = [[point(100+f*5, score=.9), point(900)] for f in range(100)]
        path = OfflineTracker(1000, 500).refine(frames)
        self.assertTrue(all(p and p['x'] == 100+f*5 for f, p in enumerate(path)))

    def test_future_evidence_and_input_immutability(self):
        frames = [[point(100, score=.61)]]
        snapshot = json.dumps(frames)
        tracker = OfflineTracker(1000, 500)
        self.assertEqual(tracker.refine(frames), [None])
        self.assertEqual(json.dumps(frames), snapshot)
        frames += [[point(110, score=.61)], [point(120, score=.61)]]
        self.assertTrue(all(tracker.refine(frames)))

    def test_dp_matches_exhaustive_objective(self):
        config = OfflineConfig(hotspot_penalty=0)
        tracker = OfflineTracker(1000, 500, config=config)
        frames = [[point(100, score=.6), point(900, score=.9)],
                  [point(120, score=.8), point(880, score=.65)], [], [point(850)]]
        tracker.refine(frames)
        def utility(path):
            value = 0.0
            previous = None
            for state, candidates in zip(path, frames):
                candidate = candidates[state-1] if state else None
                if bool(candidate) != bool(previous):
                    value -= config.switch_cost
                if candidate:
                    value += candidate['score']-config.confidence_reference
                    if previous:
                        d = math.hypot(candidate['x']-previous['x'], candidate['y']-previous['y'])
                        value -= config.motion_penalty*(1-math.exp(-.5*(d/(config.motion_radius*tracker.diagonal))**2))
                previous = candidate
            return value-config.switch_cost*bool(previous)
        optimum = max(utility(path) for path in product(*(range(len(cs)+1) for cs in frames)))
        self.assertAlmostEqual(tracker.path_score, optimum)

    def test_validation(self):
        for changes in [dict(motion_radius=0), dict(hotspot_penalty=float('nan')),
                        dict(hotspot_min_episodes=1), dict(hotspot_min_episodes=2.5)]:
            with self.assertRaises(ValueError):
                OfflineConfig(**changes)
        tracker = OfflineTracker(1000, 500)
        for candidate in [point(-1), point(100, score=2), point(float('nan'))]:
            with self.assertRaises(ValueError):
                tracker.refine([[candidate]])


class OfflineDemoTests(unittest.TestCase):
    def test_two_pass_render_cache_and_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/'video.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
            for _ in range(3):
                writer.write(np.full((240, 320, 3), 30, np.uint8))
            writer.release()
            # Weak isolated candidate is visible in raw data but absent in final rendering.
            candidates = [[point(125, 102, .51)], [], []]
            with patch('src.shuttle.demo.ShuttleDetector') as detector:
                detector.return_value.device = 'cpu'
                detector.return_value.predict_candidates.side_effect = candidates
                run_demo(video, 'unused.pt', root/'out', offline_refine=True, comparison_video=True)
                self.assertEqual(detector.return_value.predict_candidates.call_count, 3)
                detector.return_value.predict.assert_not_called()
            rows = [json.loads(line) for line in (root/'out/shuttle_tracking.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['raw_detection']['x'], 125)
            self.assertEqual(rows[0]['status'], 'missing')
            self.assertIsNone(rows[0]['x'])
            cap = cv2.VideoCapture(str(root/'out/shuttle_demo.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            ok, image = cap.read()
            cap.release()
            self.assertTrue(ok)
            # No green shuttle marker near the raw point below the status bar.
            region = image[88:119, 110:145].astype(float)
            self.assertLess(float((region[:, :, 1]-region[:, :, 2]).max()), 30)
            cap = cv2.VideoCapture(str(root/'out/shuttle_comparison.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 960)
            cap.release()
            with patch('src.shuttle.demo.ShuttleDetector') as detector:
                run_demo(video, 'unused.pt', root/'cached', offline_refine=True,
                         candidate_cache=root/'out/shuttle_tracking.jsonl')
                detector.assert_not_called()
            self.assertEqual((root/'cached/shuttle_tracking.jsonl').read_text(),
                             (root/'out/shuttle_tracking.jsonl').read_text())
            with self.assertRaises(FileExistsError):
                run_demo(video, 'unused.pt', root/'out', offline_refine=True)
            with self.assertRaises(ValueError):
                run_demo(video, 'unused.pt', root/'bad', offline_refine=True, temporal_filter=True)
            with self.assertRaises(ValueError):
                run_demo(video, 'unused.pt', root/'bad', comparison_video=True)
            with self.assertRaises(ValueError):
                run_demo(video, 'unused.pt', root/'bad', offline_refine=True, threshold=.7,
                         candidate_cache=root/'out/shuttle_tracking.jsonl')
