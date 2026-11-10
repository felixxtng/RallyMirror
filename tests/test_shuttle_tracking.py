"""Focused temporal, coordinate, evaluation and demo compatibility tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.shuttle.dataset import letterbox_transform, to_heatmap_point
from src.shuttle.demo import ShuttleDetector, extract_candidates, run_demo
from src.shuttle.evaluate import evaluate, load_predictions, validate_alignment
from src.shuttle.tracker import TemporalTracker, TrackerConfig


def candidate(x, y=100, score=0.95):
    return dict(x=x, y=y, score=score)


class TrackingTests(unittest.TestCase):
    def test_full_frame_candidates_and_compatible_wrapper(self):
        t = letterbox_transform(200, 100, 100, 100)
        heatmap = np.zeros((100, 100), np.float32)
        heatmap[0:3, 5:8] = 1  # Padding excluded.
        heatmap[35:37, 20:22] = .8
        heatmap[60:62, 80:82] = .9
        heatmap[50, 50] = 1  # Tiny noise excluded.
        found = extract_candidates(heatmap, t, 200, 100)
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0]['area'], 4)
        self.assertAlmostEqual(found[0]['x'], 161)
        self.assertAlmostEqual(found[0]['y'], 71)
        hx, hy = to_heatmap_point(found[0]['x'], found[0]['y'], t)
        self.assertAlmostEqual(hx, 80.5)
        self.assertAlmostEqual(hy, 60.5)
        detector = ShuttleDetector.__new__(ShuttleDetector)
        detector.predict_candidates = lambda *args: found
        self.assertEqual(detector.predict(None, None, None),
                         {k: found[0][k] for k in ('x', 'y', 'score')})
        detector.predict_candidates = lambda *args: []
        self.assertIsNone(detector.predict(None, None, None))

    def test_movement_reversal_and_smash(self):
        tracker = TemporalTracker(2200, 1100)
        for frame, x in enumerate([100, 130, 160, 190, 390, 220, 240]):
            self.assertIsNotNone(tracker.update([candidate(x, score=.99)], frame))

    def test_hotspot_apex_and_expiration(self):
        tracker = TemporalTracker(1000, 500)
        for frame in range(6):  # Short apex survives.
            self.assertIsNotNone(tracker.update([candidate(100)], frame))
        for frame in range(6, 25):
            result = tracker.update([candidate(100)], frame)
        self.assertIsNone(result)
        tracker.update([], 100)  # Old evidence expires.
        self.assertIsNotNone(tracker.update([candidate(100)], 101))

    def test_background_candidate_does_not_win(self):
        tracker = TemporalTracker(2200, 1100)
        for frame in range(25):
            accepted = tracker.update([candidate(100+frame*25, score=.95),
                                       candidate(1800, score=.99)], frame)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted['x'], 700)

    def test_loss_global_reacquisition_and_empty(self):
        tracker = TemporalTracker(1000, 500)
        self.assertIsNone(tracker.update([], 0))
        tracker.update([candidate(100)], 1)
        for frame in range(2, 5):
            self.assertIsNone(tracker.update([], frame))
        self.assertEqual(tracker.update([candidate(950)], 5)['x'], 950)
        self.assertEqual(tracker.reason, 'reacquired')
        with self.assertRaises(ValueError):
            tracker.update([], 5)

    def test_demo_disabled_and_missing_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/'test.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 32))
            for _ in range(4):
                writer.write(np.zeros((32, 64, 3), np.uint8))
            writer.release()
            with patch('src.shuttle.demo.ShuttleDetector') as detector:
                detector.return_value.predict.side_effect = [candidate(10), None, candidate(20), None]
                run_demo(video, 'unused.pt', root/'out')
                detector.return_value.predict_candidates.assert_not_called()
            rows = load_predictions(root/'out/shuttle_tracking.jsonl')
            self.assertEqual(rows[1]['status'], 'missing')
            self.assertIsNone(rows[1]['x'])
            self.assertNotIn('raw_detection', rows[0])
            with self.assertRaises(FileExistsError):
                run_demo(video, 'unused.pt', root/'out')
            with patch('src.shuttle.demo.ShuttleDetector') as detector:
                detector.return_value.predict_candidates.return_value = [candidate(10)]
                run_demo(video, 'unused.pt', root/'filtered', temporal_filter=True,
                         tracker_config=TrackerConfig(minimum_quality=.99))
                detector.return_value.predict.assert_not_called()
            filtered = load_predictions(root/'filtered/shuttle_tracking.jsonl')
            self.assertEqual(filtered[0]['raw_detection']['x'], 10)
            self.assertIsNone(filtered[0]['x'])
            self.assertEqual(filtered[0]['status'], 'missing')


class EvaluationTests(unittest.TestCase):
    def test_counts_unknown_and_jump_gaps(self):
        labels = {i: dict(visible=v, x=10, y=10) for i, v in enumerate([1, 1, 0, 0, None, 1])}
        predictions = {i: dict(status='detected', x=x, y=10) for i, x in enumerate([10, 300, 10, 10, 500, 10])}
        predictions[3] = dict(status='missing')
        metrics = evaluate(labels, predictions, 20, 200)
        self.assertEqual([metrics[k] for k in ('tp', 'fp', 'fn', 'tn')], [2, 2, 1, 1])
        self.assertEqual(metrics['large_jumps'], 2)
        self.assertEqual(metrics['skipped_unknown'], 1)
        self.assertEqual(metrics['invisible_fp_rate'], .5)
        self.assertIsNone(evaluate({}, {})['precision'])

    def test_alignment_refuses_wrong_scale_indices_and_times(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder)/'test.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 32))
            writer.write(np.zeros((32, 64, 3), np.uint8))
            writer.release()
            meta = dict(original_width=64, original_height=32, frame_count=1, fps=30)
            labels = {0: dict(visible=1, x=10, y=10)}
            rows = {0: dict(status='detected', x=10, y=10, time_seconds=0)}
            validate_alignment(labels, rows, meta, video, confirm_original_coordinates=True)
            with self.assertRaises(ValueError):
                validate_alignment(labels, rows, meta, video)
            sidecar = dict(width=64, height=32, frame_count=1, fps=30,
                           frame_origin=0, coordinate_system='heatmap_pixels', video=str(video))
            with self.assertRaises(ValueError):
                validate_alignment(labels, rows, meta, video, sidecar)
            for changed in ({1: rows[0]}, {0: dict(rows[0], x=100)},
                            {0: dict(rows[0], time_seconds=1)}):
                with self.assertRaises(ValueError):
                    validate_alignment(labels, changed, meta, video, confirm_original_coordinates=True)
            with self.assertRaises(ValueError):
                validate_alignment(labels, rows, dict(meta, original_width=128), video,
                                   confirm_original_coordinates=True)


if __name__ == '__main__':
    unittest.main()
