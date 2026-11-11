"""Court geometry checks (numpy only; no video, YOLO or MediaPipe needed)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court.calibration import build_calibration, load_calibration, save_calibration
from src.court.court_model import (COURT_LENGTH, COURT_POINTS, COURT_WIDTH,
                                   in_court_bounds)
from src.court.homography import (apply_homography, compute_homography,
                                  leave_one_out_errors_m, reprojection_error_px)
from src.court.mapper import CourtSmoother, foot_point, map_record

# A made-up but realistic camera: court -> image homography (perspective, no lens distortion).
H_COURT_TO_IMAGE = np.array([[110.0, 35.0, 300.0],
                             [-6.0, -30.0, 900.0],
                             [0.0, 0.0012, 1.0]])
H_IMAGE_TO_COURT = np.linalg.inv(H_COURT_TO_IMAGE)
NAMES = ["DL-B0", "DR-B0", "DL-B1", "DR-B1", "DL-SS0", "DR-SS0", "DL-SS1", "DR-SS1"]


def to_image(names):
    return apply_homography(H_COURT_TO_IMAGE, [COURT_POINTS[n] for n in names])


class CourtModelTests(unittest.TestCase):
    def test_bwf_dimensions(self):
        self.assertEqual(COURT_POINTS["DR-B1"], (COURT_WIDTH, COURT_LENGTH))
        self.assertAlmostEqual(COURT_POINTS["DL-SS0"][1], 4.72)
        self.assertAlmostEqual(COURT_POINTS["DL-SS1"][1], 8.68)
        self.assertAlmostEqual(COURT_POINTS["DL-LS0"][1], 0.76)
        self.assertAlmostEqual(COURT_POINTS["SL-B0"][0], 0.46)
        self.assertAlmostEqual(COURT_POINTS["C-B0"][0], 3.05)
        self.assertEqual(COURT_POINTS["DL-NET"], (0.0, 6.70))

    def test_bounds(self):
        self.assertTrue(in_court_bounds((3, 6.7)))
        self.assertFalse(in_court_bounds((-0.5, 6.7)))
        self.assertTrue(in_court_bounds((-0.5, 6.7), margin=1.0))


class HomographyTests(unittest.TestCase):
    def test_exact_round_trip(self):
        H = compute_homography(to_image(NAMES), [COURT_POINTS[n] for n in NAMES])
        test_court = np.array([[3.05, 6.7], [1.0, 2.0], [5.5, 12.0]])
        pixels = apply_homography(H_COURT_TO_IMAGE, test_court)
        np.testing.assert_allclose(apply_homography(H, pixels), test_court, atol=1e-6)

    def test_click_noise_stays_small(self):
        rng = np.random.default_rng(0)
        noisy = to_image(NAMES) + rng.normal(0, 1.0, (len(NAMES), 2))   # ~1 px click error
        H = compute_homography(noisy, [COURT_POINTS[n] for n in NAMES])
        centre = apply_homography(H_COURT_TO_IMAGE, [3.05, 6.7])
        self.assertLess(np.linalg.norm(apply_homography(H, centre) - [3.05, 6.7]), 0.15)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compute_homography([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])  # 3 points
        line = [[0, 0], [1, 1], [2, 2], [3, 3]]
        with self.assertRaises(ValueError):
            compute_homography(line, [[0, 0], [1, 0], [1, 1], [0, 1]])             # collinear

    def test_error_measures(self):
        court = [COURT_POINTS[n] for n in NAMES]
        H = compute_homography(to_image(NAMES), court)
        self.assertLess(reprojection_error_px(H, to_image(NAMES), court).max(), 1e-6)
        self.assertLess(leave_one_out_errors_m(to_image(NAMES), court).max(), 1e-6)
        bad = to_image(NAMES)
        bad[2] += [40, -40]                      # one wild mis-click
        self.assertGreater(leave_one_out_errors_m(bad, court).max(), 0.3)


class CalibrationTests(unittest.TestCase):
    def test_build_save_load(self):
        calib = build_calibration(NAMES, to_image(NAMES), "clip.mp4", 0, (1280, 720))
        self.assertEqual(calib["stats"]["n_points"], 8)
        self.assertLess(calib["stats"]["loo_max_m"], 1e-6)
        self.assertEqual(calib["warnings"], [])
        with tempfile.TemporaryDirectory() as tmp:
            save_calibration(Path(tmp) / "c.json", calib)
            loaded = load_calibration(Path(tmp) / "c.json")
        np.testing.assert_allclose(loaded["H"], np.array(calib["H_image_to_court"]))

    def test_four_points_warns_and_unknown_name_fails(self):
        calib = build_calibration(NAMES[:4], to_image(NAMES[:4]))
        self.assertTrue(calib["warnings"])
        with self.assertRaises(ValueError):
            build_calibration(["NOPE"] + NAMES[:3], to_image(NAMES[:4]))


def kp(name, x, y, vis=0.9):
    return {"name": name, "x": x, "y": y, "visibility": vis, "presence": vis}


class MapperTests(unittest.TestCase):
    BBOX = [100, 100, 200, 300]

    def test_foot_point_sources(self):
        both = {"bbox": self.BBOX, "keypoints": [kp("left_ankle", 130, 280), kp("right_ankle", 170, 290)]}
        self.assertEqual(foot_point(both), ((150.0, 285.0), "ankles"))
        one = {"bbox": self.BBOX, "keypoints": [kp("left_ankle", 130, 280), kp("right_ankle", 170, 290, 0.1)]}
        self.assertEqual(foot_point(one), ((130, 280), "ankle"))
        self.assertEqual(foot_point({"bbox": self.BBOX, "keypoints": []}), ((150.0, 300), "bbox"))
        wild = {"bbox": self.BBOX, "keypoints": [kp("left_ankle", 130, 120)]}   # ankle near head
        self.assertEqual(foot_point(wild)[1], "bbox")
        self.assertEqual(foot_point({}), (None, "none"))

    def test_smoother_median_and_gap_reset(self):
        smoother = CourtSmoother(window=3, max_gap=5)
        for frame, x in enumerate([1.0, 1.0, 9.0]):          # 9.0 is a glitch
            result = smoother.update(1, frame, (x, 0.0))
        self.assertEqual(result, (1.0, 0.0))
        self.assertEqual(smoother.update(1, 50, (4.0, 4.0)), (4.0, 4.0))   # long gap: fresh start

    def test_map_record_end_to_end(self):
        foot_court = (2.0, 3.0)
        foot_px = apply_homography(H_COURT_TO_IMAGE, foot_court)
        record = {"frame": 7, "player_id": 3,
                  "bbox": [foot_px[0] - 40, foot_px[1] - 200, foot_px[0] + 40, foot_px[1]],
                  "keypoints": []}
        out = map_record(record, H_IMAGE_TO_COURT, CourtSmoother())
        self.assertEqual(out["foot_source"], "bbox")
        np.testing.assert_allclose(out["court_xy"], foot_court, atol=0.01)
        self.assertTrue(out["in_bounds"])
        self.assertEqual(out["player_id"], 3)               # original fields preserved

    def test_far_outside_court_is_flagged(self):
        px = apply_homography(H_COURT_TO_IMAGE, (-4.0, 3.0))
        record = {"frame": 0, "player_id": 1, "bbox": [px[0] - 10, px[1] - 80, px[0] + 10, px[1]]}
        self.assertFalse(map_record(record, H_IMAGE_TO_COURT, CourtSmoother())["in_bounds"])


if __name__ == "__main__":
    unittest.main()
