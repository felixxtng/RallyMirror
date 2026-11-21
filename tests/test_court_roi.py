"""Metric-space court ROI filter in target_selector (numpy + court only

No video / YOLO / MediaPipe needed: we build a synthetic court->image
homography, invert it, and drive CourtROI / select_targets directly.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court.court_model import COURT_POINTS, COURT_LENGTH, COURT_WIDTH
from src.court.homography import apply_homography
from src.detection.target_selector import (
    CourtROI, SelectionConfig, select_targets, bbox_bottom_centre,
)

# Same style of synthetic perspective camera as test_court_geometry.
H_COURT_TO_IMAGE = np.array([[110.0, 35.0, 300.0],
                             [-6.0, -30.0, 900.0],
                             [0.0, 0.0012, 1.0]])
H_IMAGE_TO_COURT = np.linalg.inv(H_COURT_TO_IMAGE)


def _image_xy(court_xy):
    return tuple(float(v) for v in apply_homography(H_COURT_TO_IMAGE, court_xy))


def _bbox_with_feet_at(court_xy, w=40.0, h=120.0):
    """A bbox whose bottom-centre projects to the given court point."""
    fx, fy = _image_xy(court_xy)
    return [fx - w / 2, fy - h, fx + w / 2, fy]


class CourtROITests(unittest.TestCase):
    def setUp(self):
        self.roi = CourtROI(H=H_IMAGE_TO_COURT, calib_size=None, margin_m=1.0)

    def test_centre_of_court_is_inside(self):
        pt = _image_xy((COURT_WIDTH / 2, COURT_LENGTH / 2))
        self.assertTrue(self.roi.is_inside(pt))

    def test_both_baselines_inside(self):
        for y in (0.0, COURT_LENGTH):
            pt = _image_xy((COURT_WIDTH / 2, y))
            self.assertTrue(self.roi.is_inside(pt))

    def test_far_off_court_is_rejected(self):
        # Well outside the doubles court + 1 m margin.
        pt = _image_xy((COURT_WIDTH / 2, COURT_LENGTH + 5.0))
        self.assertFalse(self.roi.is_inside(pt))
        pt2 = _image_xy((-5.0, COURT_LENGTH / 2))
        self.assertFalse(self.roi.is_inside(pt2))

    def test_margin_widens_acceptance(self):
        just_out = (COURT_WIDTH + 0.6, COURT_LENGTH / 2)  # 0.6 m past the side line
        pt = _image_xy(just_out)
        self.assertFalse(CourtROI(H=H_IMAGE_TO_COURT, margin_m=0.0).is_inside(pt))
        self.assertTrue(CourtROI(H=H_IMAGE_TO_COURT, margin_m=1.0).is_inside(pt))

    def test_frame_size_scaling(self):
        # Calibration fit at 2x the frame size: a point in frame space must be
        # scaled up before mapping. Build the ROI as if calibrated at 2x.
        H_half = H_IMAGE_TO_COURT.copy()
        # Instead of rescaling H, assert the scaling path runs and matches the
        # explicit upscale: calib_size = 2 * frame_size.
        pt_frame = tuple(v / 2 for v in _image_xy((COURT_WIDTH / 2, COURT_LENGTH / 2)))
        roi = CourtROI(H=H_IMAGE_TO_COURT, calib_size=(200, 200))
        # frame half the calib size -> point scaled x2 back to calib space.
        self.assertTrue(roi.is_inside(pt_frame, frame_size=(100, 100)))

    def test_select_targets_drops_off_court_player(self):
        on = {"track_id": 1, "bbox": _bbox_with_feet_at((COURT_WIDTH / 2, 3.0)),
              "confidence": 0.9}
        off = {"track_id": 2, "bbox": _bbox_with_feet_at((COURT_WIDTH / 2, COURT_LENGTH + 6.0)),
               "confidence": 0.9}
        cfg = SelectionConfig(max_targets=4, min_confidence=0.0,
                              min_bbox_height=None, min_aspect_ratio=None,
                              court_roi=self.roi)
        sel = select_targets([on, off], cfg)
        self.assertEqual([t["track_id"] for t in sel], [1])

    def test_no_roi_is_passthrough(self):
        a = {"track_id": 1, "bbox": [10, 10, 50, 130], "confidence": 0.9}
        b = {"track_id": 2, "bbox": [60, 10, 100, 130], "confidence": 0.9}
        cfg = SelectionConfig(max_targets=4, min_confidence=0.0,
                              min_bbox_height=None, min_aspect_ratio=None)
        sel = select_targets([a, b], cfg)
        self.assertEqual(len(sel), 2)


if __name__ == "__main__":
    unittest.main()
