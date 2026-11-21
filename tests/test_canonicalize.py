"""Canonicalization of fragmented tracker ids into stable slots.

Court path uses a synthetic court->image homography (numpy only); gap path is
pure geometry. No video / YOLO / MediaPipe needed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court.court_model import NET_Y, COURT_WIDTH
from src.court.homography import apply_homography
from src.detection.target_selector import CourtROI
from src.detection.canonicalize import (
    canonicalize_court, canonicalize_gaps, SLOT_A, SLOT_B, REJECTED_SLOT,
)

H_COURT_TO_IMAGE = np.array([[110.0, 35.0, 300.0],
                             [-6.0, -30.0, 900.0],
                             [0.0, 0.0012, 1.0]])
H_IMAGE_TO_COURT = np.linalg.inv(H_COURT_TO_IMAGE)
ROI = CourtROI(H=H_IMAGE_TO_COURT, calib_size=None, margin_m=1.0)


def _bbox_at(court_xy, w=40.0, h=120.0):
    """A bbox whose bottom-centre projects to the given court point."""
    fx, fy = (float(v) for v in apply_homography(H_COURT_TO_IMAGE, court_xy))
    return [fx - w / 2, fy - h, fx + w / 2, fy]


def _rec(frame, pid, court_xy):
    return {"frame": frame, "player_id": pid, "bbox": _bbox_at(court_xy),
            "confidence": 0.9}


class CourtCanonTests(unittest.TestCase):
    def test_handoff_same_half_folds_into_one_slot(self):
        # id 1 near-court frames 0-2, then id 84 near-court frames 3-5.
        recs = [_rec(f, 1, (COURT_WIDTH / 2, 3.0)) for f in range(3)]
        recs += [_rec(f, 84, (COURT_WIDTH / 2, 3.5)) for f in range(3, 6)]
        out = canonicalize_court(recs, ROI)
        slots = {r["player_id"] for r in out}
        self.assertEqual(slots, {SLOT_A})
        # Original ids preserved.
        self.assertEqual({r["raw_player_id"] for r in out}, {1, 84})

    def test_two_players_opposite_halves_get_distinct_slots(self):
        recs = []
        for f in range(5):
            recs.append(_rec(f, 1, (COURT_WIDTH / 2, 3.0)))      # near
            recs.append(_rec(f, 2, (COURT_WIDTH / 2, 10.0)))     # far
        out = canonicalize_court(recs, ROI)
        near = [r for r in out if r["player_id"] == SLOT_A]
        far = [r for r in out if r["player_id"] == SLOT_B]
        self.assertEqual({r["raw_player_id"] for r in near}, {1})
        self.assertEqual({r["raw_player_id"] for r in far}, {2})
        self.assertEqual(len(near), 5)
        self.assertEqual(len(far), 5)

    def test_spectator_in_occupied_half_is_rejected_not_merged(self):
        # Two players present each frame, plus a stray in the NEAR half.
        recs = []
        for f in range(4):
            recs.append(_rec(f, 1, (COURT_WIDTH / 2, 3.0)))      # near player
            recs.append(_rec(f, 2, (COURT_WIDTH / 2, 10.0)))     # far player
            recs.append(_rec(f, 999, (COURT_WIDTH / 2 + 1, 2.0)))  # stray, also near
        out = canonicalize_court(recs, ROI)
        rejected = [r for r in out if r["player_id"] == REJECTED_SLOT]
        # The stray must be rejected, never folded onto slot A or B.
        self.assertEqual({r["raw_player_id"] for r in rejected}, {999})
        # Real slots still carry only their own ids.
        a_raw = {r["raw_player_id"] for r in out if r["player_id"] == SLOT_A}
        b_raw = {r["raw_player_id"] for r in out if r["player_id"] == SLOT_B}
        self.assertEqual(a_raw, {1})
        self.assertEqual(b_raw, {2})

    def test_at_most_one_record_per_slot_per_frame(self):
        recs = []
        for f in range(4):
            recs.append(_rec(f, 1, (COURT_WIDTH / 2, 3.0)))
            recs.append(_rec(f, 2, (COURT_WIDTH / 2, 10.0)))
            recs.append(_rec(f, 50, (COURT_WIDTH / 2 - 1, 3.2)))  # extra near
        out = canonicalize_court(recs, ROI)
        by_frame = {}
        for r in out:
            if r["player_id"] > 0:
                by_frame.setdefault(r["frame"], []).append(r["player_id"])
        for ids in by_frame.values():
            self.assertEqual(len(ids), len(set(ids)))


class GapCanonTests(unittest.TestCase):
    def test_clean_handoff_merges(self):
        # id 1 ends at frame 2 near pixel (500, 400); id 7 starts frame 4 close by.
        recs = [{"frame": f, "player_id": 1, "bbox": [480, 280, 520, 400]}
                for f in range(3)]
        recs += [{"frame": f, "player_id": 7, "bbox": [485, 285, 525, 405]}
                 for f in range(4, 7)]
        out = canonicalize_gaps(recs)
        self.assertEqual({r["player_id"] for r in out}, {SLOT_A})
        self.assertEqual({r["raw_player_id"] for r in out}, {1, 7})

    def test_concurrent_third_track_is_rejected(self):
        # Two ids run concurrently for the whole clip (both slots busy), a third
        # id appears mid-way while both are active -> cannot stitch -> rejected.
        recs = []
        for f in range(10):
            recs.append({"frame": f, "player_id": 1, "bbox": [100, 100, 140, 220]})
            recs.append({"frame": f, "player_id": 2, "bbox": [600, 100, 640, 220]})
        for f in range(4, 7):
            recs.append({"frame": f, "player_id": 3, "bbox": [360, 100, 400, 220]})
        out = canonicalize_gaps(recs)
        rej = {r["raw_player_id"] for r in out if r["player_id"] == REJECTED_SLOT}
        self.assertEqual(rej, {3})


if __name__ == "__main__":
    unittest.main()
