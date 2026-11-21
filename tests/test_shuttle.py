"""Shuttle rally-state signal: speed computation + validity + guardrails."""
import csv
import os
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.shuttle import (
    load_shuttle_track,
    shuttle_speed_series,
    annotate_pose_with_shuttle,
)

DIAG = math.hypot(960, 544)


def _write_labels(path, rows):
    """rows: list of (frame, x, y, visible) with x/y possibly None."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "visible"])
        for frame, x, y, vis in rows:
            w.writerow([frame, "" if x is None else x, "" if y is None else y, vis])


def test_load_track_marks_invisible_as_none(tmp_path):
    p = str(tmp_path / "labels.csv")
    _write_labels(p, [(0, 100, 100, 1), (1, None, None, 0), (2, 110, 100, 1)])
    track = load_shuttle_track(p)
    assert track[0] == (100.0, 100.0)
    assert track[1] is None
    assert track[2] == (110.0, 100.0)


def test_stationary_visible_shuttle_yields_low_speed(tmp_path):
    # The guardrail: a shuttle lying still on the floor (visible, not moving)
    # must read as near-zero speed -- NOT as "in play". This is the core reason
    # we use speed, not presence.
    p = str(tmp_path / "labels.csv")
    _write_labels(p, [(f, 500.0, 500.0, 1) for f in range(10)])
    track = load_shuttle_track(p)
    series = shuttle_speed_series(track, DIAG, smoothing_span=5)
    # Every frame visible+still: speed ~0, valid 1 (except frame 0 which has no
    # predecessor -> invalid).
    assert series[0] == (0.0, 0.0)
    for f in range(1, 10):
        speed, valid = series[f]
        assert valid == 1.0
        assert speed < 1e-6


def test_fast_shuttle_yields_high_speed(tmp_path):
    p = str(tmp_path / "labels.csv")
    # Moves 100 px/frame -> normalized ~100/diag.
    _write_labels(p, [(f, 100.0 * f, 0.0, 1) for f in range(6)])
    series = shuttle_speed_series(load_shuttle_track(p), DIAG, smoothing_span=3)
    speed, valid = series[5]
    assert valid == 1.0
    assert abs(speed - 100.0 / DIAG) < 1e-6


def test_missing_shuttle_is_unknown_not_still(tmp_path):
    # An absent shuttle must be (0.0, 0.0) -- speed 0 WITH validity 0 -- so it
    # reads as "unknown", never as a real still shuttle.
    p = str(tmp_path / "labels.csv")
    _write_labels(p, [(0, 100, 100, 1), (1, None, None, 0), (2, 100, 100, 1)])
    series = shuttle_speed_series(load_shuttle_track(p), DIAG)
    assert series[1] == (0.0, 0.0)          # missing -> unknown
    assert series[2] == (0.0, 0.0)          # prev missing -> not computable


def test_gap_resets_smoothing(tmp_path):
    # Fast motion, then a missing frame, then one visible frame: the frame right
    # after the gap is not computable (prev missing) and the average must not
    # carry the pre-gap fast values across the gap.
    p = str(tmp_path / "labels.csv")
    _write_labels(p, [(0, 0, 0, 1), (1, 100, 0, 1), (2, 200, 0, 1),
                      (3, None, None, 0), (4, 205, 0, 1), (5, 210, 0, 1)])
    series = shuttle_speed_series(load_shuttle_track(p), DIAG, smoothing_span=5)
    assert series[3] == (0.0, 0.0)           # gap -> unknown
    assert series[4] == (0.0, 0.0)           # prev was gap -> not computable
    # frame 5: only the post-gap 5px step is averaged (not the pre-gap 100px).
    speed5, valid5 = series[5]
    assert valid5 == 1.0
    assert abs(speed5 - 5.0 / DIAG) < 1e-6


def test_annotate_pose_adds_fields_and_stats(tmp_path):
    import json
    from src.temporal.contracts import KEYPOINT_NAMES

    # minimal pose.jsonl (3 frames) + labels.csv
    pose = str(tmp_path / "pose.jsonl")
    with open(pose, "w", encoding="utf-8") as fh:
        for f in range(3):
            kps = [{"name": n, "x": 10.0, "y": 10.0, "visibility": 0.9,
                    "presence": 0.9} for n in KEYPOINT_NAMES]
            fh.write(json.dumps({"frame": f, "player_id": 1,
                                 "bbox": [0, 0, 100, 200], "confidence": 0.9,
                                 "keypoints": kps}) + "\n")
    labels = str(tmp_path / "labels.csv")
    _write_labels(labels, [(0, 100, 100, 1), (1, 160, 100, 1), (2, 220, 100, 1)])
    out = str(tmp_path / "out.jsonl")
    stats = annotate_pose_with_shuttle(pose, labels, out, 960, 544, smoothing_span=5)
    assert stats["records"] == 3
    recs = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert all("shuttle_speed" in r and "shuttle_speed_valid" in r for r in recs)
    assert recs[0]["shuttle_speed_valid"] == 0   # no predecessor
    assert recs[1]["shuttle_speed_valid"] == 1   # moving, computable
