"""Boundary-aware / segment-level evaluation metrics (pure logic, no torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal.eval_boundary import (
    frame_exact_accuracy,
    boundary_tolerant_accuracy,
    smoothed_accuracy,
    segment_detection,
    _segments,
    _smooth,
)

# Phase indices (match schema order): 0 ready,1 movement,2 stroke,3 recovery,4 idle


def _tl(trues, preds):
    """Timeline with frames 0..n-1."""
    return [(i, t, p) for i, (t, p) in enumerate(zip(trues, preds))]


def test_frame_exact_matches_hand_count():
    tl = _tl([0, 0, 1, 1], [0, 1, 1, 1])  # 3/4 correct
    assert abs(frame_exact_accuracy([tl]) - 0.75) < 1e-9


def test_boundary_tolerant_k0_equals_frame_exact():
    tl = _tl([0, 0, 1, 1], [0, 1, 1, 1])
    assert boundary_tolerant_accuracy([tl], k=0) == frame_exact_accuracy([tl])


def test_boundary_tolerant_forgives_transition_jitter():
    # True switches 0->1 at frame 2. Prediction switches one frame early (frame
    # 1). Frame-exact: frame 1 wrong (pred 1, true 0). With k=1, frame 1's pred
    # (1) matches the true label at frame 2 -> forgiven.
    trues = [0, 0, 1, 1]
    preds = [0, 1, 1, 1]
    assert boundary_tolerant_accuracy([_tl(trues, preds)], k=0) == 0.75
    assert boundary_tolerant_accuracy([_tl(trues, preds)], k=1) == 1.0


def test_boundary_tolerant_does_not_forgive_far_wrong():
    # A genuinely wrong phase far from any matching true label stays wrong.
    trues = [0, 0, 0, 0]
    preds = [0, 0, 4, 0]  # idle in the middle of a ready stretch
    assert boundary_tolerant_accuracy([_tl(trues, preds)], k=1) == 0.75


def test_smooth_majority_filter():
    # Single-frame flicker removed by a trailing majority window.
    assert _smooth([0, 0, 1, 0, 0], window=3) == [0, 0, 0, 0, 0]
    # window=1 is a passthrough.
    assert _smooth([0, 1, 0], window=1) == [0, 1, 0]


def test_smoothed_accuracy_recovers_flicker():
    trues = [0, 0, 0, 0, 0]
    preds = [0, 0, 1, 0, 0]       # one wrong frame
    assert abs(smoothed_accuracy([_tl(trues, preds)], window=1) - 0.8) < 1e-9
    assert smoothed_accuracy([_tl(trues, preds)], window=3) == 1.0


def test_segments_runs():
    assert _segments([0, 0, 1, 1, 1, 0]) == [(0, 0, 2), (1, 2, 5), (0, 5, 6)]


def test_segment_detection_overlap_threshold():
    # One true segment of class 1 over frames 2..4 (3 frames). Predictions hit
    # class 1 on 2 of 3 -> 0.67 overlap.
    trues = [0, 0, 1, 1, 1]
    preds = [0, 0, 1, 1, 0]
    per_class, overall = segment_detection([_tl(trues, preds)], num_classes=5,
                                           min_overlap=0.5)
    # class 0 segment (frames 0-1): fully hit -> detected. class 1: 2/3 -> >=0.5.
    assert per_class[1] == (1, 1)
    assert per_class[0] == (1, 1)
    # Raise the bar above 0.67 -> class 1 segment no longer detected.
    per_class2, _ = segment_detection([_tl(trues, preds)], num_classes=5,
                                      min_overlap=0.8)
    assert per_class2[1] == (0, 1)


def test_segment_detection_counts_all_segments():
    # Two separate class-0 segments split by a class-1 segment.
    trues = [0, 0, 1, 1, 0, 0]
    preds = [0, 0, 1, 1, 0, 0]
    per_class, overall = segment_detection([_tl(trues, preds)], num_classes=5)
    assert per_class[0] == (2, 2)   # both class-0 segments detected
    assert per_class[1] == (1, 1)
    assert overall == 1.0
