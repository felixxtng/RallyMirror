"""Tests for the temporal interface: windower, classifier, and pipeline."""

import pytest

from src.temporal import schema
from src.temporal.classifier import FEATURE_LEN, RuleBasedClassifier, PhaseClassifier
from src.temporal.contracts import LEFT_WRIST, NUM_KEYPOINTS, stub_pose_stream
from src.temporal.pipeline import PhasePipeline
from src.temporal.windower import MultiTrackWindower, TrackWindow


def _feat(value=0.0):
    return [value] * FEATURE_LEN


# --- Windower ---------------------------------------------------------------

def test_window_returns_none_until_full():
    w = TrackWindow(window_size=3)
    assert w.push(0, _feat()) is None
    assert w.push(1, _feat()) is None
    out = w.push(2, _feat())
    assert out is not None and len(out) == 3


def test_window_slides_after_full():
    w = TrackWindow(window_size=2)
    w.push(0, _feat(0.0))
    out = w.push(1, _feat(1.0))
    assert out[0][0] == 0.0 and out[1][0] == 1.0
    out2 = w.push(2, _feat(2.0))
    # Oldest dropped; window slid forward.
    assert out2[0][0] == 1.0 and out2[1][0] == 2.0


def test_window_resets_on_frame_gap():
    w = TrackWindow(window_size=3)
    w.push(0, _feat())
    w.push(1, _feat())
    # Gap: frame jumps from 1 to 5 -> buffer resets, not yet full.
    assert w.push(5, _feat()) is None
    assert len(w) == 1


def test_multitrack_isolates_players():
    mw = MultiTrackWindower(window_size=2)
    assert mw.push(player_id=1, frame=0, features=_feat(1.0)) is None
    assert mw.push(player_id=2, frame=0, features=_feat(2.0)) is None
    # Each track independently reaches fullness.
    out1 = mw.push(player_id=1, frame=1, features=_feat(1.0))
    out2 = mw.push(player_id=2, frame=1, features=_feat(2.0))
    assert out1 is not None and out1[0][0] == 1.0
    assert out2 is not None and out2[0][0] == 2.0
    assert set(mw.active_tracks()) == {1, 2}


def test_window_size_validation():
    with pytest.raises(ValueError):
        TrackWindow(0)
    with pytest.raises(ValueError):
        MultiTrackWindower(0)


# --- Rule-based classifier --------------------------------------------------

def test_rule_classifier_implements_protocol():
    assert isinstance(RuleBasedClassifier(), PhaseClassifier)


def test_rule_classifier_outputs_valid_phase():
    clf = RuleBasedClassifier()
    window = [_feat(0.5) for _ in range(10)]
    phase = clf.predict(window)
    assert phase in schema.PHASES


def test_rule_classifier_detects_stroke_on_wrist_swing():
    clf = RuleBasedClassifier(swing_threshold=0.08, movement_threshold=0.04)
    # Build a static pose, then move a wrist sharply across the window.
    base = _feat(0.5)
    window = [list(base) for _ in range(5)]
    # Move the left wrist's x across the window (use the constant so the test
    # tracks the keypoint layout, COCO or MediaPipe).
    for i, frame in enumerate(window):
        frame[2 * LEFT_WRIST] = 0.5 + 0.03 * i  # grows over 5 frames
    assert clf.predict(window) == schema.PHASE_STROKE


def test_rule_classifier_returns_ready_when_static():
    clf = RuleBasedClassifier()
    window = [_feat(0.5) for _ in range(5)]  # no motion -> settled/ready
    assert clf.predict(window) == schema.PHASE_READY


def test_rule_classifier_detects_movement_on_body_shift():
    clf = RuleBasedClassifier(swing_threshold=0.5, movement_threshold=0.04)
    window = []
    for i in range(5):
        frame = _feat(0.3 + 0.02 * i)  # whole body translates, no relative swing
        window.append(frame)
    assert clf.predict(window) == schema.PHASE_MOVEMENT


def test_rule_classifier_rejects_bad_feature_length():
    clf = RuleBasedClassifier()
    with pytest.raises(ValueError):
        clf.predict([[0.0, 0.0, 0.0]])  # wrong length


# --- Pipeline ---------------------------------------------------------------

def test_pipeline_returns_none_before_window_full_then_emits():
    pipe = PhasePipeline(window_size=5)
    stream = list(stub_pose_stream(num_frames=10, player_id=1))
    results = [pipe.process(rec) for rec in stream]
    # First 4 frames: not enough context.
    assert all(r is None for r in results[:4])
    # From frame index 4 onward we get predictions.
    emitted = [r for r in results if r is not None]
    assert len(emitted) == 6
    for frame, player_id, phase in emitted:
        assert player_id == 1
        assert phase in schema.PHASES


def test_pipeline_handles_two_players_independently():
    pipe = PhasePipeline(window_size=3)
    p1 = list(stub_pose_stream(num_frames=6, player_id=1, seed=1))
    p2 = list(stub_pose_stream(num_frames=6, player_id=2, seed=2))
    # Interleave the two players' frames.
    seen = set()
    for a, b in zip(p1, p2):
        for rec in (a, b):
            res = pipe.process(rec)
            if res is not None:
                seen.add(res[1])
    assert seen == {1, 2}


def test_pipeline_smoothing_reduces_flicker():
    # With smoothing, output should be a valid phase and stable across a static
    # stream (no crash, majority logic exercised).
    pipe = PhasePipeline(window_size=3, smoothing=3)
    stream = list(stub_pose_stream(num_frames=12, player_id=1))
    phases = [r[2] for r in (pipe.process(rec) for rec in stream) if r is not None]
    assert phases, "expected some emitted phases"
    assert all(p in schema.PHASES for p in phases)


def test_pipeline_smoothing_validation():
    with pytest.raises(ValueError):
        PhasePipeline(window_size=3, smoothing=0)


def test_pipeline_accepts_custom_classifier():
    class AlwaysStroke:
        def predict(self, window):
            return schema.PHASE_STROKE

    pipe = PhasePipeline(window_size=2, classifier=AlwaysStroke())
    stream = list(stub_pose_stream(num_frames=4, player_id=1))
    out = [r for r in (pipe.process(rec) for rec in stream) if r is not None]
    assert out and all(phase == schema.PHASE_STROKE for _, _, phase in out)
