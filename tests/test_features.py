"""Tests for the sequence-level velocity feature transform."""

import pytest

from src.temporal.contracts import (
    NUM_KEYPOINTS, SHUTTLE_FEATURE_LEN, pose_record_to_vector, court_xy_norm,
    shuttle_features,
)
from src.temporal.features import (
    BASE_FEATURE_LEN,
    COURT_FEATURE_LEN,
    KEYPOINT_FEATURE_LEN,
    add_velocity,
    base_feature_len,
    feature_len,
    transform_window,
)


def _coord_frame(val):
    return [val] * BASE_FEATURE_LEN


def test_base_feature_len_is_coords_only():
    assert BASE_FEATURE_LEN == 2 * NUM_KEYPOINTS


def test_feature_len_doubles_with_velocity():
    assert feature_len(False) == BASE_FEATURE_LEN
    assert feature_len(True) == 2 * BASE_FEATURE_LEN


# --- court feature ---------------------------------------------------------

def test_base_and_feature_len_with_court():
    assert base_feature_len(use_court=False) == KEYPOINT_FEATURE_LEN
    assert base_feature_len(use_court=True) == KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN
    # All four combinations of (velocity, court).
    assert feature_len(False, False) == KEYPOINT_FEATURE_LEN
    assert feature_len(True, False) == 2 * KEYPOINT_FEATURE_LEN
    assert feature_len(False, True) == KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN
    assert feature_len(True, True) == 2 * (KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN)


def test_pose_record_to_vector_appends_normalized_court():
    rec = {"frame": 0, "player_id": 1,
           "keypoints": [[0.5, 0.5]] * NUM_KEYPOINTS,
           "court_xy": [3.05, 6.70]}  # court centre in metres
    v_no = pose_record_to_vector(rec, use_court=False)
    v_ct = pose_record_to_vector(rec, use_court=True)
    assert len(v_no) == KEYPOINT_FEATURE_LEN
    assert len(v_ct) == KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN
    # 3.05/6.10 = 0.5, 6.70/13.40 = 0.5.
    assert v_ct[-2:] == [0.5, 0.5]


def test_court_xy_norm_falls_back_to_centre_when_missing():
    assert court_xy_norm({"keypoints": []}) == (0.5, 0.5)
    # Malformed (wrong length / non-numeric) also falls back, never crashes.
    assert court_xy_norm({"court_xy": [1.0]}) == (0.5, 0.5)
    assert court_xy_norm({"court_xy": ["a", "b"]}) == (0.5, 0.5)


def test_add_velocity_covers_court_channels():
    # A court-aware base frame: velocity must span the whole base (incl. court).
    base = [0.0] * (KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN)
    out = add_velocity([base, base, base])
    assert all(len(f) == 2 * (KEYPOINT_FEATURE_LEN + COURT_FEATURE_LEN) for f in out)


# --- shuttle feature -------------------------------------------------------

def test_shuttle_features_reads_and_falls_back():
    rec = {"shuttle_speed": 0.08, "shuttle_speed_valid": 1}
    assert shuttle_features(rec) == (0.08, 1.0)
    # Missing -> (0, 0): unknown, never a false "still shuttle".
    assert shuttle_features({}) == (0.0, 0.0)
    assert shuttle_features({"shuttle_speed": 0.05}) == (0.0, 0.0)  # no valid flag


def test_feature_len_shuttle_not_velocity_doubled():
    K = KEYPOINT_FEATURE_LEN
    S = SHUTTLE_FEATURE_LEN
    # Shuttle channels are carried through once, NOT doubled by velocity.
    assert feature_len(True, False, False) == 2 * K
    assert feature_len(True, False, True) == 2 * K + S
    assert feature_len(False, False, True) == K + S
    assert feature_len(True, True, True) == 2 * (K + COURT_FEATURE_LEN) + S


def test_pose_record_to_vector_shuttle_tail():
    rec = {"frame": 0, "player_id": 1, "keypoints": [[0.5, 0.5]] * NUM_KEYPOINTS,
           "shuttle_speed": 0.08, "shuttle_speed_valid": 1}
    v = pose_record_to_vector(rec, use_shuttle=True)
    assert len(v) == KEYPOINT_FEATURE_LEN + SHUTTLE_FEATURE_LEN
    assert v[-2:] == [0.08, 1.0]


def test_transform_window_leaves_shuttle_tail_undifferenced():
    # Build a court+shuttle base; the shuttle tail must pass through transform
    # unchanged while the kp+court prefix is velocity-doubled.
    K, C, S = KEYPOINT_FEATURE_LEN, COURT_FEATURE_LEN, SHUTTLE_FEATURE_LEN
    base = [0.3] * (K + C) + [0.08, 1.0]
    out = transform_window([base, base, base], use_velocity=True, use_shuttle=True)
    assert len(out[0]) == 2 * (K + C) + S
    # Tail preserved exactly.
    assert out[0][-S:] == [0.08, 1.0]
    assert out[1][-S:] == [0.08, 1.0]
    # The velocity block (kp+court) exists and is zero on the first frame.
    assert out[0][K + C:2 * (K + C)] == [0.0] * (K + C)


def test_add_velocity_first_frame_is_zero():
    window = [_coord_frame(0.2), _coord_frame(0.5)]
    out = add_velocity(window)
    # Each frame doubles in length.
    assert all(len(f) == 2 * BASE_FEATURE_LEN for f in out)
    # First frame velocity half is all zeros.
    assert out[0][BASE_FEATURE_LEN:] == [0.0] * BASE_FEATURE_LEN
    # Coordinate half is preserved.
    assert out[0][:BASE_FEATURE_LEN] == _coord_frame(0.2)


def test_add_velocity_computes_temporal_delta():
    # Second frame moved +0.3 on every coord -> velocity 0.3 everywhere.
    window = [_coord_frame(0.2), _coord_frame(0.5)]
    out = add_velocity(window)
    vel = out[1][BASE_FEATURE_LEN:]
    assert all(abs(v - 0.3) < 1e-9 for v in vel)


def test_add_velocity_is_cross_frame_not_cross_keypoint():
    # Build a frame where keypoints differ WITHIN the frame, but the frame is
    # identical across time -> velocity must be zero (proves it is temporal,
    # not spatial/among-keypoints).
    frame = [float(i) for i in range(BASE_FEATURE_LEN)]
    window = [list(frame), list(frame)]
    out = add_velocity(window)
    assert out[1][BASE_FEATURE_LEN:] == [0.0] * BASE_FEATURE_LEN


def test_add_velocity_rejects_ragged_window():
    # add_velocity now infers the base length from the frames, so the guard is
    # against RAGGED windows (frames of differing length), which is the real
    # corruption risk once the base length is dynamic (court on/off).
    window = [_coord_frame(0.1), [0.0] * (BASE_FEATURE_LEN + 3)]
    with pytest.raises(ValueError):
        add_velocity(window)


def test_add_velocity_rejects_empty_window():
    with pytest.raises(ValueError):
        add_velocity([])


def test_transform_window_passthrough_without_velocity():
    window = [_coord_frame(0.1), _coord_frame(0.2)]
    assert transform_window(window, use_velocity=False) == window


def test_transform_window_applies_velocity_when_enabled():
    window = [_coord_frame(0.1), _coord_frame(0.2)]
    out = transform_window(window, use_velocity=True)
    assert all(len(f) == 2 * BASE_FEATURE_LEN for f in out)
