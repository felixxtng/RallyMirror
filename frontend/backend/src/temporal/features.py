"""Sequence-level feature transforms for pose windows.

Velocity is a *temporal* quantity: how far a keypoint moved since the previous
frame. It therefore cannot be computed from a single pose record -- it must be
computed across a window of consecutive frames. This module is the ONE place
that transform lives, so training (``build_windowed_training_dataset``) and
inference (``PhasePipeline``) apply it identically. If they diverged, the model
would see inputs at inference it never trained on.

A "window" here is ``list[list[float]]`` -- ``window_size`` frames, each a flat
coordinate vector ``[x0, y0, x1, y1, ...]`` of length ``BASE_FEATURE_LEN`` (as
produced by ``contracts.pose_record_to_vector`` and buffered by the windower).

``add_velocity`` appends, per keypoint, the per-frame delta ``(vx, vy)`` so each
frame becomes ``[x0, y0, ..., vx0, vy0, ...]`` of length ``2 *
BASE_FEATURE_LEN``. The first frame of the window has no predecessor, so its
velocity is zero.
"""

from __future__ import annotations

from typing import List

from src.temporal.contracts import NUM_KEYPOINTS, SHUTTLE_FEATURE_LEN

# Keypoint coordinates only (x, y per keypoint). The per-frame BASE vector may
# additionally carry court position and/or shuttle rally-state channels, in this
# fixed order (appended by ``contracts.pose_record_to_vector`` BEFORE windowing):
#     [kp coords ...] [court_x, court_y]? [shuttle_speed, shuttle_valid]?
KEYPOINT_FEATURE_LEN = 2 * NUM_KEYPOINTS
COURT_FEATURE_LEN = 2  # normalized court_xy
# SHUTTLE_FEATURE_LEN (speed, validity) is defined in contracts and imported.

# Back-compat alias: historically the base vector was keypoints only.
BASE_FEATURE_LEN = KEYPOINT_FEATURE_LEN


def base_feature_len(use_court: bool = False, use_shuttle: bool = False) -> int:
    """Per-frame length BEFORE the velocity transform (coords, +court, +shuttle)."""
    n = KEYPOINT_FEATURE_LEN
    if use_court:
        n += COURT_FEATURE_LEN
    if use_shuttle:
        n += SHUTTLE_FEATURE_LEN
    return n


def feature_len(use_velocity: bool, use_court: bool = False,
                use_shuttle: bool = False) -> int:
    """Per-frame feature length for the chosen feature set (the TCN's channel
    count).

    Velocity doubles only the keypoint(+court) part. The shuttle channels are
    deliberately EXCLUDED from velocity differencing (shuttle_speed is already a
    velocity-like quantity; the validity flag is categorical), so they are
    carried through once, not doubled.
    """
    kp_court = KEYPOINT_FEATURE_LEN + (COURT_FEATURE_LEN if use_court else 0)
    shuttle = SHUTTLE_FEATURE_LEN if use_shuttle else 0
    return (kp_court * 2 if use_velocity else kp_court) + shuttle


def add_velocity(window: List[List[float]]) -> List[List[float]]:
    """Augment each frame in a window with its per-channel velocity.

    ``window`` is ``window_size`` frames of equal length (oldest -> newest).
    Each output frame is ``frame + velocity`` (double length); velocity for
    frame ``t`` is ``frame[t] - frame[t-1]``, zero for the first frame.

    This differences EVERY channel it is given, so callers that want some
    channels left un-differenced (the shuttle tail) must strip them off first
    and re-append after -- see ``transform_window``.

    Raises
    ------
    ValueError
        If the window is empty or frames differ in length.
    """
    if not window:
        raise ValueError("cannot add velocity to an empty window")
    base_len = len(window[0])
    for i, frame in enumerate(window):
        if len(frame) != base_len:
            raise ValueError(
                f"frame {i} has length {len(frame)}, expected a uniform length "
                f"{base_len}; add_velocity must run on uniform windows, not "
                "ragged ones"
            )

    out: List[List[float]] = []
    prev: List[float] | None = None
    for frame in window:
        if prev is None:
            velocity = [0.0] * base_len
        else:
            velocity = [frame[j] - prev[j] for j in range(base_len)]
        out.append(list(frame) + velocity)
        prev = frame
    return out


def transform_window(window: List[List[float]], use_velocity: bool,
                     use_shuttle: bool = False) -> List[List[float]]:
    """Apply the configured feature transform to a base window.

    The single entry point both training and inference call so they stay in
    lockstep. The per-frame output layout is::

        [kp+court coords] [velocity(kp+court)]? [shuttle_speed, shuttle_valid]?

    Court channels ARE part of the velocity block (court motion is meaningful).
    Shuttle channels are split off the tail BEFORE differencing and re-appended
    untouched, because shuttle_speed is already velocity-like and the validity
    flag must not be differenced. With ``use_velocity`` False and no shuttle,
    this is a passthrough.
    """
    if not window:
        return window
    if use_shuttle:
        # Split the shuttle tail (last SHUTTLE_FEATURE_LEN channels) off each
        # frame, transform the kp+court prefix, then re-append the tail.
        prefix = [frame[:-SHUTTLE_FEATURE_LEN] for frame in window]
        tail = [frame[-SHUTTLE_FEATURE_LEN:] for frame in window]
        transformed = add_velocity(prefix) if use_velocity else prefix
        return [t + s for t, s in zip(transformed, tail)]
    return add_velocity(window) if use_velocity else window
