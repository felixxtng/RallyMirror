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

from src.temporal.contracts import NUM_KEYPOINTS

# Coordinates only (x, y per keypoint).
BASE_FEATURE_LEN = 2 * NUM_KEYPOINTS


def feature_len(use_velocity: bool) -> int:
    """Per-frame feature length for the chosen feature set."""
    return BASE_FEATURE_LEN * 2 if use_velocity else BASE_FEATURE_LEN


def add_velocity(window: List[List[float]]) -> List[List[float]]:
    """Augment each coordinate frame in a window with per-keypoint velocity.

    Parameters
    ----------
    window:
        ``window_size`` frames, each a ``BASE_FEATURE_LEN``-long coordinate
        vector (oldest -> newest).

    Returns
    -------
    A new window where each frame is ``coords + velocity`` (length
    ``2 * BASE_FEATURE_LEN``). Velocity for frame ``t`` is ``coords[t] -
    coords[t-1]``; the first frame's velocity is all zeros.

    Raises
    ------
    ValueError
        If any frame is not ``BASE_FEATURE_LEN`` long (guards against being
        called on an already-augmented window).
    """
    for i, frame in enumerate(window):
        if len(frame) != BASE_FEATURE_LEN:
            raise ValueError(
                f"frame {i} has length {len(frame)}, expected coordinate-only "
                f"length {BASE_FEATURE_LEN}; add_velocity must run on raw "
                "coordinate windows, not already-augmented ones"
            )

    out: List[List[float]] = []
    prev: List[float] | None = None
    for frame in window:
        if prev is None:
            velocity = [0.0] * BASE_FEATURE_LEN
        else:
            velocity = [frame[j] - prev[j] for j in range(BASE_FEATURE_LEN)]
        out.append(list(frame) + velocity)
        prev = frame
    return out


def transform_window(window: List[List[float]], use_velocity: bool) -> List[List[float]]:
    """Apply the configured feature transform to a coordinate window.

    The single entry point both training and inference call so they stay in
    lockstep. With ``use_velocity`` False this is a passthrough.
    """
    return add_velocity(window) if use_velocity else window
