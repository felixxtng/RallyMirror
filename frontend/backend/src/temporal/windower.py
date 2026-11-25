"""Temporal windowing for phase recognition.

A single frame cannot tell you whether a player is "moving forward" or
"recovering backward" -- you need a sequence. This module buffers per-frame pose
feature vectors (one per player track) into fixed-length sliding windows that the
phase classifier consumes.

Design points:
- Per-track isolation: each ``player_id`` has its own independent buffer. In
  singles there are two players; their sequences must never be mixed.
- Online operation: frames arrive one at a time (streaming), matching how the
  demo/real pipeline feeds data. ``push`` returns the current window (or ``None``
  until enough frames have accumulated).
- Gap handling: if frames for a track are non-consecutive (a track dropped out
  and came back), the buffer is reset so a window never spans a discontinuity.
  Phase reasoning across a gap would be meaningless.

The window is a list of equal-length feature vectors (oldest -> newest). The
classifier decides what to do with it; the windower stays model-agnostic.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional


class TrackWindow:
    """Fixed-length rolling buffer of feature vectors for a single track."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buffer: Deque[List[float]] = deque(maxlen=window_size)
        self._last_frame: Optional[int] = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_frame = None

    def push(self, frame: int, features: List[float]) -> Optional[List[List[float]]]:
        """Add a frame's features; return the current full window or ``None``.

        If ``frame`` is not exactly one greater than the previous frame for this
        track (a gap or out-of-order arrival), the buffer resets first so no
        window spans a discontinuity.
        """
        if self._last_frame is not None and frame != self._last_frame + 1:
            # Discontinuity: start a fresh sequence at this frame.
            self.reset()
        self._buffer.append(list(features))
        self._last_frame = frame
        if len(self._buffer) == self.window_size:
            return [list(f) for f in self._buffer]
        return None

    def is_full(self) -> bool:
        return len(self._buffer) == self.window_size

    def __len__(self) -> int:
        return len(self._buffer)


class MultiTrackWindower:
    """Maintains an independent :class:`TrackWindow` per ``player_id``."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._windows: Dict[int, TrackWindow] = {}

    def push(
        self,
        player_id: int,
        frame: int,
        features: List[float],
    ) -> Optional[List[List[float]]]:
        """Push one frame of features for ``player_id``.

        Returns the current full window for that track, or ``None`` if the
        track has not yet accumulated ``window_size`` consecutive frames.
        """
        window = self._windows.get(player_id)
        if window is None:
            window = TrackWindow(self.window_size)
            self._windows[player_id] = window
        return window.push(frame, features)

    def reset_track(self, player_id: int) -> None:
        if player_id in self._windows:
            self._windows[player_id].reset()

    def reset_all(self) -> None:
        self._windows.clear()

    def active_tracks(self) -> List[int]:
        return list(self._windows.keys())
