"""Tests for the segment tagger's non-interactive helpers.

The interactive loop (run_tagger) requires OpenCV + a display and is not unit
tested; these cover the pure helpers that back the display-sizing fix and the
keymap.
"""

from src.temporal import schema
from src.temporal.segment_tagger import (
    MAX_DISPLAY_H,
    MAX_DISPLAY_W,
    _build_keymap,
    _fit_within,
)


def test_fit_within_scales_down_large_1080p():
    # 1920x1080 must shrink to fit the 1280x720 box, preserving aspect ratio.
    w, h = _fit_within(1920, 1080, MAX_DISPLAY_W, MAX_DISPLAY_H)
    assert (w, h) == (1280, 720)
    assert abs((w / h) - (1920 / 1080)) < 1e-6


def test_fit_within_leaves_small_untouched():
    # A clip already within the box is not upscaled.
    assert _fit_within(1280, 720, MAX_DISPLAY_W, MAX_DISPLAY_H) == (1280, 720)
    assert _fit_within(640, 360, MAX_DISPLAY_W, MAX_DISPLAY_H) == (640, 360)


def test_fit_within_preserves_aspect_for_tall_frame():
    w, h = _fit_within(1080, 1920, MAX_DISPLAY_W, MAX_DISPLAY_H)
    # Height is the binding constraint here.
    assert h == 720
    assert abs((w / h) - (1080 / 1920)) < 1e-6


def test_fit_within_handles_degenerate_size():
    assert _fit_within(0, 0, MAX_DISPLAY_W, MAX_DISPLAY_H) == (MAX_DISPLAY_W, MAX_DISPLAY_H)


def test_keymap_covers_every_label_uniquely():
    keymap = _build_keymap()
    assert set(keymap.values()) == set(schema.ANNOTATION_LABELS)
    # No two labels share a key.
    assert len(set(keymap.keys())) == len(schema.ANNOTATION_LABELS)
