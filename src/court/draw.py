"""Drawing helpers: court-line overlay on video frames and a top-down minimap."""
from collections import deque

import cv2
import numpy as np

from src.court.court_model import (COURT_LENGTH, COURT_LINES, COURT_POINTS,
                                   COURT_WIDTH, NET_LINE)
from src.court.homography import apply_homography, invert

SCALE = 50   # minimap pixels per metre
PAD = 25
WHITE, ORANGE, GREEN = (255, 255, 255), (0, 165, 255), (60, 110, 60)
PALETTE = [(0, 255, 255), (255, 0, 255), (255, 200, 0), (0, 200, 0), (0, 0, 255)]


def overlay_court(frame, H_image_to_court, color=(0, 255, 0), thickness=2):
    """Project the real court lines into the frame. If they sit on the painted
    lines, the calibration is good. Draws in place and returns the frame."""
    H_inv = invert(H_image_to_court)
    h, w = frame.shape[:2]
    segments = [(a, b, color) for a, b in COURT_LINES] + [(*NET_LINE, ORANGE)]
    for a, b, line_color in segments:
        pa, pb = apply_homography(H_inv, [a, b])
        if not np.all(np.isfinite([pa, pb])):
            continue
        p1, p2 = [tuple(int(v) for v in np.clip(p, -1e6, 1e6).round()) for p in (pa, pb)]
        visible, q1, q2 = cv2.clipLine((0, 0, w, h), p1, p2)
        if visible:
            cv2.line(frame, q1, q2, line_color, thickness, cv2.LINE_AA)
    return frame


def court_to_canvas(xy, scale=SCALE, pad=PAD):
    """Court metres -> minimap pixels. The B0 end (y=0) is at the bottom."""
    x, y = xy
    return int(round(pad + x * scale)), int(round(pad + (COURT_LENGTH - y) * scale))


def make_court_canvas(scale=SCALE, pad=PAD, highlight=None, marked=()):
    """Top-down court picture. `highlight` (a point name) is ringed in red,
    `marked` points are dotted blue (already clicked)."""
    canvas = np.full((int(COURT_LENGTH * scale + 2 * pad),
                      int(COURT_WIDTH * scale + 2 * pad), 3), GREEN, np.uint8)
    for a, b in COURT_LINES:
        cv2.line(canvas, court_to_canvas(a, scale, pad), court_to_canvas(b, scale, pad),
                 WHITE, 2, cv2.LINE_AA)
    cv2.line(canvas, court_to_canvas(NET_LINE[0], scale, pad),
             court_to_canvas(NET_LINE[1], scale, pad), ORANGE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "B0 end", (pad, canvas.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "far end", (pad, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    for name in marked:
        cv2.circle(canvas, court_to_canvas(COURT_POINTS[name], scale, pad), 5, (255, 120, 0), -1)
    if highlight:
        cv2.circle(canvas, court_to_canvas(COURT_POINTS[highlight], scale, pad), 9, (0, 0, 255), 3)
    return canvas


class MiniMap:
    """Top-down minimap that accumulates a short trail per player."""

    def __init__(self, trail_len=45, scale=SCALE, pad=PAD):
        self.scale, self.pad = scale, pad
        self.base = make_court_canvas(scale, pad)
        self.trails = {}
        self.trail_len = trail_len

    def render(self, players):
        """players: iterable of (player_id, (x, y) metres or None) for this frame."""
        for player_id, xy in players:
            if xy is not None:
                self.trails.setdefault(player_id, deque(maxlen=self.trail_len)).append(
                    court_to_canvas(xy, self.scale, self.pad))
        canvas = self.base.copy()
        for player_id, trail in self.trails.items():
            color = PALETTE[int(player_id or 0) % len(PALETTE)]
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, trail[-1], 7, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"ID {player_id}", (trail[-1][0] + 9, trail[-1][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        return canvas


def paste_inset(frame, inset, height_fraction=0.4, margin=10):
    """Paste `inset` in the top-right corner of `frame` (in place)."""
    h, w = frame.shape[:2]
    new_h = max(1, int(h * height_fraction))
    new_w = max(1, int(inset.shape[1] * new_h / inset.shape[0]))
    small = cv2.resize(inset, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if new_w + 2 * margin > w or new_h + 2 * margin > h:
        return frame
    x0, y0 = w - new_w - margin, margin
    frame[y0:y0 + new_h, x0:x0 + new_w] = small
    cv2.rectangle(frame, (x0, y0), (x0 + new_w, y0 + new_h), WHITE, 1)
    return frame
