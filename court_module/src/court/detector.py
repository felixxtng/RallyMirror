"""Automatic baseline for a visible green badminton court, viewed end-on.

White-line candidates are fitted to the complete court model. No clicked points
or fixed image coordinates are used. Other floor colours and camera angles need
additional detection methods; this is deliberately a first baseline.
"""
from itertools import product

import cv2
import numpy as np

from .court_model import COURT_LENGTH, COURT_LINES, COURT_WIDTH
from .homography import apply_homography


def court_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (32, 30, 35), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or max(map(cv2.contourArea, contours)) < frame.shape[0] * frame.shape[1] * 0.08:
        raise ValueError("No large green court region found.")
    contour = max(contours, key=cv2.contourArea)
    region = np.zeros(green.shape, np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(contour), 255)
    region = cv2.dilate(region, np.ones((9, 9), np.uint8))
    # Distant markings seen through the net are dimmer and greener. Local
    # brightness contrast recovers them without including the pale green floor.
    contrast = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    white = cv2.inRange(hsv, (0, 0, 140), (179, 70, 255))
    white = cv2.bitwise_and(white, cv2.inRange(contrast, 12, 255))
    white = cv2.bitwise_and(white, region)
    return region, white, cv2.boundingRect(contour)


def _merge_lines(segments, horizontal, reference):
    """Merge nearby collinear Hough segments, then refit their endpoints."""
    groups = []
    for segment in sorted(segments, key=lambda s: -np.linalg.norm(s[2:] - s[:2])):
        x1, y1, x2, y2 = segment
        slope = (y2 - y1) / (x2 - x1) if horizontal else (x2 - x1) / (y2 - y1)
        intercept = y1 - slope * x1 if horizontal else x1 - slope * y1
        position = slope * reference + intercept
        for group in groups:
            if abs(position - group[0]) < 6 and abs(slope - group[1]) < 0.06:
                group[2].extend(segment.reshape(2, 2))
                group[3] += np.linalg.norm(segment[2:] - segment[:2])
                break
        else:
            groups.append([position, slope, list(segment.reshape(2, 2)),
                           np.linalg.norm(segment[2:] - segment[:2])])
    lines = []
    for _, _, points, strength in sorted(groups, key=lambda g: -g[3])[:12]:
        vx, vy, x, y = cv2.fitLine(np.array(points, np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        line = np.array([-vy, vx, vy * x - vx * y], dtype=float)
        lines.append((line, float(strength)))
    return lines


def _intersection(a, b):
    p = np.cross(a, b)
    return p[:2] / p[2] if abs(p[2]) > 1e-8 else np.array([np.nan, np.nan])


def detect_court(frame, min_score=0.58):
    """Return a fitted calibration candidate and diagnostic images.

    Score is image evidence, NOT a calibrated probability or accuracy estimate.
    Detection rejects weak fits rather than silently exporting a guessed court.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width)
    image = cv2.resize(frame, None, fx=scale, fy=scale)
    h, w = image.shape[:2]
    region, white, (gx, gy, gw, gh) = court_masks(image)
    edges = cv2.Canny(white, 50, 150)
    found = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=45,
                            minLineLength=max(35, int(w * 0.07)), maxLineGap=25)
    if found is None:
        raise ValueError("No white line segments found.")
    horizontal, side = [], []
    candidates = image.copy()
    for segment in found.reshape(-1, 4).astype(float):
        dx, dy = segment[2:] - segment[:2]
        if abs(dx) > 1 and abs(dy / dx) < 0.20:
            horizontal.append(segment)
        elif abs(dy) > 1 and abs(dx / dy) < 1.2:
            side.append(segment)
        cv2.line(candidates, tuple(segment[:2].astype(int)), tuple(segment[2:].astype(int)), (0, 0, 255), 1)
    horizontal = _merge_lines(horizontal, True, w / 2)
    sides = _merge_lines(side, False, gy + gh / 2)
    def x_at(line, y):
        return -(line[1] * y + line[2]) / line[0]
    def y_at(line, x):
        return -(line[0] * x + line[2]) / line[1]
    left = [line for line, _ in sides if x_at(line, gy + gh / 2) < gx + gw * 0.45]
    right = [line for line, _ in sides if x_at(line, gy + gh / 2) > gx + gw * 0.55]
    top = [line for line, _ in horizontal if gy - 15 <= y_at(line, gx + gw / 2) < gy + gh * 0.30]
    bottom = [line for line, _ in horizontal if gy + gh * 0.70 < y_at(line, gx + gw / 2) <= gy + gh + 15]
    distance = cv2.distanceTransform(255 - white, cv2.DIST_L2, 3)
    fractions = np.linspace(0.04, 0.96, 70)
    samples = np.array([np.array(a)[None, :] * (1 - fractions[:, None])
                        + np.array(b)[None, :] * fractions[:, None] for a, b in COURT_LINES])
    model_corners = np.float32([[0, 0], [COURT_WIDTH, 0],
                                [0, COURT_LENGTH], [COURT_WIDTH, COURT_LENGTH]])
    best = None
    for l, r, t, b in product(left, right, top, bottom):
        corners = np.array([_intersection(l, b), _intersection(r, b),
                            _intersection(l, t), _intersection(r, t)], np.float32)
        if not np.isfinite(corners).all():
            continue
        if (corners[:, 0].min() < -w * 0.05 or corners[:, 0].max() > w * 1.05
                or corners[:, 1].min() < 0 or corners[:, 1].max() >= h
                or corners[1, 0] - corners[0, 0] < gw * 0.50
                or corners[3, 0] - corners[2, 0] < gw * 0.20):
            continue
        H = cv2.getPerspectiveTransform(model_corners, corners)
        projected = apply_homography(H, samples.reshape(-1, 2)).reshape(samples.shape)
        if not np.isfinite(projected).all():
            continue
        px = np.rint(projected).astype(int)
        valid = (px[:, :, 0] >= 0) & (px[:, :, 0] < w) & (px[:, :, 1] >= 0) & (px[:, :, 1] < h)
        support = np.zeros(valid.shape)
        support[valid] = np.exp(-(distance[px[:, :, 1][valid], px[:, :, 0][valid]] / 3.0) ** 2)
        scores = support.mean(axis=1)
        score = float(scores.mean())
        if best is None or score > best[0]:
            best = (score, corners, H, scores)
    diagnostics = {"region": region, "white_mask": white, "candidates": candidates}
    if best is None:
        return None, diagnostics, "No plausible full-court boundary found."
    score, corners, H, scores = best
    if score < min_score or scores[:4].min() < 0.40:
        return None, diagnostics, f"Weak court fit (line support {score:.3f}); inspect the masks or try another frame."
    # H maps metres -> resized pixels. Convert back to original pixels first.
    image_to_court = np.linalg.inv(H) @ np.diag([scale, scale, 1.0])
    image_to_court /= image_to_court[2, 2]
    result = {"H_image_to_court": image_to_court.tolist(),
              "image_size": [width, height], "corners_image": (corners / scale).tolist(),
              "method": "green_region_white_lines_court_template",
              "stats": {"line_support_score": score, "per_line_support": scores.tolist()},
              "warnings": ["Baseline assumes a visible green court viewed end-on. Verify the overlay.",
                           "Line support is not an independent accuracy measurement."],
              "orientation": "B0 is the near end; left/right as viewed from that end."}
    return result, diagnostics, None
