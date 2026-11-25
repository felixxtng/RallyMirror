"""Augmentation for court keypoints. All geometry is numpy-only so it can be unit tested.

Coordinates use the OpenCV convention: pixel centres sit on integers.
"""
import cv2
import numpy as np

from src.court.homography import apply_homography, compute_homography

SWAP = {"DL": "DR", "DR": "DL", "SL": "SR", "SR": "SL"}


def flip_name(name):
    """Mirror a court point name left<->right (DL-B0 <-> DR-B0, C-B0 stays)."""
    x_line, y_line = name.split("-", 1)
    return f"{SWAP.get(x_line, x_line)}-{y_line}"


def flip_index(names):
    """For each point, the index of its mirror twin (used when flipping an image)."""
    return [names.index(flip_name(n)) for n in names]


def resize_matrix(width, height, out_width, out_height):
    return np.diag([out_width / width, out_height / height, 1.0])


def random_matrix(rng, width, height, out_width, out_height, strength=1.0):
    """Random zoom / shift / small rotation / mild perspective, then resize to the net input.

    Maps original-image pixels -> network-input pixels. `strength` scales the randomness.
    """
    cx, cy = width / 2, height / 2
    angle = np.deg2rad(rng.uniform(-6, 6) * strength)
    scale = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25)) * strength))
    shift_range = np.array([0.06, 0.06])
    if rng.random() < 0.35:          # sometimes push the court hard against a frame edge
        shift_range = np.array([0.10, 0.15])
    shift = rng.uniform(-1, 1, 2) * shift_range * [width, height] * strength
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    similarity = np.array([[cos, -sin, cx + shift[0] - cos * cx + sin * cy],
                           [sin, cos, cy + shift[1] - sin * cx - cos * cy],
                           [0, 0, 1.0]])
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    jitter = rng.uniform(-0.05, 0.05, (4, 2)) * [width, height] * strength
    if strength > 0 and rng.random() < 0.3:
        # Fake a lower camera: pull the top edge (far court) down and inwards so the far
        # half is squashed, as in very low or oblique views.
        top_down = rng.uniform(0.05, 0.25) * height * strength
        top_in = rng.uniform(0.0, 0.15) * width * strength
        jitter[0] += [top_in, top_down]
        jitter[1] += [-top_in, top_down]
    perspective = compute_homography(corners, corners + jitter)
    return resize_matrix(width, height, out_width, out_height) @ perspective @ similarity


def transform_points(M, points, out_width, out_height):
    """Apply M to (n, 2) points. Returns (points, inside_mask)."""
    moved = apply_homography(M, np.asarray(points, dtype=float).reshape(-1, 2))
    inside = (np.isfinite(moved).all(axis=1) & (moved[:, 0] >= 0) & (moved[:, 0] <= out_width - 1)
              & (moved[:, 1] >= 0) & (moved[:, 1] <= out_height - 1))
    return moved, inside


def warp_image(image, M, out_width, out_height):
    return cv2.warpPerspective(image, M, (out_width, out_height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


def photometric(image, rng):
    """Brightness/contrast, colour shuffle, grayscale, blur and noise.

    Floor and wall colours differ a lot between venues, so colour changes matter most.
    """
    out = image.astype(np.float32)
    out = out * rng.uniform(0.65, 1.35) + rng.uniform(-30, 30)
    if rng.random() < 0.4:
        out = out[:, :, rng.permutation(3)]
    if rng.random() < 0.15:
        out = np.repeat(out.mean(axis=2, keepdims=True), 3, axis=2)
    out = out * rng.uniform(0.9, 1.1, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        out = np.clip(out + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)
    return out


def random_occlusion(image, rng):
    """Paint flat boxes over the image: a scoreboard-like banner in a bottom corner
    (common in broadcast footage) and 0-2 random patches (people, banners).

    Labels are NOT changed: the court points under a box are still labelled, so the
    network learns to infer them from the rest of the court.
    """
    out = image.copy()
    height, width = out.shape[:2]

    def paint(x0, y0, box_w, box_h):
        out[y0:y0 + box_h, x0:x0 + box_w] = rng.integers(0, 256, 3)

    if rng.random() < 0.3:
        box_w, box_h = int(width * rng.uniform(0.2, 0.35)), int(height * rng.uniform(0.08, 0.15))
        paint(0 if rng.random() < 0.5 else width - box_w, height - box_h, box_w, box_h)
    for _ in range(int(rng.integers(0, 3))):
        box_w, box_h = int(width * rng.uniform(0.05, 0.2)), int(height * rng.uniform(0.04, 0.2))
        paint(int(rng.integers(0, width - box_w + 1)), int(rng.integers(0, height - box_h + 1)),
              box_w, box_h)
    return out
