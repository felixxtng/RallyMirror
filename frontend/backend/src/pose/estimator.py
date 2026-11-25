"""Single-player pose POC. Coordinates returned in original-frame pixels."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NAMES = "nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye right_eye_outer left_ear right_ear mouth_left mouth_right left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_pinky right_pinky left_index right_index left_thumb right_thumb left_hip right_hip left_knee right_knee left_ankle right_ankle left_heel right_heel left_foot_index right_foot_index".split()
# Keep the nose as the head point, plus the main body joints and feet. Other
# facial and finger landmarks add noise without helping the player-pose view.
REMOVED_LANDMARKS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22}
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]


@dataclass
class OverlapConfig:
    """Mitigations for overlapping players confusing the single-crop estimator.

    Top-down pose fills a crop with exactly one skeleton, so when two players'
    boxes overlap the model can lock onto the wrong body or merge both into one.
    These knobs (1) mask other players out of the crop, (2) tighten crop padding
    where boxes overlap, and (3) validate the returned skeleton against the
    player's own box. Defaults mirror ``configs/default.yaml -> pose.overlap``.
    """

    mask_other_players: bool = True
    base_padding: float = 0.15
    tight_padding: float = 0.0
    shrink_at_iou: float = 0.15
    validate_centroid_margin: Optional[float] = 0.35

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "OverlapConfig":
        """Build from a loaded RallyMirror config dict (``pose.overlap`` section).

        ``None`` loads ``configs/default.yaml``. Unknown keys are ignored and a
        missing section yields plain defaults, so this never raises just because
        the YAML has not been updated.
        """
        if config is None:
            from src.temporal.config import load_config

            config = load_config()
        section = (config.get("pose") or {}).get("overlap") or {}
        known = {f.name for f in fields(cls)}
        overrides = {k: v for k, v in section.items() if k in known}
        return cls(**overrides)


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _adaptive_padding(bbox: Sequence[float], others: Sequence[Sequence[float]],
                      cfg: OverlapConfig) -> float:
    """(2) Interpolate padding from base -> tight based on the max IoU with any
    other player's box. No overlap keeps the full hands/feet padding; heavy
    overlap shrinks the crop so it stops bleeding into the neighbour.
    """
    if not others or cfg.shrink_at_iou <= 0:
        return cfg.base_padding
    max_iou = max((_iou(bbox, o) for o in others), default=0.0)
    if max_iou <= 0:
        return cfg.base_padding
    frac = min(max_iou / cfg.shrink_at_iou, 1.0)
    return cfg.base_padding + (cfg.tight_padding - cfg.base_padding) * frac


def crop_bounds(frame, bbox, padding=0.15):
    """Pad the tracked box to keep hands/feet, then clip to image boundaries."""
    x1, y1, x2, y2 = map(float, bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    h, w = frame.shape[:2]
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    bounds = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
              min(w, math.ceil(x2+dx)), min(h, math.ceil(y2+dy)))
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def _mask_others(crop, bounds, others):
    """(1) Paint other players' box regions inside this crop a flat colour so
    the single-pose model has far less of the neighbour to latch onto. ``crop``
    is modified in place; coordinates in ``others`` are full-frame pixels and
    are translated into crop-local space via ``bounds`` offset.
    """
    cx1, cy1, cx2, cy2 = bounds
    ch, cw = crop.shape[:2]
    for ox1, oy1, ox2, oy2 in others:
        # Clip the other box to the crop, then shift into crop-local coords.
        mx1 = int(max(cx1, math.floor(ox1)) - cx1)
        my1 = int(max(cy1, math.floor(oy1)) - cy1)
        mx2 = int(min(cx2, math.ceil(ox2)) - cx1)
        my2 = int(min(cy2, math.ceil(oy2)) - cy1)
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(cw, mx2), min(ch, my2)
        if mx2 > mx1 and my2 > my1:
            crop[my1:my2, mx1:mx2] = 0
    return crop


def _centroid_within_bbox(points, bbox, margin, threshold=0.5):
    """(3) True if the visible keypoints' centroid sits within ``bbox`` expanded
    by ``margin`` (fraction of box size). A merged/wrong skeleton usually lands
    its centre on the neighbour and fails this. Returns True (accept) when there
    are no visible points to judge -- that is the caller's empty-pose path.
    """
    xs, ys = [], []
    for p in points:
        if p["visibility"] >= threshold and p["presence"] >= threshold:
            xs.append(p["x"])
            ys.append(p["y"])
    if not xs:
        return True
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (x1 - mx) <= cx <= (x2 + mx) and (y1 - my) <= cy <= (y2 + my)


class PlayerPose:
    def __init__(self, model_path="models/pose_landmarker_lite.task",
                 overlap_config=None):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing pose model: {model_path}. See README.md setup.")
        self.overlap = overlap_config or OverlapConfig()
        self.model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE, num_poses=1))

    def extract(self, frame, target, others=None):
        """Estimate the pose of ``target``, mitigating overlap with ``others``.

        ``others`` is an iterable of the OTHER selected players' bboxes
        ([x1, y1, x2, y2], full-frame pixels). When given, the overlapping
        regions are masked out of the crop (1), crop padding is tightened where
        boxes overlap (2), and the returned skeleton is validated against the
        target's own box (3). With ``others`` empty the behaviour matches the
        original single-player path.
        """
        if target is None:
            return []
        cfg = getattr(self, "overlap", None) or OverlapConfig()
        other_boxes = [list(o) for o in (others or [])]

        padding = _adaptive_padding(target["bbox"], other_boxes, cfg)
        bounds = crop_bounds(frame, target["bbox"], padding)
        if bounds is None:
            return []
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2].copy()  # copy so masking never touches the frame
        if cfg.mask_other_players and other_boxes:
            _mask_others(crop, bounds, other_boxes)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return []
        # Model gives crop-relative values: multiply by crop size, add its offset.
        points = [{"name": name, "x": float(p.x*(x2-x1)+x1),
                   "y": float(p.y*(y2-y1)+y1), "visibility": float(p.visibility),
                   "presence": float(p.presence)}
                  for i, (name, p) in enumerate(zip(NAMES, result.pose_landmarks[0]))
                  if i not in REMOVED_LANDMARKS]

        # (3) Reject a skeleton whose centroid lands outside the player's own
        # box (typically a merged/wrong pose grabbed from the neighbour).
        if cfg.validate_centroid_margin is not None:
            if not _centroid_within_bbox(points, target["bbox"], cfg.validate_centroid_margin):
                return []
        return points

    def close(self):
        self.model.close()


def draw_pose(frame, points, threshold=0.5):
    h, w = frame.shape[:2]
    by_name = {p["name"]: p for p in points}
    def visible(name):
        p = by_name[name]
        return (p["visibility"] >= threshold and p["presence"] >= threshold
                and 0 <= p["x"] < w and 0 <= p["y"] < h)
    def xy(name):
        return round(by_name[name]["x"]), round(by_name[name]["y"])
    if not points:
        return
    for a, b in CONNECTIONS:
        if a in by_name and b in by_name and visible(a) and visible(b):
            cv2.line(frame, xy(a), xy(b), (0, 255, 255), 2, cv2.LINE_AA)
    for name in by_name:
        if visible(name):
            cv2.circle(frame, xy(name), 3, (255, 0, 255), -1, cv2.LINE_AA)
