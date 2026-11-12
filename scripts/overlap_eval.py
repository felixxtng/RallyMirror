"""Quantitative before/after check for the pose overlap mitigations (1-3).

Runs the same frames of a clip twice -- once with OverlapConfig mitigations
disabled (the old single-crop behaviour) and once enabled -- and reports, per
clip:

  * frames where BOTH selected players overlap (IoU > 0)
  * among those, how often each player's skeleton centroid lands inside the
    OTHER player's box (the "one player has both skeletons" failure proxy)
  * how many poses the enabled run rejected via centroid validation

This is a measurement harness, not a unit test: it needs the real model + clips
and is meant to be run manually to tune the knobs. Usage:

    .venv/Scripts/python.exe -m scripts.overlap_eval rally05 recorded08 --max-frames 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detection.target_selector import select_targets, load_selection_config
from src.detection.tracker import PlayerTracker
from src.pose.estimator import PlayerPose, OverlapConfig, _iou, _centroid_within_bbox


def _centroid(points, threshold=0.5):
    xs = [p["x"] for p in points if p["visibility"] >= threshold and p["presence"] >= threshold]
    ys = [p["y"] for p in points if p["visibility"] >= threshold and p["presence"] >= threshold]
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def run_pass(video_path, model_path, cfg, selection_config, max_frames):
    video = cv2.VideoCapture(str(video_path))
    pose = PlayerPose(model_path, cfg)
    tracker = PlayerTracker()
    stats = {"overlap_frames": 0, "cross_assignments": 0, "rejected": 0, "frames": 0}
    frame_number = 0
    try:
        while max_frames is None or frame_number < max_frames:
            ok, frame = video.read()
            if not ok:
                break
            targets = select_targets(tracker.track_people(frame), selection_config)
            boxes = [t["bbox"] for t in targets]
            overlapping = len(boxes) >= 2 and _iou(boxes[0], boxes[1]) > 0.0
            if overlapping:
                stats["overlap_frames"] += 1
            for idx, target in enumerate(targets):
                others = [t["bbox"] for j, t in enumerate(targets) if j != idx]
                points = pose.extract(frame, target, others)
                if not points:
                    # Distinguish a validation rejection from a genuine no-pose
                    # by re-running without validation is overkill here; count
                    # empties only when overlapping (the interesting case).
                    if overlapping:
                        stats["rejected"] += 1
                    continue
                if overlapping:
                    c = _centroid(points)
                    if c is not None:
                        for o in others:
                            if _point_in_box(c, o):
                                stats["cross_assignments"] += 1
                                break
            frame_number += 1
    finally:
        pose.close()
        video.release()
    stats["frames"] = frame_number
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--model", default="models/pose_landmarker_lite.task")
    args = ap.parse_args()

    selection_config = load_selection_config()
    off = OverlapConfig(mask_other_players=False, base_padding=0.15,
                        tight_padding=0.15, shrink_at_iou=0.0,
                        validate_centroid_margin=None)
    on = OverlapConfig.from_config()

    for clip in args.clips:
        video_path = Path(f"data/videos/{clip}.mp4")
        print(f"\n=== {clip} (first {args.max_frames} frames) ===")
        for label, cfg in (("OFF (baseline)", off), ("ON (1-3)", on)):
            s = run_pass(video_path, args.model, cfg, selection_config, args.max_frames)
            print(f"  {label:16s} overlap_frames={s['overlap_frames']:4d}  "
                  f"cross_assignments={s['cross_assignments']:4d}  "
                  f"rejected_in_overlap={s['rejected']:4d}")


if __name__ == "__main__":
    main()
