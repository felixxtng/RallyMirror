"""End-to-end check that the metric court ROI drops off-court detections.

Runs the real tracker over a clip and, for each frame, compares selection with
the court ROI OFF vs ON (same config otherwise). Reports how many raw tracks
were off-court and how the final selected set changes. Needs the project venv.

    .venv/Scripts/python.exe -m scripts.roi_check rally05 \
        --calibration outputs/court-calibration_pred/rally05.json --max-frames 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detection.target_selector import (
    load_selection_config, load_court_roi, select_targets, bbox_bottom_centre,
)
from src.detection.tracker import PlayerTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    video_path = Path(f"data/videos/{args.clip}.mp4")
    video = cv2.VideoCapture(str(video_path))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    tracker = PlayerTracker()

    cfg_off = load_selection_config()
    cfg_on = load_selection_config()
    cfg_on.court_roi = load_court_roi(args.calibration, cfg_on.court_margin_m)

    stats = {"frames": 0, "raw_tracks": 0, "offcourt_raw": 0,
             "kept_off": 0, "kept_on": 0, "frames_with_offcourt_in_off": 0}
    frame_number = 0
    while frame_number < args.max_frames:
        ok, frame = video.read()
        if not ok:
            break
        tracks = tracker.track_people(frame)
        stats["raw_tracks"] += len(tracks)

        # How many raw tracks are off-court (ROI judges each one)?
        off_raw = 0
        for t in tracks:
            if not cfg_on.court_roi.is_inside(bbox_bottom_centre(t["bbox"]), frame_size):
                off_raw += 1
        stats["offcourt_raw"] += off_raw

        sel_off = select_targets(tracks, cfg_off, frame_size)
        sel_on = select_targets(tracks, cfg_on, frame_size)
        stats["kept_off"] += len(sel_off)
        stats["kept_on"] += len(sel_on)

        # Did the OFF selection include anyone the ROI calls off-court?
        off_ids = {id(t) for t in sel_off}
        on_ids = {id(t) for t in sel_on}
        dropped = off_ids - on_ids
        if dropped:
            stats["frames_with_offcourt_in_off"] += 1
        frame_number += 1
    video.release()
    stats["frames"] = frame_number

    print(f"clip={args.clip} frames={stats['frames']}")
    print(f"  raw tracks total      : {stats['raw_tracks']}")
    print(f"  raw tracks off-court  : {stats['offcourt_raw']}")
    print(f"  selected (ROI off)    : {stats['kept_off']}")
    print(f"  selected (ROI on)     : {stats['kept_on']}")
    print(f"  frames where ROI removed an off-court pick from the ROI-off set: "
          f"{stats['frames_with_offcourt_in_off']}")


if __name__ == "__main__":
    main()
