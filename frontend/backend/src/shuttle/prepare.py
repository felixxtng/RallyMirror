"""Prepare one continuous rally and optionally import original-resolution labels."""

import argparse
import json
import re
from pathlib import Path

import cv2

from .dataset import frame_path, letterbox, letterbox_transform, read_labels, save_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--name", required=True, help="Unique clip name, e.g. korea_rally01")
    parser.add_argument("--source-group", required=True, help="Same name for all rallies from one match")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--root", type=Path, default=Path("data/shuttle/clips"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive original-video frame index")
    parser.add_argument("--import-labels", type=Path, help="CSV with original-video frame/x/y/visible")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Clip name must contain only letters, digits, underscores or hyphens")
    if args.width < 32 or args.height < 32 or args.width % 8 or args.height % 8:
        parser.error("Width and height must be at least 32 and multiples of 8")
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame <= args.start_frame):
        parser.error("Require 0 <= start-frame < end-frame")
    clip = args.root / args.name
    if clip.exists():
        parser.error(f"{clip} already exists; use the labeller to resume, or choose a new name")
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        parser.error(f"Cannot open {args.video}")
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if original_width <= 0 or original_height <= 0 or fps <= 0:
        video.release()
        parser.error("Video has invalid dimensions or FPS")
    t = letterbox_transform(original_width, original_height, args.width, args.height)
    imported = read_labels(args.import_labels) if args.import_labels else {}
    (clip / "frames").mkdir(parents=True)
    labels = {}
    original_index = 0
    try:
        # Decode sequentially so cropped clip indices agree with original labels.
        while args.end_frame is None or original_index < args.end_frame:
            success, frame = video.read()
            if not success:
                break
            if original_index >= args.start_frame:
                number = original_index - args.start_frame
                if not cv2.imwrite(str(frame_path(clip, number)), letterbox(frame, t),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Cannot save frame {number}")
                row = imported.get(original_index, {"x": None, "y": None, "visible": None})
                labels[number] = {"frame": number, "x": row["x"], "y": row["y"],
                                  "visible": row["visible"]}
                if number % 100 == 0:
                    print(f"Prepared {number + 1} frames", flush=True)
            original_index += 1
    finally:
        video.release()
    if not labels:
        raise ValueError("No frames in the selected interval; choose a valid video interval")
    meta = {
        "video": str(args.video.resolve()), "source_group": args.source_group,
        "split": args.split, "fps": fps, "original_width": original_width,
        "original_height": original_height, "start_frame": args.start_frame,
        "frame_count": len(labels), "transform": t,
    }
    (clip / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    save_labels(clip / "labels.csv", labels)
    print(f"Saved {clip}: {len(labels)} frames, {fps:g} FPS; labels stay in original pixel coordinates")


if __name__ == "__main__":
    main()
