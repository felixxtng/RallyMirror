"""Court-space demo: map pose.jsonl to metres and render overlay + minimap video.

    python -m src.court.demo data/sample/clip.mp4 --calibration calibration/clip.json

Inputs : the video, outputs/pose.jsonl (from src.detection.demo) and a calibration JSON.
Outputs: outputs/court.jsonl (pose records + court fields) and outputs/court_demo.mp4.

Fields added to every record: foot_px, foot_source (ankles | ankle | bbox),
court_xy_raw and court_xy (metres, x across / y along the court), in_bounds.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from src.court.calibration import load_calibration
from src.court.draw import MiniMap, overlay_court, paste_inset
from src.court.mapper import load_config, map_jsonl
from src.video.reader import open_video


def run_demo(video_path, pose_path, calibration_path, output_path, data_path,
             config_path="configs/default.yaml", max_frames=None):
    calibration = load_calibration(calibration_path)
    H = calibration["H"]
    stats = map_jsonl(pose_path, data_path, H, load_config(config_path))
    by_frame = defaultdict(list)
    with open(data_path, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            by_frame[record["frame"]].append(record)

    video = open_video(video_path)
    output = None
    frame_number = 0
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            fps = 30.0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = calibration.get("image_size")
        if size and tuple(size) != (width, height):
            raise ValueError(f"Calibration was made at {tuple(size)} but video is "
                             f"{(width, height)}. Re-calibrate this clip.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not output.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")
        minimap = MiniMap()
        while max_frames is None or frame_number < max_frames:
            success, frame = video.read()
            if not success:
                break
            overlay_court(frame, H)
            records = by_frame.get(frame_number, [])
            for record in records:
                if record["foot_px"] is None:
                    continue
                x, y = map(int, record["foot_px"])
                cv2.circle(frame, (x, y), 6, (255, 255, 0), -1, cv2.LINE_AA)
                if record["court_xy"] is not None:
                    label = f"({record['court_xy'][0]:.1f}, {record['court_xy'][1]:.1f}) m"
                    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2, cv2.LINE_AA)
            paste_inset(frame, minimap.render(
                [(r["player_id"], r["court_xy"]) for r in records]))
            output.write(frame)
            frame_number += 1
    finally:
        video.release()
        if output is not None:
            output.release()
    print(f"Rendered {frame_number} frames. Records: {stats['records']}, mapped: "
          f"{stats['mapped']}, in bounds: {stats['in_bounds']}, foot sources: {stats['sources']}")
    print(f"Saved: {output_path}\nSaved: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player pose records to court coordinates")
    parser.add_argument("video")
    parser.add_argument("--calibration", required=True, help="JSON from src.court.calibrate")
    parser.add_argument("--pose-jsonl", default="outputs/pose.jsonl")
    parser.add_argument("--output", default="outputs/court_demo.mp4")
    parser.add_argument("--data", default="outputs/court.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    paths = {Path(p).resolve() for p in (args.video, args.pose_jsonl, args.output, args.data)}
    if len(paths) != 4:
        parser.error("Video, pose JSON, output video and output JSON paths must all differ")
    run_demo(args.video, args.pose_jsonl, args.calibration, args.output, args.data,
             args.config, args.max_frames)
