import argparse
import json
from pathlib import Path
import cv2

from src.video.reader import open_video
from src.detection.tracker import PlayerTracker
from src.detection.target_selector import select_target
from src.pose.estimator import PlayerPose, draw_pose


def run_demo(video_path, output_path, data_path,
             pose_model="models/pose_landmarker_lite.task", max_frames=None):
    video = open_video(video_path)
    output = None
    pose = None
    frame_number = target_frames = pose_frames = 0
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            fps = 30.0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for path in (output_path, data_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        pose = PlayerPose(pose_model)
        tracker = PlayerTracker()
        output = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not output.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")
        with open(data_path, "w", encoding="utf-8") as file:
            while max_frames is None or frame_number < max_frames:
                success, frame = video.read()
                if not success:
                    break
                target = select_target(tracker.track_people(frame))
                # Estimate on the clean frame, before drawing boxes or text.
                points = pose.extract(frame, target)
                if target is not None:
                    target_frames += 1
                    pose_frames += bool(points)
                    x1, y1, x2, y2 = map(int, target["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"TARGET ID {target['track_id']}",
                                (x1, max(25, y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    draw_pose(frame, points)
                    file.write(json.dumps({"frame": frame_number,
                        "player_id": target["track_id"], "bbox": target["bbox"],
                        "confidence": target["confidence"], "keypoints": points}) + "\n")
                status = "POSE FOUND" if points else ("NO POSE" if target else "NO TARGET")
                cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                output.write(frame)
                frame_number += 1
    finally:
        video.release()
        if output is not None:
            output.release()
        if pose is not None:
            pose.close()
    print(f"Processed {frame_number} frames; target: {target_frames}; pose: {pose_frames}")
    print(f"Saved: {output_path}\nSaved: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracked player to body keypoints and skeleton POC")
    parser.add_argument("video", nargs="?", default="data/sample/singles_short_01.mp4")
    parser.add_argument("--output", default="outputs/player_pose_demo.mp4")
    parser.add_argument("--data", default="outputs/pose.jsonl")
    parser.add_argument("--pose-model", default="models/pose_landmarker_lite.task")
    parser.add_argument("--max-frames", type=int, help="Optional short smoke test")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if len({Path(p).resolve() for p in (args.video, args.output, args.data)}) != 3:
        parser.error("Input video, output video and JSON paths must be different")
    run_demo(args.video, args.output, args.data, args.pose_model, args.max_frames)
