"""Run the MediaPipe pose estimator without the player tracker."""
import argparse
from pathlib import Path

import cv2

from src.detection.target_selector import select_target
from src.detection.tracker import PlayerTracker
from .estimator import PlayerPose, draw_pose


def run_demo(video_path, model_path, output_path, threshold=0.5, max_frames=None):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not output.isOpened():
        video.release()
        raise OSError(f"Could not create output video: {output_path}")

    pose = PlayerPose(model_path)
    tracker = PlayerTracker()
    detected_frames = 0
    names = set()
    frame_number = 0
    try:
        while max_frames is None or frame_number < max_frames:
            success, frame = video.read()
            if not success:
                break

            # The pose model handles one person, so isolate the foreground player.
            target = select_target(tracker.track_people(frame))
            points = pose.extract(frame, target)
            if points:
                detected_frames += 1
                names.update(point["name"] for point in points)
                draw_pose(frame, points, threshold)

            if target is not None:
                x1, y1, x2, y2 = map(round, target["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

            cv2.putText(
                frame, f"Tracked pose: {len(points)} keypoints", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            output.write(frame)
            frame_number += 1
    finally:
        pose.close()
        video.release()
        output.release()

    if frame_number == 0:
        raise ValueError(f"Video contains no readable frames: {video_path}")

    print(f"Processed {frame_number} frames; pose detected in {detected_frames}")
    print(f"Returned keypoints ({len(names)}): {', '.join(sorted(names)) or 'none'}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("models/pose_landmarker_lite.task")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pose_demo.mp4")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    run_demo(args.video, args.model, args.output, args.threshold, args.max_frames)


if __name__ == "__main__":
    main()
