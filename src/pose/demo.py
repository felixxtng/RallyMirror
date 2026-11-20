"""Run the MediaPipe pose estimator without the player tracker."""
import argparse
from pathlib import Path

import json
import cv2

from src.detection.target_selector import select_targets, load_selection_config
from src.detection.tracker import PlayerTracker
from src.detection.target_selector import load_court_roi
from .estimator import PlayerPose, OverlapConfig, draw_pose


def _selection_with_court(calibration_path):
    """Load the selection config and, if a court calibration is given, attach a
    metric-space CourtROI so off-court people (spectators / officials) are
    dropped. Missing/absent calibration leaves the ROI stage a pass-through.
    """
    selection_config = load_selection_config()
    if calibration_path and Path(calibration_path).exists():
        selection_config.court_roi = load_court_roi(
            str(calibration_path), selection_config.court_margin_m
        )
        print(f"[pose-demo] court ROI active from {calibration_path}")
    elif calibration_path:
        print(f"[pose-demo] calibration not found ({calibration_path}); ROI off")
    return selection_config


def run_demo(video_path, model_path, output_path, data_path, threshold=0.5,
             max_frames=None, calibration_path=None, racket_weights=None,
             racket_device="cpu"):
    """Run detection + MediaPipe pose, optionally appending a racket keypoint.

    When ``racket_weights`` is given, a :class:`RacketKeypointAnnotator` runs
    once per frame AFTER body pose and appends a ``racket_center`` keypoint to
    each player's ``keypoints`` (last position). This is what produces the
    18-keypoint pose.jsonl a racket-aware temporal model trains on. Without it
    the output is the original 17 body keypoints, unchanged.
    """
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

    pose = PlayerPose(model_path, OverlapConfig.from_config())
    tracker = PlayerTracker()
    selection_config = _selection_with_court(calibration_path)
    racket_annotator = None
    if racket_weights:
        # Imported lazily so the default (no-racket) path stays torch-free here.
        from src.racket.pose_integration import RacketKeypointAnnotator

        racket_annotator = RacketKeypointAnnotator(racket_weights, device=racket_device)
        print(f"[pose-demo] racket keypoint active from {racket_weights}")
    frame_size = (width, height)
    detected_frames = 0
    names = set()
    frame_number = 0
    tracking_data = []

    try:
        while max_frames is None or frame_number < max_frames:
            success, frame = video.read()
            if not success:
                break

            # The pose model handles one person at a time, so run it per
            # selected player and emit one record per player per frame.
            targets = select_targets(tracker.track_people(frame), selection_config,
                                     frame_size)

            # Pass 1: body pose per player. Collect the players that got a pose;
            # the racket model (if enabled) needs ALL of them for one frame at
            # once (shared scene crop, per-player cross-validation), so defer
            # drawing and recording until after the racket pass.
            players = []
            for idx, target in enumerate(targets):
                # Pass the OTHER selected players so the estimator can mask /
                # tighten / validate against overlap.
                others = [t["bbox"] for j, t in enumerate(targets) if j != idx]
                points = pose.extract(frame, target, others)
                x1, y1, x2, y2 = map(round, target["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                if points:
                    players.append({
                        "player_id": target["track_id"],
                        "bbox": target["bbox"],
                        "confidence": target["confidence"],
                        "keypoints": points,
                    })

            # Pass 2 (optional): append the racket_center keypoint to every
            # player that got a body pose this frame.
            if racket_annotator is not None and players:
                players = racket_annotator.annotate(frame, players)

            # Pass 3: draw + record using the (possibly racket-augmented) points.
            frame_keypoints = 0
            for player in players:
                points = player["keypoints"]
                frame_keypoints += len(points)
                names.update(point["name"] for point in points)
                draw_pose(frame, points, threshold)
                tracking_data.append({
                    "frame": frame_number,
                    "player_id": player["player_id"],
                    "bbox": player["bbox"],
                    "confidence": player["confidence"],
                    "keypoints": points,
                })

            if players:
                detected_frames += 1

            cv2.putText(
                frame, f"Tracked pose: {frame_keypoints} keypoints", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            output.write(frame)
            frame_number += 1

    finally:
        pose.close()
        video.release()
        output.release()

    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as file:
        for item in tracking_data:
            file.write(json.dumps(item) + "\n")

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
    parser.add_argument(
        "--data", type=Path, default=Path("outputs/pose.jsonl")
    )
    parser.add_argument(
        "--calibration", type=Path
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--racket-weights", type=Path, default=None,
        help="racket ownership checkpoint; when given, append a racket_center "
             "keypoint per player (produces an 18-keypoint pose.jsonl)")
    parser.add_argument("--racket-device", default="cpu")
    args = parser.parse_args()
    run_demo(args.video, args.model, args.output, args.data, args.threshold, args.max_frames,
             args.calibration if args.calibration and args.calibration.exists() else None,
             racket_weights=str(args.racket_weights) if args.racket_weights else None,
             racket_device=args.racket_device)

if __name__ == "__main__":
    main()
