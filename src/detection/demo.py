import json
import cv2

from src.video.reader import open_video
from src.detection.tracker import PlayerTracker
from src.detection.target_selector import select_targets, load_selection_config


def run_demo(video_path, output_path, data_path):
    video = open_video(video_path)

    # Selection thresholds (confidence + geometry safeguards) come from
    # configs/default.yaml -> detection.target_selection.
    selection_config = load_selection_config()

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker = PlayerTracker()
    frame_number = 0
    tracking_data = []

    while True:
        success, frame = video.read()

        if not success:
            break

        tracks = tracker.track_people(frame)
        targets = select_targets(tracks, selection_config)

        for target in targets:
            x1, y1, x2, y2 = map(int, target["bbox"])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )

            cv2.putText(
                frame,
                f"TARGET ID {target['track_id']}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            tracking_data.append({
                "frame": frame_number,
                "player_id": target["track_id"],
                "bbox": target["bbox"],
                "confidence": target["confidence"]
            })

        output.write(frame)
        frame_number += 1

    video.release()
    output.release()

    with open(data_path, "w") as file:
        for item in tracking_data:
            file.write(json.dumps(item) + "\n")

    print(f"Saved: {output_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    run_demo(
        "data/videos/rally01.mp4",
        "outputs/rally01_tracking_demo.mp4",
        "outputs/tracking.jsonl"
    )