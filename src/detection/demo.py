import json
import cv2

from src.video.reader import open_video
from src.detection.tracker import PlayerTracker
from src.detection.target_selector import select_target


def run_demo(video_path, output_path, data_path):
    video = open_video(video_path)

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
        target = select_target(tracks)

        if target is not None:
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
        "data/sample/singles_short_01.mp4",
        "outputs/player_tracking_demo.mp4",
        "outputs/tracking.jsonl"
    )