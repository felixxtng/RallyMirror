from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def track_people(self, frame):
        results = self.model.track(
            frame,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracks = []

        for box in results[0].boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            track_id = int(box.id[0])

            tracks.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

        return tracks
