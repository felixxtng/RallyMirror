from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def detect_people(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        ##class=[0] is for person class in COCO dataset

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

        return detections
