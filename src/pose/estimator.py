"""Single-player pose POC. Coordinates returned in original-frame pixels."""
import math
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NAMES = "nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye right_eye_outer left_ear right_ear mouth_left mouth_right left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_pinky right_pinky left_index right_index left_thumb right_thumb left_hip right_hip left_knee right_knee left_ankle right_ankle left_heel right_heel left_foot_index right_foot_index".split()
# Body, arms, hands, legs and feet (omit facial lines for a clearer demo).
CONNECTIONS = [(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
               (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
               (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
               (24,26),(26,28),(28,30),(30,32),(28,32)]


def crop_bounds(frame, bbox, padding=0.15):
    """Pad the tracked box to keep hands/feet, then clip to image boundaries."""
    x1, y1, x2, y2 = map(float, bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    h, w = frame.shape[:2]
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    bounds = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
              min(w, math.ceil(x2+dx)), min(h, math.ceil(y2+dy)))
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


class PlayerPose:
    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing pose model: {model_path}. See README.md setup.")
        self.model = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE, num_poses=1))

    def extract(self, frame, target):
        if target is None:
            return []
        bounds = crop_bounds(frame, target["bbox"])
        if bounds is None:
            return []
        x1, y1, x2, y2 = bounds
        rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        result = self.model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return []
        # Model gives crop-relative values: multiply by crop size, add its offset.
        return [{"name": name, "x": float(p.x*(x2-x1)+x1),
                 "y": float(p.y*(y2-y1)+y1), "visibility": float(p.visibility),
                 "presence": float(p.presence)}
                for name, p in zip(NAMES, result.pose_landmarks[0])]

    def close(self):
        self.model.close()


def draw_pose(frame, points, threshold=0.5):
    h, w = frame.shape[:2]
    def visible(i):
        p = points[i]
        return (p["visibility"] >= threshold and p["presence"] >= threshold
                and 0 <= p["x"] < w and 0 <= p["y"] < h)
    def xy(i):
        return round(points[i]["x"]), round(points[i]["y"])
    if not points:
        return
    for a, b in CONNECTIONS:
        if visible(a) and visible(b):
            cv2.line(frame, xy(a), xy(b), (0, 255, 255), 2, cv2.LINE_AA)
    for i in range(len(points)):
        if visible(i):
            cv2.circle(frame, xy(i), 3, (255, 0, 255), -1, cv2.LINE_AA)
