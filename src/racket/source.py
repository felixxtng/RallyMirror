"""Read original videos or naturally sorted image sequences lazily."""
import re
from pathlib import Path

import cv2


def image_files(path):
    return sorted((p for p in Path(path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                  key=lambda p: [int(t) if t.isdigit() else t.lower()
                                 for t in re.split(r"(\d+)", p.name)])


class FrameSource:
    def __init__(self, meta):
        self.meta = meta
        self.capture = None
        if meta["source_kind"] == "video":
            self.capture = cv2.VideoCapture(meta["source"])
            if not self.capture.isOpened():
                raise ValueError(f"Cannot open {meta['source']}")

    def read(self, frame):
        if not 0 <= frame < self.meta["frame_count"]:
            raise IndexError(frame)
        original = self.meta.get("start_frame", 0) + frame
        if self.capture is not None:
            if int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) != original:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, original)
            ok, image = self.capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {original}")
        else:
            image = cv2.imread(str(Path(self.meta["source"]) / self.meta["images"][original]))
        if image is None or image.shape[:2] != (self.meta["height"], self.meta["width"]):
            raise ValueError(f"Frame {frame}: unreadable image or inconsistent dimensions")
        return image

    def close(self):
        if self.capture is not None:
            self.capture.release()
