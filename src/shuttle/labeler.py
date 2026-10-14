"""Click the shuttle centre; labels save immediately and can be resumed."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import frame_path, read_labels, save_labels, to_heatmap_point, to_original_point


class Labeler:
    def __init__(self, clip):
        self.clip = Path(clip)
        self.meta = json.loads((self.clip / "meta.json").read_text())
        self.t = self.meta["transform"]
        self.labels_path = self.clip / "labels.csv"
        self.labels = read_labels(self.labels_path)
        self.count = self.meta["frame_count"]
        for i in range(self.count):
            self.labels.setdefault(i, {"frame": i, "x": None, "y": None, "visible": None})
        self.index = next((i for i in range(self.count) if self.labels[i]["visible"] is None), 0)
        self.mouse = None
        self.cached_index = None
        self.images = None
        self.panel_width = 240
        self.preview_height = 136
        # Fit prepared images onto typical Mac laptop screens; CSV stays at original scale.
        self.display_scale = min(1.0, 960 / self.t["width"], 544 / self.t["height"])
        self.display_width = round(self.t["width"] * self.display_scale)
        self.display_height = round(self.t["height"] * self.display_scale)
        self.window = "RallyMirror - Shuttle labels"

    def save(self):
        save_labels(self.labels_path, self.labels)

    def move(self, amount):
        self.index = min(self.count - 1, max(0, self.index + amount))

    def mouse_event(self, event, x, y, flags, parameter):
        if not (0 <= x < self.display_width and 0 <= y < self.display_height):
            return
        hx = x * self.t["width"] / self.display_width
        hy = y * self.t["height"] / self.display_height
        self.mouse = (round(hx), round(hy))
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = to_original_point(hx, hy, self.t)
            if 0 <= ox < self.meta["original_width"] and 0 <= oy < self.meta["original_height"]:
                self.labels[self.index] = {"frame": self.index, "x": round(ox, 2),
                                           "y": round(oy, 2), "visible": 1}
                self.save()
                self.move(1)

    def load_images(self):
        if self.cached_index != self.index:
            numbers = [self.index, max(0, self.index - 1), min(self.count - 1, self.index + 1)]
            self.images = [cv2.imread(str(frame_path(self.clip, number))) for number in numbers]
            if any(image is None for image in self.images):
                raise FileNotFoundError(f"Missing prepared images around frame {self.index}")
            self.cached_index = self.index

    def draw(self):
        self.load_images()
        main = self.images[0].copy()
        row = self.labels[self.index]
        if row["visible"] == 1:
            x, y = to_heatmap_point(row["x"], row["y"], self.t)
            cv2.circle(main, (round(x), round(y)), 8, (0, 255, 0), 1)
            cv2.drawMarker(main, (round(x), round(y)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        main = cv2.resize(main, (self.display_width, self.display_height))
        canvas_width = max(512, self.display_width + self.panel_width)
        main_area_height = max(500, self.display_height)
        canvas_height = main_area_height + self.preview_height + 58
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:self.display_height, :self.display_width] = main
        if self.mouse is not None:
            x, y = self.mouse
            x = min(self.t["width"] - 1, max(0, x))
            y = min(self.t["height"] - 1, max(0, y))
            crop = self.images[0][max(0, y - 25):min(self.t["height"], y + 26),
                                  max(0, x - 25):min(self.t["width"], x + 26)]
            zoom = cv2.resize(crop, (self.panel_width, self.panel_width), interpolation=cv2.INTER_NEAREST)
            height = min(self.panel_width, self.display_height)
            canvas[:height, self.display_width:self.display_width + self.panel_width] = zoom[:height]
        labelled = sum(row["visible"] is not None for row in self.labels.values())
        status = {None: "UNLABELLED", 0: "NOT VISIBLE", 1: "VISIBLE"}[row["visible"]]
        info = [f"Frame {self.index}/{self.count - 1}", status,
                f"Labelled {labelled}/{self.count}", "Click: label + next", "N: not visible + next",
                "A / D: previous / next", "U: clear label", "S: save   Q: quit"]
        for i, line in enumerate(info):
            cv2.putText(canvas, line, (self.display_width + 8, 265 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        for image, x, name in zip(self.images[1:], [0, 256], ["PREVIOUS", "NEXT"]):
            preview = cv2.resize(image, (240, self.preview_height))
            top = main_area_height + 24
            canvas[top:top + self.preview_height, x:x + 240] = preview
            cv2.putText(canvas, name, (x + 5, main_area_height + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(canvas, "Hover to zoom. Do not guess a hidden shuttle; N marks it absent.",
                    (8, canvas_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        return canvas

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse_event)
        try:
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("a"):
                    self.move(-1)
                elif key in (ord("d"), 32):
                    self.move(1)
                elif key == ord("n"):
                    self.labels[self.index] = {"frame": self.index, "x": None, "y": None, "visible": 0}
                    self.save()
                    self.move(1)
                elif key == ord("u"):
                    self.labels[self.index] = {"frame": self.index, "x": None, "y": None, "visible": None}
                    self.save()
                elif key == ord("s"):
                    self.save()
        finally:
            self.save()
            cv2.destroyAllWindows()
        print(f"Saved {self.labels_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Prepared clip directory")
    args = parser.parse_args()
    Labeler(args.clip).run()


if __name__ == "__main__":
    main()
