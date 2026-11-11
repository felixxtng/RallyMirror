"""Manual twelve-point annotation; net-top points stay separate from floor points."""
import copy
from pathlib import Path

import cv2
import numpy as np

from .draw import make_court_canvas
from .learning_data import (LEARNING_DESCRIPTIONS, POINT_NAMES, complete, load_manifest,
                            representative_indices, save_manifest)


class CourtLabeler:
    def __init__(self, root, one_per_view=False):
        self.root = Path(root)
        self.data = load_manifest(root)
        if not self.data["images"]:
            raise ValueError("Prepare some images first")
        self.stamp = (self.root / "annotations.json").stat().st_mtime_ns
        self.indices = representative_indices(self.data["images"]) if one_per_view else list(range(len(self.data["images"])))
        self.index = next((i for i in self.indices
                           if not self.data["images"][i]["unusable"] and not complete(self.data["images"][i])), self.indices[0])
        self.selected = None
        self.history = []
        self.mouse = None
        self.cached_index = None
        self.window = "RallyMirror - Manual court labels"
        self.choose_pending()

    @property
    def record(self):
        return self.data["images"][self.index]

    def choose_pending(self):
        self.selected = next((i for i, n in enumerate(POINT_NAMES)
                              if self.record["points"][n]["status"] == "unlabelled"), None)

    def save(self):
        path = self.root / "annotations.json"
        if path.stat().st_mtime_ns != self.stamp:
            raise RuntimeError("Annotations changed in another session. Close this window and reopen it.")
        save_manifest(self.root, self.data)
        self.stamp = path.stat().st_mtime_ns

    def move(self, amount):
        position = self.indices.index(self.index)
        self.index = self.indices[min(len(self.indices) - 1, max(0, position + amount))]
        self.choose_pending()
        self.mouse = None

    def annotate(self, xy=None):
        if self.selected is None or self.record["unusable"]:
            return
        name = POINT_NAMES[self.selected]
        self.history.append((self.index, self.selected, copy.deepcopy(self.record["points"][name])))
        self.record["points"][name] = {"status": "visible" if xy is not None else "hidden", "xy": xy}
        self.save()
        self.choose_pending()

    def undo(self):
        if self.history:
            self.index, self.selected, previous = self.history.pop()
            self.record["points"][POINT_NAMES[self.selected]] = previous
            self.save()

    def load_image(self):
        if self.cached_index != self.index:
            self.image = cv2.imread(str(self.root / self.record["image"]))
            if self.image is None:
                raise ValueError(f"Missing {self.record['image']}")
            h, w = self.image.shape[:2]
            self.scale = min(1.0, 960 / w, 540 / h)
            self.dw, self.dh = round(w * self.scale), round(h * self.scale)
            self.cached_index = self.index

    def on_mouse(self, event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and self.dw <= x < self.dw + 320 and 360 <= y < 636:
            selected = (y - 360) // 23
            if selected < len(POINT_NAMES):
                self.selected = selected
            return
        if not (0 <= x < self.dw and 42 <= y < 42 + self.dh):
            return
        ox = min(self.record["width"] - 1, x * self.record["width"] / self.dw)
        oy = min(self.record["height"] - 1, (y - 42) * self.record["height"] / self.dh)
        self.mouse = (ox, oy)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.annotate([round(ox, 2), round(oy, 2)])

    def draw(self):
        self.load_image()
        canvas = np.full((max(720, self.dh + 175), self.dw + 320, 3), 28, np.uint8)
        image = cv2.resize(self.image, (self.dw, self.dh))
        for i, name in enumerate(POINT_NAMES):
            p = self.record["points"][name]
            if p["status"] == "visible":
                xy = (round(p["xy"][0] * self.dw / self.record["width"]),
                      round(p["xy"][1] * self.dh / self.record["height"]))
                cv2.drawMarker(image, xy, (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(image, str(i + 1), (xy[0] + 8, xy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        canvas[42:42 + self.dh, :self.dw] = image
        def text(line, x, y, color=(230, 230, 230), scale=0.50):
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        name = POINT_NAMES[self.selected] if self.selected is not None else None
        prompt = f"Click {name}: {LEARNING_DESCRIPTIONS[name]}" if name else "All 12 points reviewed. Enter = next image; click sidebar to edit."
        if self.record["unusable"]:
            prompt = "IMAGE EXCLUDED. X = restore; D = next image."
        text(prompt, 8, 28, (0, 255, 255), 0.52)
        text(f"Image {self.indices.index(self.index) + 1}/{len(self.indices)}: {self.record['id']}", 8, self.dh + 67)
        text("Click = label   H = hidden   U = undo   Click sidebar = select/edit point", 8, self.dh + 95)
        text("A/D = previous/next image   Enter = next when complete   X = exclude image", 8, self.dh + 122)
        text("Q/Esc = quit (each edit saves automatically). Hover to zoom.", 8, self.dh + 149)
        marked = [n for n in POINT_NAMES if n not in ("NL-TOP", "NR-TOP") and self.record["points"][n]["status"] == "visible"]
        diagram = cv2.resize(make_court_canvas(highlight=name if name not in ("NL-TOP", "NR-TOP") else None,
                                              marked=marked), (150, 305))
        canvas[20:325, self.dw + 10:self.dw + 160] = diagram
        # Separate front-view sketch for raised net points; the plan-view orange
        # line represents the net's floor position, not the top of the net.
        xleft, xright = self.dw + 185, self.dw + 295
        cv2.line(canvas, (xleft, 220), (xright, 220), (0, 165, 255), 2)
        for x, label in ((xleft, "NL-TOP"), (xright, "NR-TOP")):
            cv2.line(canvas, (x, 220), (x, 315), (180, 180, 180), 2)
            cv2.circle(canvas, (x, 220), 6, (0, 0, 255) if name == label else (0, 165, 255), 2)
        cv2.line(canvas, (xleft-10, 315), (xright+10, 315), (150, 150, 150), 1)
        text("Net: front view", self.dw + 170, 190, scale=0.43)
        text("9 left / 10 right TOP", self.dw + 165, 340, scale=0.40)
        text("Near = B0", self.dw + 8, 348)
        for i, n in enumerate(POINT_NAMES):
            status = self.record["points"][n]["status"]
            text(f"{i+1} {n}: {status}", self.dw + 8, 375 + i * 23,
                 (0, 255, 255) if i == self.selected else (230, 230, 230))
        if self.mouse:
            x, y = map(round, self.mouse)
            left, top = max(0, x-25), max(0, y-25)
            crop = self.image[top:y+26, left:x+26]
            zoom = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_NEAREST)
            cursor = (round((x-left) * 140 / crop.shape[1]), round((y-top) * 140 / crop.shape[0]))
            cv2.drawMarker(zoom, cursor, (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
            canvas[20:160, self.dw + 170:self.dw + 310] = zoom
        text("1-9, 0, [, ] select points 1-12", self.dw + 8, 660, scale=0.43)
        text("Net TOP points are above the floor", self.dw + 8, 683, (0, 165, 255), scale=0.43)
        return canvas

    def run(self):
        self.load_image()
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)
        try:
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("h"):
                    self.annotate()
                elif key == ord("u"):
                    self.undo()
                elif ord("1") <= key <= ord("9"):
                    self.selected = key - ord("1")
                elif key in (ord("0"), ord("["), ord("]")):
                    self.selected = {ord("0"): 9, ord("["): 10, ord("]"): 11}[key]
                elif key == ord("a"):
                    self.move(-1)
                elif key == ord("d") or key in (10, 13) and (complete(self.record) or self.record["unusable"]):
                    self.move(1)
                elif key == ord("x"):
                    self.record["unusable"] = not self.record["unusable"]
                    self.save()
        finally:
            cv2.destroyAllWindows()
        print(f"Labels saved in {self.root / 'annotations.json'}")
