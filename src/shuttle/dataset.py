"""One coordinate transform shared by preparation, labels and inference."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox_transform(original_width, original_height, width, height):
    scale = min(width / original_width, height / original_height)
    resized_width = max(1, min(width, round(original_width * scale)))
    resized_height = max(1, min(height, round(original_height * scale)))
    return {
        "width": width, "height": height,
        "resized_width": resized_width, "resized_height": resized_height,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
        "pad_x": (width - resized_width) // 2,
        "pad_y": (height - resized_height) // 2,
    }


def letterbox(frame, transform):
    t = transform
    resized = cv2.resize(frame, (t["resized_width"], t["resized_height"]))
    canvas = np.zeros((t["height"], t["width"], 3), dtype=np.uint8)
    x, y = t["pad_x"], t["pad_y"]
    canvas[y:y + t["resized_height"], x:x + t["resized_width"]] = resized
    return canvas


def to_heatmap_point(x, y, transform):
    return (x * transform["scale_x"] + transform["pad_x"],
            y * transform["scale_y"] + transform["pad_y"])


def to_original_point(x, y, transform):
    return ((x - transform["pad_x"]) / transform["scale_x"],
            (y - transform["pad_y"]) / transform["scale_y"])


def stack_frames(frames):
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    array = np.concatenate(rgb, axis=2).transpose(2, 0, 1).copy()
    return torch.from_numpy(array).float().div_(255)


def read_labels(path):
    """Blank visibility = not labelled; 0 = absent; 1 = visible position."""
    labels = {}
    with Path(path).open(newline="") as file:
        for raw in csv.DictReader(file):
            row = {str(key).strip().lower(): value for key, value in raw.items()}
            number = int(row["frame"])
            visibility = str(row.get("visible", row.get("visibility", ""))).strip()
            if visibility not in ("", "0", "1"):
                raise ValueError(f"{path}: frame {number} visibility must be blank, 0 or 1")
            visible = None if visibility == "" else int(visibility)
            x = float(row["x"]) if visible == 1 else None
            y = float(row["y"]) if visible == 1 else None
            if visible == 1 and not np.isfinite([x, y]).all():
                raise ValueError(f"{path}: frame {number} has non-finite coordinates")
            labels[number] = {"frame": number, "x": x, "y": y, "visible": visible}
    return labels


def save_labels(path, labels):
    path = Path(path)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "x", "y", "visible"])
        writer.writeheader()
        for number in sorted(labels):
            writer.writerow(labels[number])
    temporary.replace(path)


def frame_path(clip, frame):
    return Path(clip) / "frames" / f"{frame:06d}.jpg"


class ShuttleDataset(Dataset):
    def __init__(self, root, split, augment=False, radius=3):
        self.samples = []
        self.clips = []
        self.groups = set()
        self.dimensions = set()
        self.augment = augment
        self.radius = radius
        for path in sorted(Path(root).glob("*/meta.json")):
            meta = json.loads(path.read_text())
            if meta["split"] != split:
                continue
            clip = path.parent
            labels = read_labels(clip / "labels.csv")
            labelled = [row for row in labels.values() if row["visible"] is not None]
            if not labelled:
                continue
            self.clips.append(str(clip))
            self.groups.add(meta["source_group"])
            self.dimensions.add((meta["transform"]["width"], meta["transform"]["height"]))
            for row in labelled:
                if not 0 <= row["frame"] < meta["frame_count"]:
                    raise ValueError(f"{clip}: label outside clip")
                if row["visible"] == 1 and not (
                    0 <= row["x"] < meta["original_width"]
                    and 0 <= row["y"] < meta["original_height"]
                ):
                    raise ValueError(f"{clip}: frame {row['frame']} label outside original image")
                self.samples.append((clip, meta, row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, meta, row = self.samples[index]
        number = row["frame"]
        indices = [max(0, number - 1), number, min(meta["frame_count"] - 1, number + 1)]
        frames = [cv2.imread(str(frame_path(clip, i))) for i in indices]
        if any(frame is None for frame in frames):
            raise FileNotFoundError(f"Missing prepared frames in {clip} around frame {number}")
        t = meta["transform"]
        if any(frame.shape[:2] != (t["height"], t["width"]) for frame in frames):
            raise ValueError(f"Prepared frame dimensions disagree with {clip}/meta.json")
        target = np.zeros((t["height"], t["width"]), dtype=np.float32)
        if row["visible"] == 1:
            x, y = to_heatmap_point(row["x"], row["y"], t)
            cv2.circle(target, (min(t["width"] - 1, max(0, round(x))),
                                min(t["height"] - 1, max(0, round(y)))),
                       self.radius, 1, -1)
        tensor = stack_frames(frames)
        target = torch.from_numpy(target).unsqueeze(0)
        if self.augment:
            if np.random.random() < 0.5:
                tensor = tensor.flip(-1)
                target = target.flip(-1)
            tensor = (tensor * np.random.uniform(0.8, 1.2)).clamp(0, 1)
        return tensor, target
