"""Dataset that reads labels.json from export_dataset and applies augmentation."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.court.augment import (flip_index, photometric, random_matrix, random_occlusion,
                               resize_matrix, transform_points, warp_image)
from src.court.net import INPUT_HEIGHT, INPUT_WIDTH, make_heatmaps


def load_labels(dataset_dir):
    data = json.loads((Path(dataset_dir) / "labels.json").read_text(encoding="utf-8"))
    return data["point_names"], data["images"]


class CourtDataset(Dataset):
    """One item = (image tensor 3xHxW, heatmaps KxhxW, visibility K).
    split is "train", "val", "test", or "all" (every labelled image)."""

    def __init__(self, dataset_dir, split, train, repeat=1, seed=0):
        self.names, entries = load_labels(dataset_dir)
        self.entries = [e for e in entries if split == "all" or e["split"] == split]
        self.train, self.repeat = train, repeat
        self.rng = np.random.default_rng(seed)
        self.flip = flip_index(self.names)
        self.images = [cv2.imread(str(Path(dataset_dir) / "images" / e["file"])) for e in self.entries]

    def __len__(self):
        return len(self.entries) * self.repeat

    def sample(self, i):
        """Returns (input image uint8 RGB, points in input px (K,2), visible (K,), matrix M)."""
        i %= len(self.entries)
        entry, image = self.entries[i], self.images[i]
        height, width = image.shape[:2]
        points = np.array([entry["points"][n] for n in self.names], dtype=float)
        visible = np.array([entry["visible"][n] for n in self.names], dtype=bool)
        if self.train:
            if self.rng.random() < 0.5:                       # mirror the whole scene
                image = image[:, ::-1]
                points[:, 0] = width - 1 - points[:, 0]
                points, visible = points[self.flip], visible[self.flip]
            M = random_matrix(self.rng, width, height, INPUT_WIDTH, INPUT_HEIGHT)
        else:
            M = resize_matrix(width, height, INPUT_WIDTH, INPUT_HEIGHT)
        out = warp_image(np.ascontiguousarray(image), M, INPUT_WIDTH, INPUT_HEIGHT)
        moved, inside = transform_points(M, points, INPUT_WIDTH, INPUT_HEIGHT)
        if self.train:
            out = random_occlusion(photometric(out, self.rng), self.rng)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), moved, visible & inside, M

    def __getitem__(self, i):
        image, points, visible, _ = self.sample(i)
        heat = make_heatmaps(points, visible)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor, torch.from_numpy(heat), torch.from_numpy(visible.astype(np.float32))
