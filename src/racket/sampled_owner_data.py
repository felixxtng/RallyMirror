"""v006: retain independent located racket queries; H never means a negative."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .mediapipe_model import image_tensor
from .real_owner_data import SIDES, augment_pair, real_ownership_loss, unit_box

FORMAT = 'racket_owner_masked_v006'


class MaskedFrames(Dataset):
    def __init__(self, bundle, records, training=False, size=768, seed=42):
        self.bundle = Path(bundle)
        self.records = records
        self.training = training
        self.size = size
        self.rng = np.random.default_rng(seed)
        if training and any(r['split'] != 'train' or not r['train_eligible'] for r in records):
            raise ValueError('Only eligible training frames may be augmented')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = cv2.imread(str(self.bundle/r['image']))
        if image is None:
            raise ValueError('Unreadable training frame: '+r['id'])
        image = cv2.resize(image, (self.size, self.size))
        poses = np.asarray([r['players'][s]['pose'] for s in SIDES], np.float32)
        boxes = np.asarray([unit_box(r['players'][s]['owner_box'], r['crop'])
                            if r['players'][s]['owner_box'] else [0., 0., 0., 0.] for s in SIDES], np.float32)
        targets = np.asarray([r['players'][s]['target'] if r['players'][s]['valid_target']
                              else [.5, .5] for s in SIDES], np.float32)
        mask = np.asarray([r['players'][s]['valid_target'] for s in SIDES], bool)
        if self.training:
            image, poses, boxes, targets = augment_pair(image, poses, boxes, targets, self.rng)
        return (image_tensor(image), torch.from_numpy(poses.copy()), torch.from_numpy(boxes.copy()),
                torch.from_numpy(targets.copy()), torch.from_numpy(mask.copy()), index)


def select_queries(images, poses, boxes, targets, masks, separations):
    """Unknown queries are never forwarded and cannot affect localization/ranking."""
    valid = masks.flatten()
    rgb = images.repeat_interleave(2, dim=0)[valid]
    pose = poses.flatten(0, 1)[valid]
    box = boxes.flatten(0, 1)[valid]
    target = targets.flatten(0, 1)[valid]
    opponent = targets.flip(1).flatten(0, 1)[valid]
    ranked = (masks & masks.flip(1) & (separations[:, None] > 40)).flatten()[valid]
    return rgb, pose, box, target, opponent, ranked


def masked_loss(logits, target, opponent, ranked, weight=.25):
    if not len(target):
        raise ValueError('No verified located query in this training batch')
    return real_ownership_loss(logits, target, opponent, ranked, weight)
