"""Frozen near-side diagnostic retained for comparing the final model's reports.

This reads historical natural crops only; it does not create synthetic players
or expose a separate training pipeline.
"""
from collections import defaultdict
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .mediapipe_bundle import to_original
from .mediapipe_model import decode, image_tensor


def summarize(predictions):
    """Keep missing predictions in the PCK denominator."""
    valid = [p['error_original_pixels'] for p in predictions if p['error_original_pixels'] is not None]
    count = len(predictions)
    return dict(samples=count, missing=count-len(valid), within_20px=sum(e <= 20 for e in valid),
                within_50px=sum(e <= 50 for e in valid),
                pck20_including_misses=sum(e <= 20 for e in valid)/count if count else 0.,
                pck50_including_misses=sum(e <= 50 for e in valid)/count if count else 0.,
                mean_error_detected_only=float(np.mean(valid)) if valid else None,
                median_error_detected_only=float(np.median(valid)) if valid else None,
                coordinate_system='original_video_pixels',
                scope='located near-side centres; not both-racket recall')


class LegacySamples(Dataset):
    def __init__(self, bundle, records):
        self.bundle = Path(bundle)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.bundle/record['image']))
        if image is None:
            raise ValueError('Unreadable crop: '+record['image'])
        x, y, side = record['crop']
        box = np.asarray([(record['owner_box'][0]-x)/side, (record['owner_box'][1]-y)/side,
                          (record['owner_box'][2]-x)/side, (record['owner_box'][3]-y)/side], np.float32)
        return image_tensor(image), torch.tensor(record['pose'], dtype=torch.float32), torch.from_numpy(box), index


@torch.inference_mode()
def evaluate(model, bundle, records, device, batch=8):
    model.eval()
    usable = [r for r in records if r['valid_input']]
    predictions = {}
    for images, poses, boxes, indices in DataLoader(LegacySamples(bundle, usable), batch_size=batch,
                                                   shuffle=False, num_workers=0):
        points, scores = decode(model(images.to(device), poses.to(device), boxes.to(device)))
        for point, score, index in zip(points.cpu().tolist(), scores.cpu().tolist(), indices.tolist()):
            record = usable[index]
            original = to_original(point, record['crop'])
            width, height = record['original_size']
            valid = all(math.isfinite(v) for v in original) and 0 <= original[0] < width and 0 <= original[1] < height
            predictions[record['id']] = dict(id=record['id'], clip=record['clip'], frame=record['frame'],
                source_group=record['source_group'], point_original=original if valid else None,
                target_original=record['target_original'],
                error_original_pixels=math.dist(original, record['target_original']) if valid else None,
                concentration=score, kind='natural')
    rows = [predictions.get(r['id'], dict(id=r['id'], clip=r['clip'], frame=r['frame'],
        source_group=r['source_group'], point_original=None, target_original=r['target_original'],
        error_original_pixels=None, kind='natural')) for r in records]
    metrics = summarize(rows)
    for key in ('source_group', 'clip'):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        metrics['per_'+key] = {name: summarize(group) for name, group in groups.items()}
    return metrics, rows
