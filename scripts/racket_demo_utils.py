"""Batch inference used by the racket comparison demo."""
import math

import numpy as np
import torch

from src.racket.real_owner_data import scene_bounds, selected_pose, unit_box
from src.racket.mediapipe_bundle import square_image, to_original
from src.racket.mediapipe_model import image_tensor, decode


@torch.inference_mode()
def predict_batch(predictor, items):
    images, poses, boxes = [], [], []
    for item in items:
        players = item['players']
        if not players:
            continue
        crop = scene_bounds([p['bbox'] for p in players], predictor.context_padding)
        item['crop'] = crop
        image = image_tensor(square_image(item['frame'], crop, predictor.size))
        for i, player in enumerate(players):
            pose, status = selected_pose(player['keypoints'], player['bbox'],
                [other['bbox'] for j, other in enumerate(players) if i != j], crop)
            player['pose_feature_status'] = status
            images.append(image)
            poses.append(pose)
            boxes.append(unit_box(player['bbox'], crop))
    if not images:
        return
    points, mass = decode(predictor.model(torch.stack(images).to(predictor.device),
        torch.tensor(np.asarray(poses), device=predictor.device),
        torch.tensor(np.asarray(boxes), device=predictor.device)))
    offset = 0
    for item in items:
        for player in item['players']:
            point = to_original(points[offset].cpu().tolist(), item['crop'])
            height, width = item['frame'].shape[:2]
            located = all(math.isfinite(v) for v in point) and 0 <= point[0] < width and 0 <= point[1] < height
            player['racket'] = dict(x=point[0] if located else None, y=point[1] if located else None,
                located=located, concentration=float(mass[offset]), coordinate_system='original_pixels')
            offset += 1
