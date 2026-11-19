"""Shared body-feature geometry and file utilities for the racket model."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

BODY_NAMES = 'nose left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_hip right_hip left_knee right_knee left_ankle right_ankle left_heel right_heel left_foot_index right_foot_index'.split()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def square_bounds(box):
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:
        raise ValueError('Invalid person box')
    side = int(math.ceil(max((y2-y1)*1.8, (x2-x1)*2.5)))
    return [math.floor((x1+x2-side)/2), math.floor((y1+y2-side)/2), side]


def to_unit(point, crop):
    return [(point[0]-crop[0])/crop[2], (point[1]-crop[1])/crop[2]]


def to_original(point, crop):
    return [point[0]*crop[2]+crop[0], point[1]*crop[2]+crop[1]]


def square_image(image, crop, size):
    x,y,side = crop
    # Warp the square directly, avoiding huge temporary arrays for edge players.
    matrix = np.float32([[size/side,0,-x*size/side], [0,size/side,-y*size/side]])
    return cv2.warpAffine(image,matrix,(size,size),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))




def pose_array(points, crop):
    by_name={p['name']:p for p in points}
    result=[]
    for name in BODY_NAMES:
        p=by_name.get(name)
        if p is None:
            result.append([0.,0.,0.]); continue
        x,y=to_unit([p['x'],p['y']],crop)
        score=max(0.,min(1.,p.get('visibility',0)*p.get('presence',0)))
        if not all(math.isfinite(v) for v in (x,y,score)):
            result.append([0.,0.,0.]); continue
        result.append([x,y,score if 0<=x<1 and 0<=y<1 else 0.])
    return result
