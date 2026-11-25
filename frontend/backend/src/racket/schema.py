"""Versioned original-pixel annotations, immutable source references, atomic saves."""
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

NAMES = ["handle_end", "shaft_head_joint", "head_center"]
CENTER_NAMES = ["head_center"]
SPLITS = {"train", "val", "test"}


def group_id(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows scanners can briefly lock a just-written annotation during rapid
    # autosaves. Keep the old file intact and retry only those sharing failures.
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 5:
                raise
            time.sleep(.02*(attempt+1))


def blank(frame, names=None):
    return {"frame": frame, "racket_id": "foreground", "state": "unlabelled",
            "provenance": "manual", "approved": False,
            "keypoints": [{"name": name, "x": None, "y": None, "v": None} for name in (names or NAMES)]}


def point_errors(row, width, height, names=None):
    errors = []
    points = row.get("keypoints", [])
    if [p.get("name") for p in points] != (names or NAMES):
        return ["missing points or inconsistent keypoint order"]
    for p in points:
        if isinstance(p.get("v"), bool) or p.get("v") not in (0, 1, 2):
            errors.append(f"{p['name']}: missing/invalid visibility")
        elif p["v"] == 0:
            if p.get("x") is not None or p.get("y") is not None:
                errors.append(f"{p['name']}: unavailable points require null coordinates")
        elif not all(isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool)
                     and math.isfinite(p[k]) for k in ("x", "y")):
            errors.append(f"{p['name']}: missing/nonfinite coordinates")
        elif not (0 <= p["x"] < width and 0 <= p["y"] < height):
            errors.append(f"{p['name']}: coordinates outside original image")
    if row.get("state") == "invisible" and any(p.get("v") != 0 for p in points):
        errors.append("invisible racket has located points")
    if row.get("state") == "visible" and not any(p.get("v") in (1, 2) for p in points):
        errors.append("visible racket has no located points")
    if row.get("state") == "unavailable" and any(p.get("v") != 0 for p in points):
        errors.append("unavailable centre has located points")
    if row.get("state") not in {"visible", "invisible", "unavailable"}:
        errors.append("annotation incomplete")
    return errors


def anchors_current(row, rows):
    return all(str(frame) in rows and digest(rows[str(frame)]) == signature
               for frame, signature in row.get("anchors", {}).items())


def approved(row, meta, rows):
    return (row.get("approved") is True and not point_errors(row, meta["width"], meta["height"], meta.get("keypoint_names"))
            and anchors_current(row, rows))


def load_clip(clip):
    clip = Path(clip)
    return read_json(clip / "meta.json"), read_json(clip / "annotations.json")


def copy_annotation(row, frame):
    result = copy.deepcopy(row)
    result.update(frame=frame, provenance="copied", approved=False)
    result.pop("anchors", None)
    result.pop("review_reasons", None)
    return result
