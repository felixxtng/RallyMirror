"""Map tracked players (pose.jsonl records) into court coordinates (metres)."""
import json
from collections import deque
from pathlib import Path

import numpy as np

from src.court.court_model import in_court_bounds
from src.court.homography import apply_homography

DEFAULTS = {"foot_min_visibility": 0.5, "bounds_margin_m": 1.0,
            "smoothing_window": 5, "smoothing_max_gap_frames": 15}
ANKLES = ("left_ankle", "right_ankle")


def load_config(path="configs/default.yaml"):
    """Read the `court:` section of the YAML config; fall back to DEFAULTS."""
    config = dict(DEFAULTS)
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config.update(data.get("court") or {})
    except (OSError, ImportError):
        pass
    return config


def foot_point(record, min_visibility=0.5):
    """Best ground-contact pixel for a player: ((x, y) or None, source).

    Prefers the ankles (when visible and sane), else the bottom-centre of the box.
    """
    bbox = record.get("bbox")
    if not bbox:
        return None, "none"
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    keypoints = {k["name"]: k for k in (record.get("keypoints") or [])}
    ankles = []
    for name in ANKLES:
        k = keypoints.get(name)
        if (k and k["visibility"] >= min_visibility and k["presence"] >= min_visibility
                and x1 - 0.1 * w <= k["x"] <= x2 + 0.1 * w
                and y1 + 0.5 * h <= k["y"] <= y2 + 0.1 * h):   # must be in the lower half
            ankles.append((k["x"], k["y"]))
    if len(ankles) == 2:
        return ((ankles[0][0] + ankles[1][0]) / 2, (ankles[0][1] + ankles[1][1]) / 2), "ankles"
    if len(ankles) == 1:
        return ankles[0], "ankle"
    return ((x1 + x2) / 2, y2), "bbox"


class CourtSmoother:
    """Causal rolling median per player. Resets after a gap so old data can't leak in."""

    def __init__(self, window=5, max_gap=15):
        self.window, self.max_gap = window, max_gap
        self.history = {}

    def update(self, player_id, frame, xy):
        entry = self.history.setdefault(player_id, {"last": None, "xy": deque(maxlen=self.window)})
        if entry["last"] is not None and frame - entry["last"] > self.max_gap:
            entry["xy"].clear()
        entry["last"] = frame
        entry["xy"].append(tuple(xy))
        return tuple(np.median(np.array(entry["xy"]), axis=0))


def map_record(record, H, smoother, config=None):
    """Return a copy of `record` with foot_px, court_xy_raw, court_xy, in_bounds added."""
    config = {**DEFAULTS, **(config or {})}
    out = dict(record)
    foot, source = foot_point(record, config["foot_min_visibility"])
    out.update(foot_px=None, foot_source=source, court_xy_raw=None, court_xy=None, in_bounds=None)
    if foot is None:
        return out
    out["foot_px"] = [round(float(foot[0]), 2), round(float(foot[1]), 2)]
    raw = apply_homography(H, foot)
    if not np.all(np.isfinite(raw)):
        return out
    smooth = smoother.update(record.get("player_id"), record.get("frame", 0), raw)
    out["court_xy_raw"] = [round(float(raw[0]), 3), round(float(raw[1]), 3)]
    out["court_xy"] = [round(float(smooth[0]), 3), round(float(smooth[1]), 3)]
    out["in_bounds"] = bool(in_court_bounds(out["court_xy"], config["bounds_margin_m"]))
    return out


def map_jsonl(in_path, out_path, H, config=None):
    """Read pose.jsonl, write court.jsonl. Returns a small stats dict."""
    config = {**DEFAULTS, **(config or {})}
    smoother = CourtSmoother(config["smoothing_window"], config["smoothing_max_gap_frames"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stats = {"records": 0, "mapped": 0, "in_bounds": 0,
             "sources": {"ankles": 0, "ankle": 0, "bbox": 0, "none": 0}}
    with open(in_path, encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            mapped = map_record(json.loads(line), H, smoother, config)
            stats["records"] += 1
            stats["sources"][mapped["foot_source"]] += 1
            stats["mapped"] += mapped["court_xy"] is not None
            stats["in_bounds"] += bool(mapped["in_bounds"])
            dst.write(json.dumps(mapped) + "\n")
    return stats
