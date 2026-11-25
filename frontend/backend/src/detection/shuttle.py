"""Shuttle rally-state signal: per-frame smoothed shuttle SPEED + validity.

What this writes
----------------
For each pose record (matched to the shuttle track by frame index), two fields:
  * ``shuttle_speed``        -- smoothed, normalized per-frame displacement of
                                the shuttle (fraction of the frame diagonal, so
                                it is resolution-independent like the pose
                                coords). 0.0 when not computable.
  * ``shuttle_speed_valid``  -- 1 if speed was computable this frame (shuttle
                                visible this frame AND the previous one), else 0.

The validity flag matters: the shuttle is missing ~21% of frames (mid-rally
occlusion OR off-screen between points). A missing shuttle must read as
"unknown" (speed 0, valid 0), NOT as a real "still" shuttle -- otherwise an
occluded mid-rally frame would look like idle. The model sees both channels and
learns "unknown" rather than being told a false zero.

Smoothing
---------
Raw per-frame displacement is noisy: the shuttle detector flickers, and a single
frame's position can jump. A short causal moving-average (``smoothing_span``
frames) over the raw speed bridges that flicker while staying responsive, and is
decoupled from the TCN window size.

Coordinate space
----------------
``labels.csv`` coordinates live in the clip's letterboxed detection space (e.g.
960x544, per ``meta.json``), NOT the pose/bbox space. Speed is normalized by the
diagonal of THAT space, so it is comparable across clips without needing to map
shuttle pixels into any other frame.
"""
from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default smoothing span (frames) for the causal moving average over raw speed.
DEFAULT_SMOOTHING_SPAN = 5


def load_shuttle_track(labels_csv: str) -> Dict[int, Optional[Tuple[float, float]]]:
    """Read ``labels.csv`` -> ``{frame: (x, y) or None}``.

    A frame maps to ``None`` when the shuttle is not visible (``visible != 1``
    or empty/again-missing coordinates). Robust to blank cells.
    """
    track: Dict[int, Optional[Tuple[float, float]]] = {}
    with open(labels_csv, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                frame = int(row["frame"])
            except (KeyError, ValueError):
                continue
            visible = (row.get("visible") or "").strip()
            if visible != "1":
                track[frame] = None
                continue
            try:
                track[frame] = (float(row["x"]), float(row["y"]))
            except (KeyError, ValueError):
                track[frame] = None
    return track


def shuttle_speed_series(
    track: Dict[int, Optional[Tuple[float, float]]],
    frame_diag: float,
    smoothing_span: int = DEFAULT_SMOOTHING_SPAN,
) -> Dict[int, Tuple[float, float]]:
    """Compute ``{frame: (smoothed_speed, valid)}`` from a shuttle track.

    Raw speed at frame f is the normalized distance between the shuttle at f and
    at f-1, valid only if BOTH are visible. The raw speed is then smoothed with a
    causal moving average over the last ``smoothing_span`` VALID speeds. A frame
    whose raw speed is invalid gets ``(0.0, 0.0)`` and does not contribute to the
    average. ``frame_diag`` normalizes pixels -> fraction of the frame diagonal.
    """
    if frame_diag <= 0:
        raise ValueError("frame_diag must be positive")
    out: Dict[int, Tuple[float, float]] = {}
    recent: deque = deque(maxlen=max(1, smoothing_span))
    for frame in sorted(track):
        cur = track.get(frame)
        prev = track.get(frame - 1)
        if cur is not None and prev is not None:
            raw = math.hypot(cur[0] - prev[0], cur[1] - prev[1]) / frame_diag
            recent.append(raw)
            smoothed = sum(recent) / len(recent)
            out[frame] = (smoothed, 1.0)
        else:
            # Unknown this frame: emit explicit unknown and break the smoothing
            # run so a gap does not average across it.
            recent.clear()
            out[frame] = (0.0, 0.0)
    return out


def annotate_pose_with_shuttle(
    pose_path: str,
    labels_csv: str,
    out_path: str,
    frame_width: int,
    frame_height: int,
    smoothing_span: int = DEFAULT_SMOOTHING_SPAN,
) -> Dict[str, Any]:
    """Add ``shuttle_speed`` / ``shuttle_speed_valid`` to each pose record.

    Reads the pose JSONL and the shuttle ``labels.csv``, joins by frame index
    (both are 0-based in the clip's own numbering), and writes an annotated
    JSONL to ``out_path``. Returns a small stats dict including the per-clip
    valid-speed fraction -- a low value flags a clip whose shuttle track mostly
    failed (so it can be excluded rather than trusted).
    """
    track = load_shuttle_track(labels_csv)
    frame_diag = math.hypot(frame_width, frame_height)
    speeds = shuttle_speed_series(track, frame_diag, smoothing_span)

    records: List[Dict[str, Any]] = []
    with open(pose_path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    matched = valid = 0
    for rec in records:
        speed, ok = speeds.get(rec.get("frame"), (0.0, 0.0))
        rec["shuttle_speed"] = round(float(speed), 6)
        rec["shuttle_speed_valid"] = int(ok)
        if rec.get("frame") in speeds:
            matched += 1
        valid += int(ok)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    n = len(records)
    return {
        "records": n,
        "matched_frames": matched,
        "valid_speed": valid,
        "valid_fraction": (valid / n) if n else 0.0,
        "smoothing_span": smoothing_span,
    }
