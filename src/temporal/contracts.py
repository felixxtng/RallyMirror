"""Upstream data contracts for the temporal/phase-recognition module.

The temporal module sits at the end of the pipeline:

    Detection            ->  Pose             ->  Temporal
    tracked-clip records     per-frame poses      keypoint sequence -> phase

Contract 1 -- tracked-clip record:
    {
      "frame": int,                      # 0-based frame index
      "player_id": int,                  # stable track id
      "bbox": [x1, y1, x2, y2],          # pixel coords, floats ok
      "confidence": float,               # optional
      ... any extra fields (e.g. "shuttle") are ALLOWED and IGNORED
    }
  Reference shape = the JSONL that src/detection/demo.py already writes.

Contract 2 -- per-frame pose:
  Pose estimation (MediaPipe Pose Landmarker, 33 keypoints) runs sequentially
  after player tracking, so each JSONL line carries BOTH the tracked-clip
  fields and a ``keypoints`` list:
    {
      "frame": int, "player_id": int, "bbox": [x1, y1, x2, y2], "confidence": ...,
      "keypoints": [{"name": str, "x": px, "y": px,
                     "visibility": float, "presence": float}, ... x33 ]
    }
  MediaPipe keypoints are ABSOLUTE PIXELS. ``load_pose_jsonl`` normalizes them
  to the record's bbox (coords -> ~[0, 1]) and emits the INTERNAL pose shape the
  windower/classifier consume:
    {"frame": int, "player_id": int, "keypoints": [[x_norm, y_norm], ...]}

  The validators also still accept the internal list form ([x, y] / [x, y,
  score]) so stubs and unit tests keep working.

These validators intentionally raise on malformed *required* fields but never
complain about unknown extra fields.
"""

from __future__ import annotations

import math
import random
import json
from typing import Any, Dict, Iterable, Iterator, List

# The pose estimator (src/pose/estimator.py) emits a REDUCED, name-ordered
# subset of MediaPipe's 33 landmarks: the nose plus the main body joints and
# feet (eyes, ears, mouth and finger landmarks are dropped upstream). This list
# is the single source of truth for the surviving keypoints *in the order the
# estimator outputs them*. Everything downstream derives from it, so if the
# estimator's subset changes again, only this list needs updating here.
KEYPOINT_NAMES = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # derived, currently 17

# Wrist positions WITHIN the reduced layout (used as a swing signal by the stub
# and the rule-based classifier). Derived from the names so they never drift
# from the estimator's output order.
LEFT_WRIST = KEYPOINT_NAMES.index("left_wrist")    # -> 5 in the reduced layout
RIGHT_WRIST = KEYPOINT_NAMES.index("right_wrist")  # -> 6 in the reduced layout

class ContractError(ValueError):
    """Raised when an upstream record violates a required part of the contract."""


# ---------------------------------------------------------------------------
# Contract 1: tracked-clip record
# ---------------------------------------------------------------------------

def validate_track_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single tracked-clip record. Extra fields (e.g. ``shuttle``)
    are permitted and preserved. Returns the record unchanged on success.
    """
    if not isinstance(record, dict):
        raise ContractError(f"track record must be a dict, got {type(record).__name__}")

    for field in ("frame", "player_id", "bbox"):
        if field not in record:
            raise ContractError(f"track record missing required field {field!r}")

    if not isinstance(record["frame"], int) or isinstance(record["frame"], bool):
        raise ContractError("track record 'frame' must be an int")
    if not isinstance(record["player_id"], int) or isinstance(record["player_id"], bool):
        raise ContractError("track record 'player_id' must be an int")

    bbox = record["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ContractError("track record 'bbox' must be [x1, y1, x2, y2]")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
        raise ContractError("track record 'bbox' entries must be numbers")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ContractError(f"track record 'bbox' is degenerate: {bbox}")

    return record


# ---------------------------------------------------------------------------
# Contract 2: per-frame pose record
# ---------------------------------------------------------------------------

def validate_pose_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single per-frame pose record. Extra fields are permitted.
    Returns the record unchanged on success.
    """
    if not isinstance(record, dict):
        raise ContractError(f"pose record must be a dict, got {type(record).__name__}")

    for field in ("frame", "player_id", "keypoints"):
        if field not in record:
            raise ContractError(f"pose record missing required field {field!r}")

    if not isinstance(record["frame"], int) or isinstance(record["frame"], bool):
        raise ContractError("pose record 'frame' must be an int")
    if not isinstance(record["player_id"], int) or isinstance(record["player_id"], bool):
        raise ContractError("pose record 'player_id' must be an int")

    kps = record["keypoints"]
    if not isinstance(kps, (list, tuple)) or len(kps) != NUM_KEYPOINTS:
        raise ContractError(
            f"pose record 'keypoints' must have {NUM_KEYPOINTS} entries, "
            f"got {len(kps) if isinstance(kps, (list, tuple)) else type(kps).__name__}"
        )
    for i, kp in enumerate(kps):
        x, y = _keypoint_coords(kp, i)
        # Normalized to bbox: expect roughly [0, 1]. Allow a small margin for
        # points that fall just outside the box.
        if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
            raise ContractError(
                f"keypoint {i} = ({x}, {y}) not in normalized range; "
                "expected bbox-relative coords in ~[0, 1]. If this is raw "
                "upstream data in pixels, load it via load_pose_jsonl which "
                "normalizes to the bbox."
            )

    return record


def _keypoint_coords(kp: Any, index: int) -> tuple[float, float]:
    """Extract (x, y) from a keypoint in either supported form.

    Two accepted shapes:
    - list/tuple ``[x, y]`` or ``[x, y, score]`` (internal / stub format)
    - dict ``{"x": ..., "y": ..., ...}`` (MediaPipe Pose Landmarker format,
      which also carries ``name``/``visibility``/``presence`` -- all ignored)
    """
    if isinstance(kp, dict):
        if "x" not in kp or "y" not in kp:
            raise ContractError(f"keypoint {index} dict missing 'x'/'y'")
        x, y = kp["x"], kp["y"]
    elif isinstance(kp, (list, tuple)) and len(kp) in (2, 3):
        x, y = kp[0], kp[1]
    else:
        raise ContractError(
            f"keypoint {index} must be [x, y], [x, y, score], or "
            "{{'x':..., 'y':...}}"
        )
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
        raise ContractError(f"keypoint {index} coords must be numbers")
    return float(x), float(y)


# Court-space normalization. Person 3's mapper emits ``court_xy`` in METRES on a
# BWF court (x across 0..6.10, y along 0..13.40). To make court position
# commensurate with the bbox-normalized keypoints (both ~[0, 1]) before they are
# concatenated into one feature vector, court metres are divided by the court
# dimensions. A record with no ``court_xy`` (no trustworthy foot/fit that frame)
# falls back to court centre (0.5, 0.5) -- a neutral "position unknown" value
# that, unlike (0, 0), does not read as a specific court corner.
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
COURT_CENTRE_NORM = (0.5, 0.5)

# Shuttle rally-state feature. The signal that separates dead-time (idle) from
# live play is shuttle SPEED, not presence: during idle the shuttle lies on the
# floor or is carried (slow/still) while visible, so presence alone does not
# distinguish idle from ready. ``src.shuttle``-style mapping writes a smoothed
# per-frame ``shuttle_speed`` (already normalized, resolution-independent) and a
# ``shuttle_speed_valid`` flag (1.0 if speed was computable from two consecutive
# visible detections, else 0.0). The validity flag is essential: a missing
# shuttle (21% of frames, mid-rally occlusion OR off-screen between points) must
# read as "unknown", NOT as speed 0 (which would look like idle). Encoding both
# lets the model learn "unknown" rather than be told a false zero.
SHUTTLE_FEATURE_LEN = 2  # (smoothed speed, validity flag)


def pose_record_to_vector(record: Dict[str, Any], use_court: bool = False,
                          use_shuttle: bool = False) -> List[float]:
    """Flatten a validated pose record into the per-frame BASE feature vector.

    Layout (each optional block appended in this fixed order):
        [kp coords ...]                      length 2 * NUM_KEYPOINTS
        [court_x_norm, court_y_norm]         if use_court   (+2)
        [shuttle_speed, shuttle_valid]       if use_shuttle (+2)

    Keypoints are expected bbox-normalized (use ``load_pose_jsonl`` first).
    Court is appended BEFORE shuttle deliberately: the velocity transform
    (``features.add_velocity``) differences the keypoint+court prefix (court
    motion is meaningful), but must NOT difference the shuttle channels --
    shuttle_speed is already a velocity-like quantity and the validity flag is
    categorical. ``features.transform_window`` relies on this ordering to split
    the shuttle tail off before differencing.

    Training and inference MUST agree on both flags (the classifier persists
    them alongside ``use_velocity``).
    """
    kps = record["keypoints"]
    vector: List[float] = []
    for i, kp in enumerate(kps):
        x, y = _keypoint_coords(kp, i)
        vector.append(x)
        vector.append(y)
    if use_court:
        cx, cy = court_xy_norm(record)
        vector.append(cx)
        vector.append(cy)
    if use_shuttle:
        speed, valid = shuttle_features(record)
        vector.append(speed)
        vector.append(valid)
    return vector


def court_xy_norm(record: Dict[str, Any]) -> tuple[float, float]:
    """Return the record's court position normalized to ~[0, 1].

    Reads ``court_xy`` (metres, as added by ``src.court.mapper``). Missing or
    malformed -> court centre ``COURT_CENTRE_NORM`` ("position unknown").
    """
    court = record.get("court_xy")
    if (not isinstance(court, (list, tuple)) or len(court) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in court)):
        return COURT_CENTRE_NORM
    return (float(court[0]) / COURT_WIDTH_M, float(court[1]) / COURT_LENGTH_M)


def shuttle_features(record: Dict[str, Any]) -> tuple[float, float]:
    """Return ``(shuttle_speed, shuttle_speed_valid)`` for a record.

    Reads ``shuttle_speed`` (smoothed, normalized) and ``shuttle_speed_valid``
    (0/1) as written by the shuttle mapping stage. When either is missing or
    malformed the pair is ``(0.0, 0.0)`` -- speed 0 with validity 0, i.e. an
    explicit "unknown" the model can learn, never a false "still shuttle".
    """
    speed = record.get("shuttle_speed")
    valid = record.get("shuttle_speed_valid")
    if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
            or not isinstance(valid, (int, float)) or isinstance(valid, bool)):
        return (0.0, 0.0)
    return (float(speed), 1.0 if valid else 0.0)


# ---------------------------------------------------------------------------
# Stub generators
# ---------------------------------------------------------------------------

def stub_track_stream(
    num_frames: int = 120,
    player_id: int = 1,
    width: int = 1280,
    height: int = 720,
    include_shuttle: bool = True,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Yield synthetic tracked-clip records conforming to Contract 1.

    Includes a ``shuttle`` field by default to exercise the tolerant-consumer
    path (we must ignore it cleanly).
    """
    rng = random.Random(seed)
    cx, cy = width / 2.0, height / 2.0
    for frame in range(num_frames):
        # Player drifts around the court.
        cx += rng.uniform(-6, 6)
        cy += rng.uniform(-3, 3)
        cx = min(max(cx, 150), width - 150)
        cy = min(max(cy, 150), height - 150)
        w, h = 90.0, 200.0
        record: Dict[str, Any] = {
            "frame": frame,
            "player_id": player_id,
            "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            "confidence": round(rng.uniform(0.7, 0.95), 3),
        }
        if include_shuttle:
            # Extra field a tolerant consumer must ignore.
            record["shuttle"] = [rng.uniform(0, width), rng.uniform(0, height)]
        yield record


def stub_pose_stream(
    num_frames: int = 120,
    player_id: int = 1,
    seed: int = 0,
    segment_len: int = 30,
) -> Iterator[Dict[str, Any]]:
    """Yield synthetic per-frame pose records conforming to Contract 2.

    The stream is organized into repeating behavioural *segments* so that a
    downstream classifier exercises every phase rather than a single flat label:

        segment 0: idle    -- near-static pose (tiny jitter only)
        segment 1: movement-- whole body translates, wrists steady (no swing)
        segment 2: stroke  -- sharp wrist swing, body otherwise steady

    Segments repeat every ``3 * segment_len`` frames. This makes the stub a
    better behavioural stand-in for the pose module's eventual real poses: it has
    distinct, separable motion signatures. Keypoints are bbox-normalized and the
    stream remains deterministic under ``seed``.
    """
    rng = random.Random(seed + 1)
    # Accumulated body drift during movement segments (persists across frames).
    body_drift = 0.0
    for frame in range(num_frames):
        segment = (frame // segment_len) % 3
        pos_in_segment = frame % segment_len

        # Per-segment motion parameters.
        if segment == 0:          # idle
            swing = 0.0
            body_drift += 0.0
        elif segment == 1:        # movement
            swing = 0.0
            body_drift += 0.006   # steady translation across the segment
        else:                      # stroke
            # Sharp wrist swing that grows across the segment then is read by
            # the windowed displacement cue.
            swing = 0.03 * pos_in_segment

        keypoints: List[List[float]] = []
        for k in range(NUM_KEYPOINTS):
            base_x = 0.5 + 0.02 * math.cos(k) + body_drift
            base_y = k / float(NUM_KEYPOINTS)
            # Wrists carry the swing signal (positions within the reduced layout).
            kp_swing = swing if k in (LEFT_WRIST, RIGHT_WRIST) else 0.0
            x = base_x + kp_swing + rng.uniform(-0.004, 0.004)
            y = base_y + rng.uniform(-0.004, 0.004)
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            keypoints.append([round(x, 4), round(y, 4), round(rng.uniform(0.6, 0.99), 3)])
        yield {"frame": frame, "player_id": player_id, "keypoints": keypoints}


def validate_stream(
    records: Iterable[Dict[str, Any]],
    kind: str,
) -> List[Dict[str, Any]]:
    """Validate every record in a stream. ``kind`` is 'track' or 'pose'.
    Returns the validated records as a list.
    """
    if kind == "track":
        validator = validate_track_record
    elif kind == "pose":
        validator = validate_pose_record
    else:
        raise ValueError(f"unknown stream kind {kind!r}; expected 'track' or 'pose'")
    return [validator(r) for r in records]

# ---------------------------------------------------------------------------
# Loading real upstream JSONL
# ---------------------------------------------------------------------------

def load_track_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load tracked-clip JSONL into a list of validated records.

    One JSON object per line (matching ``src/detection/demo.py``). Tolerant:
    blank lines are ignored, and any line that is not valid JSON or that fails
    the Contract 1 check is skipped with a warning rather than aborting the
    whole load -- a demo over real, possibly-messy data should keep going.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[contracts] skipping malformed JSON at {path}:{line_no}: {exc}")
                continue
            try:
                records.append(validate_track_record(record))
            except ContractError as exc:
                print(f"[contracts] skipping invalid track record at {path}:{line_no}: {exc}")
                continue
    return records


def index_tracks_by_frame(records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index validated track records by frame -> single record.

    Back-compat single-target view: if multiple records share a frame (e.g. two
    players), the last one wins. Retained for the original singles demo and any
    consumer that only wants one box per frame. For multi-player data (detection
    now emits BOTH players per frame) use :func:`group_tracks_by_frame`, which
    keeps every player.
    """
    by_frame: Dict[int, Dict[str, Any]] = {}
    for record in records:
        by_frame[record["frame"]] = record
    return by_frame


def group_tracks_by_frame(
    records: Iterable[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Index validated track records by frame -> list of records.

    Multi-player view: every record for a frame is preserved, so overlays and
    per-player temporal logic see both players. Ordering within a frame follows
    input order (which is the detection pipeline's ranked order). This is the
    indexer to use now that selection returns up to N targets per frame.
    """
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_frame.setdefault(record["frame"], []).append(record)
    return by_frame


def load_tracks_by_frame(path: str) -> Dict[int, Dict[str, Any]]:
    """Convenience: load a track JSONL and index it by frame (single-record).

    Uses :func:`index_tracks_by_frame` (last-writer-wins) for back-compat. For
    the multi-player view use :func:`load_tracks_grouped_by_frame`.
    """
    return index_tracks_by_frame(load_track_jsonl(path))


def load_tracks_grouped_by_frame(path: str) -> Dict[int, List[Dict[str, Any]]]:
    """Convenience: load a track JSONL and group it by frame (all players)."""
    return group_tracks_by_frame(load_track_jsonl(path))

def normalize_keypoints_to_bbox(
    keypoints: Iterable[Any],
    bbox: List[float],
) -> List[List[float]]:
    """Convert pixel-space keypoints into bbox-relative [x, y] in ~[0, 1].

    ``x_norm = (x - x1) / (x2 - x1)`` and likewise for y. Accepts keypoints in
    either the dict (MediaPipe) or list form. Returns internal-format
    ``[x_norm, y_norm]`` pairs. Points outside the box fall outside [0, 1]; the
    pose validator tolerates a small margin.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ContractError(f"cannot normalize against degenerate bbox: {bbox}")
    normalized: List[List[float]] = []
    for i, kp in enumerate(keypoints):
        x, y = _keypoint_coords(kp, i)
        normalized.append([(x - x1) / width, (y - y1) / height])
    return normalized


def select_keypoints_by_name(keypoints: List[Any]) -> List[Any]:
    """Reorder/select named keypoints to match ``KEYPOINT_NAMES``.

    When each keypoint is a dict carrying ``name`` (the MediaPipe output), this
    returns exactly the keypoints in ``KEYPOINT_NAMES`` order, so the feature
    vector is always in the canonical layout regardless of the producer's
    ordering, and any extra named landmarks are dropped. Raises if a required
    name is missing.

    If the keypoints are not named dicts (e.g. the plain ``[x, y]`` list form
    used by stubs/tests), they are returned unchanged for positional handling.
    """
    if not keypoints or not isinstance(keypoints[0], dict) or "name" not in keypoints[0]:
        return keypoints
    by_name = {kp["name"]: kp for kp in keypoints if isinstance(kp, dict) and "name" in kp}
    missing = [n for n in KEYPOINT_NAMES if n not in by_name]
    if missing:
        raise ContractError(f"keypoints missing required names: {missing}")
    return [by_name[n] for n in KEYPOINT_NAMES]


def load_pose_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load the merged track+pose JSONL (``pose_extract`` output).

    Each line carries both a tracked-clip record (``bbox``) and a ``keypoints``
    list. MediaPipe keypoints are in absolute pixels; this loader:
    1. selects/reorders keypoints by name to the canonical ``KEYPOINT_NAMES``
       layout (when names are present), then
    2. normalizes them to the record's bbox so the output conforms to Contract
       2 (bbox-relative coords) and flows through the existing
       windower/classifier unchanged.

    Returns internal-format pose records::

        {"frame": int, "player_id": int, "keypoints": [[x_norm, y_norm], ...]}

    Tolerant: blank lines, malformed JSON, records missing a valid bbox or
    ``keypoints``, missing a required named keypoint, and anything that fails
    validation are skipped with a warning rather than aborting the load.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[contracts] skipping malformed JSON at {path}:{line_no}: {exc}")
                continue
            try:
                # Reuse the track validator to guarantee a usable bbox + ids.
                validate_track_record(raw)
                if "keypoints" not in raw:
                    raise ContractError("merged record missing 'keypoints'")
                selected = select_keypoints_by_name(raw["keypoints"])
                norm_kps = normalize_keypoints_to_bbox(selected, raw["bbox"])
                pose_record = {
                    "frame": raw["frame"],
                    "player_id": raw["player_id"],
                    "keypoints": norm_kps,
                }
                # Preserve court position (metres) when present, so the court
                # feature can be built downstream. Absent -> handled as "unknown"
                # (court centre) by court_xy_norm; never fabricated here.
                if "court_xy" in raw:
                    pose_record["court_xy"] = raw["court_xy"]
                # Likewise preserve the shuttle rally-state fields when present.
                if "shuttle_speed" in raw:
                    pose_record["shuttle_speed"] = raw["shuttle_speed"]
                if "shuttle_speed_valid" in raw:
                    pose_record["shuttle_speed_valid"] = raw["shuttle_speed_valid"]
                validate_pose_record(pose_record)
            except ContractError as exc:
                print(f"[contracts] skipping invalid pose record at {path}:{line_no}: {exc}")
                continue
            records.append(pose_record)
    return records