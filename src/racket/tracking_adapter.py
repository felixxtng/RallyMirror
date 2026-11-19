"""Boundary for the teammate's evolving YOLO/tracker output format.

Keep stable annotation IDs player_1/player_2. Change only this normalizer (or
convert the upstream output to this contract) when the final tracker schema is
agreed. An external track number is never assumed to equal an annotation ID.
Do not sort detections by position or infer athlete identity in this adapter.
"""
import copy


def normalize_tracking(raw):
    """Return explicit clip-local player associations in original pixels.

    Canonical input/output:
      {coordinate_system: original_pixels,
       frame_indexing: clip_local_zero_based,
       frames: [{frame: 0, players: {player_1: {track_id: 7, bbox: [...]}}}]}

    A future upstream adapter must account for frame offsets, resizing and
    person-to-player association before returning this canonical contract.
    Dimensions, duplicate IDs and frame bounds are checked by players.bind.
    """
    if not isinstance(raw, dict):
        raise ValueError("Tracking mapping must be a JSON object")
    if raw.get("coordinate_system") != "original_pixels" or raw.get("frame_indexing") != "clip_local_zero_based":
        raise ValueError("Mapping must declare original_pixels and clip_local_zero_based")
    if not isinstance(raw.get("frames"), list):
        raise ValueError("Tracking mapping must contain a frames list")
    for frame in raw["frames"]:
        if not isinstance(frame, dict) or "frame" not in frame or not isinstance(frame.get("players"), dict):
            raise ValueError("Each tracking frame needs a frame index and players object")
        if any(not isinstance(identity, dict) for identity in frame["players"].values()):
            raise ValueError("Each player association must be an object")
    return copy.deepcopy(raw)
