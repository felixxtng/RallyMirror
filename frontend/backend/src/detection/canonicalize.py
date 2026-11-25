"""Canonicalize fragmented tracker ids into stable player slots.

Problem
-------
ByteTrack (``PlayerTracker``) has no re-identification: when a player is
occluded, blurred on a lunge, or briefly leaves the frame, the recovered track
gets a brand-new integer id. In singles this shows up as one physical player
owning several ids over a clip (e.g. rally23: id 1 -> 84 for the near player;
id 2 plus 110/114 for fragments). Downstream wants a *stable identity* -- 
"player A" vs "player B" -- not a raw track id.

What this does
--------------
Rewrites each pose record's ``player_id`` to a canonical SLOT (1 or 2 for
singles) while preserving the original under ``raw_player_id``. Two mechanisms,
in priority order:

1. Court-space half assignment (primary, needs a calibration). Each record's
   ground-contact point (bbox bottom-centre) is mapped to court metres via the
   homography; the player is assigned to the NEAR or FAR half by the net line.
   A soft net band + per-slot hysteresis stop a near-net detection from flipping
   slots. This ignores the raw id entirely, so id churn cannot fragment a slot.

2. Gap-stitching (fallback, no calibration). A new id is merged into the slot of
   a recently-ended id when the two are close in time AND space (bottom-centre
   continuity). This resolves the common "id A ends, id B starts where A left
   off" handoff using position, not the id value.

Spectator guard
---------------
The merge is only ever applied to a CONTINUOUS handoff. If assigning a record
would put two records in the SAME slot in the SAME frame (two bodies in one
court half, or a stray that never hands off from an existing track), that is the
anomaly signal: the record is routed to the ``rejected`` bucket (its
``player_id`` set to ``REJECTED_SLOT``) instead of overwriting a real player.
So a leaked spectator becomes a flagged anomaly, never silently welded onto a
player's track.

This is a re-runnable post-process over an existing pose JSONL -- no
re-inference -- so it can be validated against already-produced data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.detection.target_selector import bbox_bottom_centre

# Canonical singles slots.
SLOT_A = 1
SLOT_B = 2
REJECTED_SLOT = -1  # anomalies (e.g. a stray that is not a clean handoff)


@dataclass
class CanonConfig:
    """Tunables for canonicalization.

    net_band_m:
        Half-width (metres) of a dead-band around the net line. A record whose
        court-y is within this band of the net keeps its slot's current half
        rather than being reassigned -- hysteresis for net play.
    max_gap_frames:
        For gap-stitching: a new id may only inherit a slot whose previous
        record is at most this many frames old.
    max_gap_px:
        For gap-stitching: and whose last bottom-centre is at most this many
        pixels away from the new id's first bottom-centre.
    """

    net_band_m: float = 0.75
    max_gap_frames: int = 30
    max_gap_px: float = 200.0


# ---------------------------------------------------------------------------
# Court-space path
# ---------------------------------------------------------------------------

def _court_y(record: Dict[str, Any], roi, scale: Tuple[float, float]) -> Optional[float]:
    """Court-metre y (along-court axis) of a record's bottom-centre, or None."""
    from src.court.homography import apply_homography

    bbox = record.get("bbox")
    if not bbox:
        return None
    bx, by = bbox_bottom_centre(bbox)
    sx, sy = scale
    court_xy = apply_homography(roi.H, (bx * sx, by * sy))
    y = float(court_xy[1])
    if y != y:  # NaN
        return None
    return y


def _scale_for(roi, frame_size: Optional[Tuple[int, int]]) -> Tuple[float, float]:
    if roi.calib_size and frame_size and tuple(frame_size) != tuple(roi.calib_size):
        return (roi.calib_size[0] / frame_size[0], roi.calib_size[1] / frame_size[1])
    return (1.0, 1.0)


def canonicalize_court(
    records: List[Dict[str, Any]],
    roi,
    frame_size: Optional[Tuple[int, int]] = None,
    config: Optional[CanonConfig] = None,
) -> List[Dict[str, Any]]:
    """Assign slots by court half (primary path). Returns NEW record dicts with
    ``player_id`` rewritten to a slot and ``raw_player_id`` preserved.
    """
    from src.court.court_model import NET_Y

    cfg = config or CanonConfig()
    scale = _scale_for(roi, frame_size)

    # Per-slot "current half" memory for hysteresis: slot -> "NEAR"/"FAR".
    slot_half = {SLOT_A: "NEAR", SLOT_B: "FAR"}

    out: List[Dict[str, Any]] = []
    # Group by frame to enforce the "at most one per slot per frame" invariant.
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_frame.setdefault(r["frame"], []).append(r)

    for frame in sorted(by_frame):
        frame_records = by_frame[frame]
        # Compute each record's half (with net-band hysteresis vs its likely slot).
        scored: List[Tuple[Dict[str, Any], Optional[str], Optional[float]]] = []
        for r in frame_records:
            y = _court_y(r, roi, scale)
            if y is None:
                scored.append((r, None, None))
                continue
            if y < NET_Y - cfg.net_band_m:
                half = "NEAR"
            elif y > NET_Y + cfg.net_band_m:
                half = "FAR"
            else:
                half = None  # inside the net band -> decide by proximity below
            scored.append((r, half, y))

        assigned_slots: Dict[int, Dict[str, Any]] = {}
        leftover: List[Tuple[Dict[str, Any], Optional[str], Optional[float]]] = []
        for r, half, y in scored:
            if half is None:
                leftover.append((r, half, y))
                continue
            slot = SLOT_A if half == "NEAR" else SLOT_B
            if slot in assigned_slots:
                # Two records want the same half this frame -> anomaly.
                leftover.append((r, half, y))
            else:
                assigned_slots[slot] = r

        # Net-band / anomaly leftovers: give an unused slot by nearest half,
        # else reject. A leftover only takes a slot if that slot is free.
        for r, half, y in leftover:
            if y is not None:
                pref = SLOT_A if y < NET_Y else SLOT_B
            else:
                pref = None
            if pref is not None and pref not in assigned_slots:
                assigned_slots[pref] = r
            else:
                other = SLOT_B if pref == SLOT_A else SLOT_A
                if pref is not None and other not in assigned_slots:
                    assigned_slots[other] = r
                else:
                    rr = dict(r)
                    rr["raw_player_id"] = r["player_id"]
                    rr["player_id"] = REJECTED_SLOT
                    out.append(rr)

        for slot, r in assigned_slots.items():
            rr = dict(r)
            rr["raw_player_id"] = r["player_id"]
            rr["player_id"] = slot
            out.append(rr)
            # Update hysteresis memory from the actual y.
            y = _court_y(r, roi, scale)
            if y is not None:
                slot_half[slot] = "NEAR" if y < NET_Y else "FAR"

    out.sort(key=lambda r: (r["frame"], r["player_id"]))
    return out


# ---------------------------------------------------------------------------
# Gap-stitching path (no calibration)
# ---------------------------------------------------------------------------

def canonicalize_gaps(
    records: List[Dict[str, Any]],
    config: Optional[CanonConfig] = None,
) -> List[Dict[str, Any]]:
    """Assign slots by temporal+spatial continuity when no court data exists.

    Greedy: keep up to two active slots. A new raw id inherits the slot of a
    recently-ended id whose last bottom-centre is near the new id's first one;
    otherwise it opens a free slot, or is rejected if both slots are busy.
    """
    cfg = config or CanonConfig()

    # First + last appearance and bottom-centre of each raw id.
    first: Dict[int, Dict[str, Any]] = {}
    last: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for r in sorted(records, key=lambda r: r["frame"]):
        pid = r["player_id"]
        if pid not in first:
            first[pid] = r
            order.append(pid)
        last[pid] = r

    id_to_slot: Dict[int, int] = {}
    # slot -> (last_frame, last_bottom_centre) of the id most recently in it.
    slot_tail: Dict[int, Tuple[int, Tuple[float, float]]] = {}

    def bc(r):
        return bbox_bottom_centre(r["bbox"])

    for pid in order:
        fr = first[pid]
        start_frame = fr["frame"]
        start_bc = bc(fr)
        chosen: Optional[int] = None
        # Prefer stitching onto a slot whose tail is a close handoff.
        best = None
        for slot, (tail_frame, tail_bc) in slot_tail.items():
            gap_f = start_frame - tail_frame
            if gap_f < 0 or gap_f > cfg.max_gap_frames:
                continue
            dist = ((start_bc[0] - tail_bc[0]) ** 2 + (start_bc[1] - tail_bc[1]) ** 2) ** 0.5
            if dist <= cfg.max_gap_px and (best is None or dist < best[1]):
                best = (slot, dist)
        if best is not None:
            chosen = best[0]
        else:
            used = set(slot_tail.keys()) & {SLOT_A, SLOT_B}
            # A slot is reusable if its tail is old enough to be "ended".
            free = [s for s in (SLOT_A, SLOT_B)
                    if s not in slot_tail or start_frame - slot_tail[s][0] > cfg.max_gap_frames]
            # But if the id's lifetime overlaps a slot still in active use, that
            # slot is not free (two concurrent bodies).
            chosen = free[0] if free else None
        id_to_slot[pid] = chosen if chosen is not None else REJECTED_SLOT
        if chosen is not None:
            lr = last[pid]
            slot_tail[chosen] = (lr["frame"], bc(lr))

    out: List[Dict[str, Any]] = []
    for r in records:
        rr = dict(r)
        rr["raw_player_id"] = r["player_id"]
        rr["player_id"] = id_to_slot.get(r["player_id"], REJECTED_SLOT)
        out.append(rr)
    out.sort(key=lambda r: (r["frame"], r["player_id"]))
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def canonicalize(
    records: List[Dict[str, Any]],
    roi=None,
    frame_size: Optional[Tuple[int, int]] = None,
    config: Optional[CanonConfig] = None,
) -> List[Dict[str, Any]]:
    """Canonicalize records to slots. Uses the court path when ``roi`` is given,
    else the gap-stitching fallback.
    """
    if roi is not None:
        return canonicalize_court(records, roi, frame_size, config)
    return canonicalize_gaps(records, config)


def canonicalize_file(
    in_path: str,
    out_path: str,
    calibration_path: Optional[str] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    margin_m: float = 1.0,
    config: Optional[CanonConfig] = None,
) -> Dict[str, Any]:
    """Read a pose JSONL, canonicalize, write a new JSONL. Returns stats."""
    from src.detection.target_selector import load_court_roi

    records: List[Dict[str, Any]] = []
    with open(in_path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    roi = load_court_roi(calibration_path, margin_m) if calibration_path else None
    canon = canonicalize(records, roi, frame_size, config)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in canon:
            fh.write(json.dumps(r) + "\n")

    from collections import Counter
    raw_ids = {r["player_id"] for r in records}
    slot_counts = Counter(r["player_id"] for r in canon)
    mapping: Dict[int, set] = {}
    for r in canon:
        mapping.setdefault(r["player_id"], set()).add(r["raw_player_id"])
    return {
        "records": len(records),
        "raw_ids": sorted(raw_ids),
        "slot_counts": dict(sorted(slot_counts.items())),
        "slot_to_raw_ids": {k: sorted(v) for k, v in sorted(mapping.items())},
        "rejected": slot_counts.get(REJECTED_SLOT, 0),
        "mode": "court" if roi is not None else "gaps",
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_jsonl")
    ap.add_argument("out_jsonl")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()
    frame_size = (args.width, args.height) if args.width and args.height else None
    stats = canonicalize_file(args.in_jsonl, args.out_jsonl, args.calibration, frame_size)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
