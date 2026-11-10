"""Phase-recognition demo (Person 4).

Drives the temporal :class:`PhasePipeline` end-to-end and produces:

1. A per-frame phase JSONL (always, no heavy deps) -- the inspectable result.
2. An overlay .mp4 (only when OpenCV is available) drawing each player's bbox
   and current phase label on top of the source video.

Scope / ownership notes:
- This is a SEPARATE demo from ``src/detection/demo.py`` (Person 1's player +
  shuttle tracking demo). We do not edit that file. This keeps workstreams from
  colliding; the two demos can later be merged by the team if desired.
- Upstream data (Person 1 tracks, Person 2 poses) is not ready, so by default
  this demo runs on STUB streams from ``contracts``. When the real upstreams
  land, swap the stub streams for loaders over their JSONL -- the pipeline code
  does not change.

Integration seams (clearly marked below):
- POSE SOURCE seam: replace ``stub_pose_stream`` with Person 2's pose output.
- TRACK SOURCE seam: pass ``source_jsonl`` to load Person 1's track JSONL
  (used only to place the overlay box; phases come from poses). Falls back to
  ``stub_track_stream`` when no JSONL is given.
- SHUTTLE seam: Person 1's records may carry a 'shuttle' field; a future
  trajectory overlay can read it here without touching phase logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.temporal.contracts import (
    group_tracks_by_frame,
    load_pose_jsonl,
    load_tracks_grouped_by_frame,
    stub_pose_stream,
    stub_track_stream,
)
from src.temporal.pipeline import PhasePipeline

# Overlay colours per phase (BGR). Only used when OpenCV is present.
_PHASE_COLORS = {
    "ready": (0, 200, 255),
    "movement": (255, 160, 0),
    "stroke": (0, 255, 0),
    "recovery": (255, 0, 200),
    "idle": (180, 180, 180),
}


def run_phase_demo(
    output_jsonl: str,
    output_video: Optional[str] = None,
    source_video: Optional[str] = None,
    source_jsonl: Optional[str] = None,
    pose_jsonl: Optional[str] = None,
    num_frames: int = 150,
    window_size: int = 30,
    smoothing: int = 5,
    player_id: int = 1,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the phase pipeline over a pose stream and write results.

    Parameters
    ----------
    source_jsonl:
        Optional path to Person 1's tracked-clip JSONL. When given, overlay
        boxes are read from it; otherwise a deterministic stub track stream is
        used. Phases always come from the pose stream, not from tracks.
    pose_jsonl:
        Optional path to the merged track+pose JSONL (MediaPipe keypoints in
        pixels). When given, it is loaded and bbox-normalized to drive phase
        recognition; otherwise the deterministic stub pose stream is used. When
        supplied, the number of frames rendered follows the loaded data.

    Returns a small summary dict (frames processed, phases emitted, counts) so
    the run is verifiable without opening the files.
    """
    pipe = PhasePipeline(window_size=window_size, smoothing=smoothing)

    # --- TRACK SOURCE seam ------------------------------------------------
    # Used only to position overlay boxes. Phases are derived from poses.
    # If a real track JSONL is given, load and index it by
    # frame; otherwise fall back to the deterministic stub so the demo still
    # runs with no upstream data.
    # Grouped by frame -> list of players, since detection now emits both
    # players per frame. The stub stream is single-player, so its groups are
    # one-element lists (identical overlay to before in the stub case).
    if source_jsonl is not None:
        tracks_by_frame = load_tracks_grouped_by_frame(source_jsonl)
    else:
        tracks_by_frame = group_tracks_by_frame(
            stub_track_stream(num_frames=num_frames, player_id=player_id, seed=seed)
        )

    # --- POSE SOURCE seam -------------------------------------------------
    # If a merged track+pose JSONL is given,
    # load and bbox-normalize it; otherwise fall back to the deterministic stub.
    if pose_jsonl is not None:
        pose_stream = load_pose_jsonl(pose_jsonl)
        # Real data drives the render length rather than the stub default.
        if pose_stream:
            num_frames = max(rec["frame"] for rec in pose_stream) + 1
    else:
        pose_stream = list(
            stub_pose_stream(num_frames=num_frames, player_id=player_id, seed=seed)
        )

    phase_rows: List[Dict[str, object]] = []
    phase_by_frame: Dict[int, str] = {}
    for pose_rec in pose_stream:
        result = pipe.process(pose_rec)
        if result is None:
            continue  # not enough temporal context yet
        frame, pid, phase = result
        row = {"frame": frame, "player_id": pid, "phase": phase}
        phase_rows.append(row)
        phase_by_frame[frame] = phase

    # Always write the JSONL result.
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in phase_rows:
            fh.write(json.dumps(row) + "\n")

    counts: Dict[str, int] = {}
    for row in phase_rows:
        counts[row["phase"]] = counts.get(row["phase"], 0) + 1

    summary: Dict[str, object] = {
        "frames_processed": len(pose_stream),
        "phases_emitted": len(phase_rows),
        "phase_counts": counts,
        "jsonl": str(out_path),
        "video": None,
    }

    # Optionally render the overlay video if OpenCV is available.
    if output_video is not None:
        rendered = _maybe_render_video(
            output_video=output_video,
            source_video=source_video,
            tracks_by_frame=tracks_by_frame,
            phase_by_frame=phase_by_frame,
            num_frames=num_frames,
        )
        summary["video"] = rendered

    return summary


def _maybe_render_video(
    output_video: str,
    source_video: Optional[str],
    tracks_by_frame: Dict[int, List[dict]],
    phase_by_frame: Dict[int, str],
    num_frames: int,
) -> Optional[str]:
    """Render an overlay video if OpenCV is installed; otherwise skip quietly.

    When ``source_video`` is given, overlays are drawn on its frames. Otherwise
    a blank canvas is used so the demo runs even without the (not-redistributed)
    footage.
    """
    try:
        import cv2  # noqa: WPS433 (optional dependency)
        import numpy as np
    except ImportError:
        print("[phase-demo] OpenCV not available; skipping video render. "
              "JSONL result was still written.")
        return None

    width, height, fps = 1280, 720, 30
    reader = None
    if source_video and Path(source_video).exists():
        reader = cv2.VideoCapture(source_video)
        if reader.isOpened():
            width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            fps = reader.get(cv2.CAP_PROP_FPS) or fps

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame_idx in range(num_frames):
        if reader is not None:
            ok, frame = reader.read()
            if not ok:
                break
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        tracks = tracks_by_frame.get(frame_idx, [])
        # --- PER-PLAYER PHASE seam: phase_by_frame is currently keyed by frame
        # only (single-player phase stream). With both players tracked, phase
        # recognition will key by (frame, player_id); until then every box in a
        # frame shares the frame's phase label. ---
        phase = phase_by_frame.get(frame_idx)
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track["bbox"])
            color = _PHASE_COLORS.get(phase, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = phase if phase is not None else "warming up"
            cv2.putText(
                frame, f"P{track['player_id']}: {label}", (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            # --- SHUTTLE seam: a future trajectory overlay could draw
            # track.get("shuttle") here without touching phase logic. ---

        writer.write(frame)

    if reader is not None:
        reader.release()
    writer.release()
    print(f"[phase-demo] Wrote overlay video: {output_video}")
    return output_video


if __name__ == "__main__":
    result = run_phase_demo(
        output_jsonl="outputs/phase_demo.jsonl",
        output_video="outputs/phase_demo.mp4",
        source_video="data/sample/test_footage_1.mp4",
        source_jsonl="outputs/tracking.jsonl",
        num_frames=278
    )
    print("[phase-demo] summary:")
    print(json.dumps(result, indent=2))
