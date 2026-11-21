"""Combined demonstration pipeline: raw video -> one overlay video.

Takes a single raw rally video and renders ONE output video that overlays every
current module's result at once:

    * court lines          (src.court: predict -> homography -> overlay_court)
    * player bounding boxes (src.detection + src.pose)
    * pose keypoints        (src.pose, 18 kp incl. racket_center)
    * estimated phase       (src.temporal TCN, per player per frame)
    * (optional) top-down minimap with each player's court position

Unlike ``src.demo_pipeline`` (which produces TRAINING data for many clips), this
is a single-video DEMO: it caps frames with ``--max-frames`` and composes one
annotated mp4.

Stage order (each reuses the training pipeline's functions so inference matches
how the data the model trained on was produced):

    1. COURT      predict a calibration (LOO-gated; a shaky court is rejected
                  and the overlay says so instead of drawing wrong lines)
    2. POSE       detection + MediaPipe pose + racket_center keypoint -> pose.jsonl
    3. CANON      stable player slots (slot 1 = NEAR half, slot 2 = FAR half)
    4. COURT-MAP  add court_xy (metres) per record  [feeds the TCN court feature]
    5. SHUTTLE    run the shuttle tracker, bridge its jsonl to the speed signal,
                  add shuttle_speed per record       [feeds the TCN shuttle feature]
    6. PHASE      run the TCN per player; collect phase by (frame, player_id)
    7. RENDER     one pass over the raw frames drawing everything

Modes (``--mode``): ``near`` labels only slot 1, ``far`` only slot 2, ``both``
labels both players.

Shuttle normalization note
--------------------------
The TCN learned ``shuttle_speed`` normalized by a FIXED detection-space diagonal
(960x544 -> ~1103 px), the value ``src.demo_pipeline`` used when building the
training data. Inference MUST reuse that exact diagonal or the shuttle channel
is off-scale, so :data:`SHUTTLE_NORM_WH` is fixed here, not derived from the
input video's resolution. The model therefore assumes ~1280x720 footage.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.court.net import load_checkpoint
from src.court.calibration import load_calibration
from src.court.draw import overlay_court, MiniMap, paste_inset
from src.detection.canonicalize import SLOT_A, SLOT_B, REJECTED_SLOT
from src.detection.shuttle import DEFAULT_SMOOTHING_SPAN
from src.pose.estimator import CONNECTIONS
from src.temporal.classifier import TrainableClassifier
from src.temporal.contracts import load_pose_jsonl
from src.temporal.pipeline import PhasePipeline

# Reuse the training pipeline's stage functions verbatim so inference-time data
# is produced identically to the data the model trained on.
from src.demo_pipeline import (
    get_court_calibration, get_pose_data, canonicalize_pose, map_court_coords,
    map_shuttle_speed, calib_path_for, DEFAULT_LOO_GATE_M,
)

# --- fixed config -----------------------------------------------------------
PHASE_MODEL = Path("models/phase_tcn.pt")
RACKET_WEIGHTS = Path("models/racket_center_v006.pt")
SHUTTLE_WEIGHTS = Path("models/shuttle_full_b8/best.pt")
OUTPUT_DIR = Path("outputs/combined")

# The diagonal the shuttle speed feature was trained with (see module docstring).
SHUTTLE_NORM_WH = (960, 544)

# Per-phase overlay colours (BGR), matching src.temporal.demo_phase.
PHASE_COLORS = {
    "ready": (0, 200, 255),
    "movement": (255, 160, 0),
    "stroke": (0, 255, 0),
    "recovery": (255, 0, 200),
    "idle": (180, 180, 180),
}
WARMING_UP = "warming up"           # label before a track fills the TCN window
SLOT_NAMES = {SLOT_A: "NEAR", SLOT_B: "FAR"}


# ---------------------------------------------------------------------------
# Shuttle bridge: tracker jsonl -> labels.csv the speed mapper consumes
# ---------------------------------------------------------------------------

def run_shuttle_tracker(video_path, out_dir, weights, device, threshold=0.5):
    """Run the offline shuttle tracker; return the path to its tracking JSONL.

    ``run_offline_demo`` refuses a non-empty output dir, so ``out_dir`` must be
    fresh. Coordinates are original-pixel (per its metadata).
    """
    from src.shuttle.offline_demo import run_offline_demo

    out_dir = Path(out_dir)
    run_offline_demo(video_path, str(weights), str(out_dir),
                     threshold=threshold, device=device)
    return out_dir / "shuttle_tracking.jsonl"


def shuttle_jsonl_to_labels_csv(tracking_jsonl, out_csv):
    """Convert a shuttle ``shuttle_tracking.jsonl`` to the ``labels.csv`` shape
    (``frame,x,y,visible``) that :func:`annotate_pose_with_shuttle` reads.

    The jsonl is original-pixel; we pass x/y through UNCHANGED (the training
    labels.csv was also original-pixel) and map ``status == "detected"`` to
    ``visible = 1``. Normalization by the fixed diagonal happens later in the
    speed mapper, exactly as it did for training.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(tracking_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            detected = rec.get("status") == "detected" and rec.get("x") is not None
            rows.append((int(rec["frame"]),
                         rec.get("x") if detected else "",
                         rec.get("y") if detected else "",
                         1 if detected else 0))
    rows.sort(key=lambda r: r[0])
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        fh.write("frame,x,y,visible\n")
        for frame, x, y, vis in rows:
            fh.write(f"{frame},{x},{y},{vis}\n")
    return out_csv


# ---------------------------------------------------------------------------
# Phase inference
# ---------------------------------------------------------------------------

def run_phase_inference(pose_path, model_path, window_size, smoothing, device):
    """Run the TCN over the pose JSONL; return ``{(frame, player_id): phase}``.

    Poses are loaded via ``load_pose_jsonl`` (bbox-normalized, racket keypoint
    kept, court/shuttle fields preserved) so the feature vector matches training.
    The pipeline bakes in the court+shuttle channels the loaded classifier
    expects. Rejected canonicalization slots (REJECTED_SLOT) are skipped -- they
    are anomalies, not a real player.
    """
    classifier = TrainableClassifier.load(str(model_path), device=device)
    pipe = PhasePipeline(window_size=window_size, classifier=classifier,
                         smoothing=smoothing)
    phases = {}
    for rec in load_pose_jsonl(str(pose_path)):
        if rec["player_id"] == REJECTED_SLOT:
            continue
        result = pipe.process(rec)
        if result is not None:
            frame, pid, phase = result
            phases[(frame, pid)] = phase
    return phases


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_player(frame, record, phase, threshold):
    """Draw one player's phase-coloured box, skeleton, and label (in place)."""
    pid = record["player_id"]
    color = PHASE_COLORS.get(phase, (255, 255, 255))
    x1, y1, x2, y2 = (int(round(v)) for v in record["bbox"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    h, w = frame.shape[:2]
    by_name = {p["name"]: p for p in (record.get("keypoints") or [])}

    def ok(name):
        p = by_name.get(name)
        return (p is not None and p["visibility"] >= threshold
                and p["presence"] >= threshold and 0 <= p["x"] < w and 0 <= p["y"] < h)

    def xy(name):
        p = by_name[name]
        return int(round(p["x"])), int(round(p["y"]))

    for a, b in CONNECTIONS:
        if ok(a) and ok(b):
            cv2.line(frame, xy(a), xy(b), (0, 255, 255), 2, cv2.LINE_AA)
    for name in by_name:
        if ok(name):
            cv2.circle(frame, xy(name), 3, (255, 0, 255), -1, cv2.LINE_AA)

    label = f"{SLOT_NAMES.get(pid, f'P{pid}')}: {phase or WARMING_UP}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _draw_shuttle(frame, trail):
    """Draw the shuttle trail + current point, matching the shuttle demo style.

    ``trail`` is a deque of recent ``(x, y) or None`` points (oldest -> newest),
    in original pixels -- the same space as the rendered frame, so no transform
    is needed. Green line through consecutive visible points, ring on the last.
    """
    pts = list(trail)
    for a, b in zip(pts, pts[1:]):
        if a is not None and b is not None:
            cv2.line(frame, a, b, (0, 255, 0), 2, cv2.LINE_AA)
    if pts and pts[-1] is not None:
        cv2.circle(frame, pts[-1], 10, (0, 255, 0), 2, cv2.LINE_AA)


def render_combined(video_path, output_path, pose_path, phases, calibration,
                    court_note, mode, max_frames, threshold, show_minimap,
                    shuttle_track=None):
    """One render pass over the raw frames drawing every signal.

    ``phases`` maps (frame, player_id) -> phase. ``calibration`` is the loaded
    court calib dict (has ``H``) or None. ``court_note`` is an on-screen status
    string drawn top-left (e.g. why court lines are absent). ``mode`` filters
    which slots are labelled (near/far/both). ``shuttle_track`` maps frame ->
    ``(x, y) or None`` (original pixels); when given, the shuttle trail is drawn
    onto the SAME frames, so the whole demo is one video.
    """
    # Index pose records by frame so the render can look players up cheaply.
    by_frame = {}
    for rec in _load_raw_pose(pose_path):
        by_frame.setdefault(rec["frame"], []).append(rec)

    wanted_slots = {"near": {SLOT_A}, "far": {SLOT_B},
                    "both": {SLOT_A, SLOT_B}}[mode]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not create output video: {output_path}")

    H = calibration["H"] if calibration else None
    minimap = MiniMap() if (show_minimap and calibration) else None
    # Shuttle trail length ~1s, matching the shuttle demo (deque of fps points).
    from collections import deque
    shuttle_trail = deque(maxlen=max(2, round(fps))) if shuttle_track else None

    frame_idx = 0
    try:
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if H is not None:
                overlay_court(frame, H)

            records = [r for r in by_frame.get(frame_idx, [])
                       if r["player_id"] in wanted_slots]
            for rec in records:
                phase = phases.get((frame_idx, rec["player_id"]))
                _draw_player(frame, rec, phase, threshold)

            if shuttle_trail is not None:
                xy = shuttle_track.get(frame_idx)
                shuttle_trail.append((int(round(xy[0])), int(round(xy[1])))
                                     if xy is not None else None)
                _draw_shuttle(frame, shuttle_trail)

            if minimap is not None:
                players = [(r["player_id"],
                            tuple(r["court_xy"]) if r.get("court_xy") else None)
                           for r in records]
                paste_inset(frame, minimap.render(players))

            if court_note:
                cv2.putText(frame, court_note, (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    print(f"[combined] wrote {frame_idx} frames -> {output_path}")
    return frame_idx


def _load_raw_pose(pose_path):
    """Read pose.jsonl records in their raw (original-pixel) form for drawing.

    Distinct from ``load_pose_jsonl`` (which bbox-normalizes for the model):
    drawing needs pixel coords, and we keep the canonical slot ``player_id``.
    """
    records = []
    with open(pose_path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(video_stem, args, device):
    from src.demo_pipeline import VIDEO_DIR

    video_path = VIDEO_DIR / f"{video_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    work = OUTPUT_DIR / video_stem
    work.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: court (LOO-gated) ---
    model, names = load_checkpoint(args.checkpoint, device)
    calibration_path, court_info = get_court_calibration(
        video_stem, video_path, model, names, device,
        frames=args.court_frames, save_overlay=False, overwrite=args.overwrite,
        loo_gate_m=args.loo_gate, write=True)

    court_note = ""
    calibration = None
    if calibration_path is not None:
        calibration = load_calibration(str(calibration_path))
    else:
        status = court_info["status"]
        if status == "rejected_loo":
            court_note = f"court uncalibrated (shaky fit, LOO {court_info['loo_mean_m']:.2f}m)"
        elif status == "no_fit":
            court_note = "court uncalibrated (no trustworthy fit)"
        else:
            court_note = "court uncalibrated"
        print(f"[combined] {court_note}; lines omitted, ROI off")

    # --- Stage 2: pose (+ racket keypoint) ---
    pose_path = get_pose_data(
        video_stem, video_path, calibration_path,
        save_preview_video=False, max_frames=args.max_frames,
        overwrite=args.overwrite,
        racket_weights=str(RACKET_WEIGHTS), racket_device=device)

    # --- Stage 3: canonicalize to NEAR/FAR slots ---
    canonicalize_pose(video_stem, video_path, pose_path, calibration_path)

    # --- Stage 4: court mapping (court_xy for the TCN court feature) ---
    map_court_coords(video_stem, pose_path, calibration_path)

    # --- Stage 5: shuttle -> speed signal (+ positions for the overlay) ---
    labels_csv = _prepare_shuttle(video_stem, video_path, work, args, device)
    shuttle_track = None
    if labels_csv is not None:
        _map_shuttle_from_csv(video_stem, pose_path, labels_csv, args.shuttle_span)
        # Same labels.csv feeds the on-frame trail, so the shuttle is drawn on
        # the combined frames rather than in a second video. {frame: (x,y)|None}.
        from src.detection.shuttle import load_shuttle_track
        shuttle_track = load_shuttle_track(str(labels_csv))
    else:
        print("[combined] WARNING: no shuttle signal; shuttle channel will read "
              "'unknown' and phase quality will drop (model trained WITH shuttle)")

    # --- Stage 6: phase inference ---
    phases = run_phase_inference(pose_path, PHASE_MODEL, args.window_size,
                                 args.smoothing, device)
    print(f"[combined] phases emitted: {len(phases)} (frame, player) labels")

    # --- Stage 7: render (court + boxes + pose + phase + shuttle + minimap) ---
    output_path = OUTPUT_DIR / f"{video_stem}_combined.mp4"
    render_combined(video_path, output_path, pose_path, phases, calibration,
                    court_note, args.mode, args.max_frames, args.threshold,
                    show_minimap=not args.no_minimap, shuttle_track=shuttle_track)
    return output_path


def _prepare_shuttle(video_stem, video_path, work, args, device):
    """Return a labels.csv path for the shuttle speed mapper, or None.

    Either consume a provided ``--shuttle-jsonl`` or run the tracker into a fresh
    temp dir. Both go through the same jsonl->csv bridge so normalization is
    identical.
    """
    if args.shuttle_jsonl:
        src_jsonl = Path(args.shuttle_jsonl)
        if not src_jsonl.exists():
            raise FileNotFoundError(f"--shuttle-jsonl not found: {src_jsonl}")
        print(f"[combined] using provided shuttle jsonl {src_jsonl}")
    else:
        if not SHUTTLE_WEIGHTS.exists():
            print(f"[combined] shuttle weights missing ({SHUTTLE_WEIGHTS}); skipping shuttle")
            return None
        # run_offline_demo needs a FRESH dir; use a unique temp sibling.
        track_dir = Path(tempfile.mkdtemp(prefix=f"{video_stem}_shuttle_", dir=str(work)))
        track_dir.rmdir()  # run_offline_demo will recreate it empty
        print(f"[combined] running shuttle tracker -> {track_dir}")
        src_jsonl = run_shuttle_tracker(video_path, track_dir, SHUTTLE_WEIGHTS,
                                        device, threshold=args.shuttle_threshold)
        # The tracker always writes its own shuttle_demo.mp4; we draw the shuttle
        # onto the combined video instead, so drop that throwaway second video.
        (track_dir / "shuttle_demo.mp4").unlink(missing_ok=True)
    return shuttle_jsonl_to_labels_csv(src_jsonl, work / "labels.csv")


def _map_shuttle_from_csv(video_stem, pose_path, labels_csv, span):
    """Add shuttle_speed to pose.jsonl using the FIXED training diagonal.

    Mirrors ``demo_pipeline.map_shuttle_speed`` but pins the frame size to
    ``SHUTTLE_NORM_WH`` (the diagonal the TCN trained with) rather than reading a
    per-clip meta.json, so a raw video with no meta is normalized identically.
    """
    from src.detection.shuttle import annotate_pose_with_shuttle

    w, h = SHUTTLE_NORM_WH
    tmp = Path(str(pose_path) + ".shuttmp")
    stats = annotate_pose_with_shuttle(str(pose_path), str(labels_csv), str(tmp),
                                       w, h, span)
    tmp.replace(pose_path)
    print(f"[combined] shuttle speed on {stats['valid_speed']}/{stats['records']} "
          f"records (valid_fraction={stats['valid_fraction']:.2f}, diag={math.hypot(w, h):.1f})")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video stem under data/videos (e.g. rally23)")
    parser.add_argument("--mode", choices=["near", "far", "both"], default="both",
                        help="which player slot(s) to label")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap frames processed/rendered (demo speed)")
    parser.add_argument("--window-size", type=int, default=15,
                        help="TCN temporal window (must match training)")
    parser.add_argument("--smoothing", type=int, default=5,
                        help="majority-vote smoothing span for emitted phases")
    parser.add_argument("--loo-gate", type=float, default=DEFAULT_LOO_GATE_M,
                        help="reject a court whose leave-one-out error (m) exceeds this")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="keypoint visibility/presence threshold for drawing")
    parser.add_argument("--shuttle-jsonl", default=None,
                        help="consume this shuttle_tracking.jsonl instead of running "
                             "the tracker (dev/CI; skips loading the shuttle model)")
    parser.add_argument("--shuttle-threshold", type=float, default=0.5,
                        help="shuttle detector confidence threshold")
    parser.add_argument("--shuttle-span", type=int, default=DEFAULT_SMOOTHING_SPAN,
                        help="smoothing span (frames) for shuttle speed")
    parser.add_argument("--no-minimap", action="store_true",
                        help="do not paste the top-down court minimap")
    parser.add_argument("--court-frames", type=int, default=15,
                        help="frames sampled for court prediction (fixed camera)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute court/pose even if cached outputs exist")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="models/court_final.pt")
    args = parser.parse_args()

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"[combined] device: {device}  mode: {args.mode}")
    out = run(args.video, args, device)
    print(f"[combined] done: {out}")


if __name__ == "__main__":
    main()
