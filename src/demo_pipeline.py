"""
Pipeline for converting an input rally video (or an array of videos) into
data suitable for the TCN Phase classifier model to train on.

Retrieves the following data from each rally video, where possible:
- Shuttle coordinates .csv (data/shuttle/clips/[video]/labels.csv)
- Player bounding boxes and keypoint coordinates JSON (data/shuttle/clips/[video]/pose.jsonl)
- Court calibration JSON (data/shuttle/clips/[video]_court-calibration.json)
- (OPTIONAL) Preview image showing court module's predicted keypoints (outputs/court_pred)
- (OPTIONAL) Preview video showing detection-pose modules' predicted bbox and keypoints (outputs/videos)

DOES NOT retrieve pose annotations for the temporal module; requires manual labelling.

Stage order (each stage depends on the previous one's output):

    1. COURT      video -> data/shuttle/clips/<video>_court-calibration.json  (+ overlay)
    2. POSE       video + calibration -> data/shuttle/clips/<video>/pose.jsonl
                  (the calibration drives the court ROI, so it runs BEFORE pose)
    3. CANONICAL  pose.jsonl + calibration -> stable player slots, in place

A missing calibration does not abort the clip: pose still runs (ROI off) and
canonicalization falls back to gap-stitching. Each stage can be skipped so the
pipeline can, e.g., re-canonicalize existing pose.jsonl files without
re-running the (slow) pose model -- which is goal (1), updating old data.

Diagnostics
-----------
Auto-predicted courts are not always trustworthy (angle changes, occluded
lines, moving cameras), and a wrong court silently corrupts every player
position downstream. To make a batch observable:

  * a LOO quality gate (``--loo-gate`` metres) rejects a predicted calibration
    whose leave-one-out error is too high, so the clip falls back to ROI-off /
    gap-stitching and is flagged rather than trusted to a shaky court;
  * a batch SUMMARY (printed, optionally ``--summary-json``) lists per clip
    whether the court succeeded (+ loo), the pose frame count, and the
    canonicalization mode + rejected count;
  * ``--dry-run`` predicts courts and reports what WOULD happen per clip
    (calibration quality, resulting canonicalization mode) without running the
    expensive pose stage or writing pose files -- cheap triage before the big
    pass.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

# Court
from src.court.net import load_checkpoint
from src.court.predict import predict_calibration, draw_prediction
from src.court.calibration import save_calibration

# Detection / Pose
from src.pose.demo import run_demo as run_pose_demo

# Identity canonicalization (fragmented tracker ids -> stable slots)
from src.detection.canonicalize import canonicalize_file

# Court-space mapping (foot pixel -> court metres) via court mapper
from src.court.mapper import map_jsonl, load_config as load_court_config

# Shuttle rally-state signal (smoothed shuttle speed + validity per frame)
from src.detection.shuttle import annotate_pose_with_shuttle, DEFAULT_SMOOTHING_SPAN


# Paths (kept together so they are easy to retarget).
VIDEO_DIR = Path("data/videos")
CALIB_OVERLAY_DIR = Path("outputs/court_pred")
CLIPS_DIR = Path("data/shuttle/clips")
# PREVIEW_VIDEO_DIR = Path("outputs/videos")
PREVIEW_VIDEO_DIR = Path("C:/Users/tayru/Downloads/RallyMirror-outputs")
POSE_MODEL = Path("models/pose_landmarker_lite.task")
COURT_CHECKPOINT = Path("models/court_final.pt")

# Default leave-one-out gate (metres). Mirrors the court module's own
# "re-click more carefully" warning threshold (calibration.WARN_LOO_M = 0.35).
DEFAULT_LOO_GATE_M = 0.35


def calib_path_for(video):
    return CLIPS_DIR / video / f"{video}_court-calibration.json"


def pose_path_for(video):
    return CLIPS_DIR / video / "pose.jsonl"


def get_court_calibration(video, video_path, model, names, device,
                          frames=15, conf=0.3, save_overlay=True, overwrite=False,
                          loo_gate_m=DEFAULT_LOO_GATE_M, write=True):
    """Stage 1. Predict a court calibration JSON.

    Returns ``(calibration_path_or_None, court_info)`` where ``court_info`` is a
    diagnostics dict for the batch summary::

        {"status": "ok" | "reused" | "rejected_loo" | "no_fit",
         "loo_mean_m": float | None, "reason": str | None}

    The LOO gate rejects a fit whose leave-one-out error exceeds ``loo_gate_m``:
    such a clip is treated as uncalibrated (ROI off downstream) and flagged, so
    a shaky auto-court never silently corrupts player positions. When
    ``write`` is False (dry-run) nothing is saved, but the fit + gate still run
    so the report reflects what would happen.
    """
    out_path = calib_path_for(video)
    if out_path.exists() and not overwrite:
        loo = _existing_loo(out_path)
        print(f"  [court] using existing {out_path} (loo_mean_m={loo})")
        return out_path, {"status": "reused", "loo_mean_m": loo, "reason": None}

    calib, first, points, conf_arr, info = predict_calibration(
        model, names, video_path, frames=frames, conf=conf, device=device)
    if calib is None:
        reason = info.get("reason")
        print(f"  [court] no trustworthy fit ({reason}); ROI will be off")
        return None, {"status": "no_fit", "loo_mean_m": None, "reason": reason}

    loo = calib["stats"].get("loo_mean_m")
    # LOO gate: refuse a plausible-but-shaky auto-court.
    if loo is not None and loo_gate_m is not None and loo > loo_gate_m:
        print(f"  [court] REJECTED by LOO gate: loo_mean_m={loo:.3f} > "
              f"{loo_gate_m} m; ROI will be off (flag for manual calibration)")
        return None, {"status": "rejected_loo", "loo_mean_m": loo,
                      "reason": f"loo {loo:.3f} > gate {loo_gate_m}"}

    if write:
        save_calibration(out_path, calib)
        print(f"  [court] saved {out_path}  (loo_mean_m={loo})")
        for warning in calib["warnings"]:
            print(f"  [court] WARNING: {warning}")
        if save_overlay:
            overlay_path = CALIB_OVERLAY_DIR / f"{video}.jpg"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay = draw_prediction(first, calib, names, points, conf_arr, conf)
            cv2.imwrite(str(overlay_path), overlay)
            print(f"  [court] saved overlay {overlay_path}")
    else:
        print(f"  [court] (dry-run) would save {out_path}  (loo_mean_m={loo})")

    return out_path, {"status": "ok", "loo_mean_m": loo, "reason": None}


def _existing_loo(calib_path):
    """Best-effort read of loo_mean_m from an existing calibration JSON."""
    try:
        data = json.loads(Path(calib_path).read_text(encoding="utf-8-sig"))
        return data.get("stats", {}).get("loo_mean_m")
    except (OSError, ValueError):
        return None


def get_pose_data(video, video_path, calibration_path, save_preview_video=True,
                  threshold=0.5, max_frames=None, overwrite=False,
                  racket_weights=None, racket_device="cpu"):
    """Stage 2. Run detection+pose (with the court ROI if a calibration exists)
    and write data/shuttle/clips/<video>/pose.jsonl. Returns its path.

    When ``racket_weights`` is given, each pose record carries an extra
    ``racket_center`` keypoint (18 keypoints total) for a racket-aware temporal
    model. A model trained on 17-keypoint data is incompatible with 18-keypoint
    pose.jsonl and vice versa, so regenerate the whole dataset when switching.
    """
    data_path = pose_path_for(video)
    if data_path.exists() and not overwrite:
        print(f"  [pose] using existing {data_path}")
        return data_path

    # run_demo always writes a preview video; point it somewhere real.
    preview = PREVIEW_VIDEO_DIR / f"{video}_tracked.mp4"
    preview.parent.mkdir(parents=True, exist_ok=True)
    run_pose_demo(
        video_path, POSE_MODEL, preview, data_path,
        threshold=threshold, max_frames=max_frames,
        calibration_path=str(calibration_path) if calibration_path else None,
        racket_weights=racket_weights, racket_device=racket_device,
    )
    if not save_preview_video:
        preview.unlink(missing_ok=True)
    return data_path


def get_pose_frame_size(video_path):
    """(width, height) of the decoded frames -- needed to scale foot pixels into
    the calibration's image space during canonicalization."""
    cap = cv2.VideoCapture(str(video_path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return size


def canonicalize_pose(video, video_path, pose_path, calibration_path):
    """Stage 3. Rewrite pose.jsonl player ids -> stable slots (court path when a
    calibration exists, gap-stitching fallback otherwise). Returns stats or None.
    """
    if not Path(pose_path).exists():
        print(f"  [canon] no pose file at {pose_path}; skipping")
        return None
    frame_size = get_pose_frame_size(video_path)
    stats = canonicalize_file(
        str(pose_path), str(pose_path),            # rewrite in place
        calibration_path=str(calibration_path) if calibration_path else None,
        frame_size=frame_size,
    )
    print(f"  [canon] mode={stats['mode']} raw_ids={stats['raw_ids']} -> "
          f"slots={stats['slot_to_raw_ids']} rejected={stats['rejected']}")
    return stats


def map_court_coords(video, pose_path, calibration_path):
    """Stage 4. Add court-space fields (``court_xy`` metres, ``in_bounds``, ...)
    to each pose.jsonl record via court mapper, rewriting in place.

    This is what makes the temporal court feature (train.py ``--court``)
    trainable: ``court_xy`` is read from these records. Needs a calibration;
    without one the stage is skipped and records keep pose only.
    Returns the mapper stats dict, or None if skipped.
    """
    if not calibration_path or not Path(calibration_path).exists():
        print("  [court-map] no calibration; pose records keep no court_xy")
        return None
    if not Path(pose_path).exists():
        print(f"  [court-map] no pose file at {pose_path}; skipping")
        return None
    from src.court.calibration import load_calibration

    H = load_calibration(str(calibration_path))["H"]
    cfg = load_court_config()
    # map_jsonl opens the output for writing (truncating) while reading the
    # input, so an in-place path would be clobbered before it is read. Write to
    # a temp sibling and atomically replace.
    tmp_path = Path(str(pose_path) + ".courttmp")
    stats = map_jsonl(str(pose_path), str(tmp_path), H, cfg)
    tmp_path.replace(pose_path)
    print(f"  [court-map] mapped {stats['mapped']}/{stats['records']} records "
          f"in_bounds={stats['in_bounds']} sources={stats['sources']}")
    return stats


def _shuttle_frame_size(video):
    """Frame size the shuttle labels.csv coordinates live in.

    labels.csv coords are in the clip's letterboxed detection space, recorded in
    meta.json under ``transform`` (width/height). Fall back to meta's original
    size, then 960x544, if absent.
    """
    meta_path = CLIPS_DIR / video / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            tf = meta.get("transform") or {}
            w = tf.get("width") or meta.get("original_width")
            h = tf.get("height") or meta.get("original_height")
            if w and h:
                return int(w), int(h)
        except (OSError, ValueError):
            pass
    return 960, 544


def map_shuttle_speed(video, pose_path, smoothing_span=DEFAULT_SMOOTHING_SPAN,
                      min_valid_fraction=0.1):
    """Stage 5. Add ``shuttle_speed`` / ``shuttle_speed_valid`` to each pose
    record from the clip's shuttle ``labels.csv``, rewriting in place.

    This makes the temporal shuttle feature (train.py ``--shuttle``) trainable.
    Needs labels.csv; without it the stage is skipped. Returns the stats dict
    (or None), and flags clips whose valid-speed fraction is suspiciously low
    (likely a failed shuttle track, not genuine dead time).
    """
    labels_csv = CLIPS_DIR / video / "labels.csv"
    if not labels_csv.exists():
        print(f"  [shuttle] no labels.csv for {video}; pose keeps no shuttle_speed")
        return None
    if not Path(pose_path).exists():
        print(f"  [shuttle] no pose file at {pose_path}; skipping")
        return None
    w, h = _shuttle_frame_size(video)
    tmp_path = Path(str(pose_path) + ".shuttmp")
    stats = annotate_pose_with_shuttle(
        str(pose_path), str(labels_csv), str(tmp_path), w, h, smoothing_span)
    tmp_path.replace(pose_path)
    frac = stats["valid_fraction"]
    flag = "  LOW-VALID (check shuttle track)" if frac < min_valid_fraction else ""
    print(f"  [shuttle] speed on {stats['valid_speed']}/{stats['records']} "
          f"records (valid_fraction={frac:.2f}) span={smoothing_span}{flag}")
    stats["low_valid"] = frac < min_valid_fraction
    return stats


def process_video(video, model, names, device, args):
    """Run the (enabled) stages for one clip. Returns a summary record."""
    print(f"Opening video: {video}")
    record = {"video": video, "court": None, "loo_mean_m": None,
              "pose_records": None, "canon_mode": None, "rejected": None,
              "court_mapped": None, "shuttle_valid_frac": None, "notes": []}

    video_path = VIDEO_DIR / f"{video}.mp4"
    if not video_path.exists():
        print(f"  SKIP: {video_path} not found")
        record["court"] = "missing_video"
        record["notes"].append("video not found")
        return record

    # --- Stage 1: court ---
    calibration_path = None
    if not args.skip_court:
        calibration_path, court_info = get_court_calibration(
            video, video_path, model, names, device,
            frames=args.court_frames, save_overlay=not args.no_overlay,
            overwrite=args.overwrite, loo_gate_m=args.loo_gate,
            write=not args.dry_run)
        record["court"] = court_info["status"]
        record["loo_mean_m"] = court_info["loo_mean_m"]
        if court_info["reason"]:
            record["notes"].append(court_info["reason"])
    else:
        existing = calib_path_for(video)
        calibration_path = existing if existing.exists() else None
        record["court"] = "skipped"
        record["loo_mean_m"] = _existing_loo(existing) if calibration_path else None
        print(f"  [court] skipped (using {calibration_path or 'no calibration'})")

    if calibration_path is None and record["court"] not in ("missing_video",):
        record["notes"].append("ROI off (no calibration)")

    # --- Dry-run: report the would-be canonicalization mode and stop. ---
    if args.dry_run:
        mode = "court" if calibration_path else "gaps"
        record["canon_mode"] = f"would-be:{mode}"
        has_pose = pose_path_for(video).exists()
        print(f"  [dry-run] canon would use mode={mode}; "
              f"pose.jsonl {'exists' if has_pose else 'absent'}")
        return record

    # --- Stage 2: pose ---
    pose_path = pose_path_for(video)
    if not args.skip_pose:
        pose_path = get_pose_data(
            video, video_path, calibration_path,
            save_preview_video=not args.no_preview_video,
            max_frames=args.max_frames, overwrite=args.overwrite,
            racket_weights=args.racket_weights, racket_device=args.racket_device)
    else:
        print(f"  [pose] skipped (using {pose_path})")

    # --- Stage 3: canonicalize ---
    if not args.skip_canon:
        stats = canonicalize_pose(video, video_path, pose_path, calibration_path)
        if stats is not None:
            record["pose_records"] = stats["records"]
            record["canon_mode"] = stats["mode"]
            record["rejected"] = stats["rejected"]
            if stats["rejected"]:
                record["notes"].append(f"{stats['rejected']} records rejected")
    else:
        print("  [canon] skipped")

    # --- Stage 4: court mapping (adds court_xy for the temporal court feature) ---
    # Runs AFTER canonicalization so the final records carry BOTH stable slots
    # and court position. No-op without a calibration.
    if not args.skip_court_map:
        cstats = map_court_coords(video, pose_path, calibration_path)
        record["court_mapped"] = (cstats["mapped"] if cstats else 0)
        if calibration_path and not cstats:
            record["notes"].append("court mapping skipped")
    else:
        print("  [court-map] skipped")

    # --- Stage 5: shuttle mapping (adds shuttle_speed for the --shuttle feature) ---
    # Runs last; independent of court. No-op without a labels.csv.
    if not args.skip_shuttle_map:
        sstats = map_shuttle_speed(video, pose_path, smoothing_span=args.shuttle_span)
        record["shuttle_valid_frac"] = (round(sstats["valid_fraction"], 3)
                                        if sstats else None)
        if sstats and sstats.get("low_valid"):
            record["notes"].append("low shuttle valid-fraction")
    else:
        print("  [shuttle] skipped")

    return record


def print_summary(records, summary_json=None):
    """Print a one-line-per-clip batch summary and optionally write it to JSON."""
    print("\n" + "=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    header = (f"{'video':<12} {'court':<13} {'loo_m':>7} {'pose':>6} {'canon':>6} "
              f"{'rej':>4} {'crtmap':>7} {'shtlok':>7}  notes")
    print(header)
    print("-" * len(header))
    flagged = []
    for r in records:
        loo = f"{r['loo_mean_m']:.3f}" if isinstance(r["loo_mean_m"], (int, float)) else "-"
        pose = r["pose_records"] if r["pose_records"] is not None else "-"
        canon = r["canon_mode"] or "-"
        rej = r["rejected"] if r["rejected"] is not None else "-"
        crtmap = r["court_mapped"] if r.get("court_mapped") is not None else "-"
        svf = r.get("shuttle_valid_frac")
        shtl = f"{svf:.2f}" if isinstance(svf, (int, float)) else "-"
        notes = "; ".join(r["notes"])
        print(f"{r['video']:<12} {str(r['court']):<13} {loo:>7} {str(pose):>6} "
              f"{str(canon):>6} {str(rej):>4} {str(crtmap):>7} {shtl:>7}  {notes}")
        # Flag clips that need attention: no/failed/shaky court, or rejections.
        if r["court"] in ("no_fit", "rejected_loo", "missing_video") or r["rejected"]:
            flagged.append(r["video"])
    print("-" * len(header))
    if flagged:
        print(f"FLAGGED ({len(flagged)}): {', '.join(flagged)}")
        print("  -> consider a manual calibration (python -m src.court.calibrate ...)")
    else:
        print("No clips flagged.")

    if summary_json:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote summary: {summary_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*", help="video stems (default: the hardcoded list)")
    parser.add_argument("--court-frames", type=int, default=15,
                        help="frames sampled for court prediction (fixed camera only)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap frames processed by the pose model (debugging)")
    parser.add_argument("--loo-gate", type=float, default=DEFAULT_LOO_GATE_M,
                        help="reject a predicted court whose leave-one-out error (m) "
                             "exceeds this; use a large value to disable")
    parser.add_argument("--dry-run", action="store_true",
                        help="predict courts + report what would happen per clip; "
                             "do NOT run pose or write pose/calibration files")
    parser.add_argument("--summary-json", default=None,
                        help="also write the batch summary to this JSON path")
    parser.add_argument("--skip-court", action="store_true")
    parser.add_argument("--skip-pose", action="store_true",
                        help="reuse existing pose.jsonl (e.g. to only re-canonicalize)")
    parser.add_argument("--skip-canon", action="store_true")
    parser.add_argument("--skip-court-map", action="store_true",
                        help="do not add court_xy to pose records (stage 4); "
                             "the temporal --court feature needs this stage")
    parser.add_argument("--skip-shuttle-map", action="store_true",
                        help="do not add shuttle_speed to pose records (stage 5); "
                             "the temporal --shuttle feature needs this stage")
    parser.add_argument("--shuttle-span", type=int, default=DEFAULT_SMOOTHING_SPAN,
                        help="smoothing span (frames) for shuttle speed")
    parser.add_argument("--racket-weights", default=None,
                        help="racket ownership checkpoint (e.g. "
                             "models/racket_center_v006.pt); when given, pose.jsonl "
                             "gains a racket_center keypoint (18 total) for a "
                             "racket-aware temporal model. Regenerate ALL clips "
                             "with --overwrite when switching 17<->18 keypoints")
    parser.add_argument("--racket-device", default="cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute court/pose even if their outputs exist")
    parser.add_argument("--no-overlay", action="store_true", help="skip court overlay image")
    parser.add_argument("--no-preview-video", action="store_true",
                        help="delete the pose preview video after writing pose.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=str(COURT_CHECKPOINT))
    args = parser.parse_args()

    videos = args.videos or [
        "rally41", "rally42", "rally43", "rally44", "rally45",
        "rally46", "rally47", "rally48"
    ]

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    print(f"Device: {device}")
    if args.dry_run:
        print("DRY RUN: predicting courts only; no pose/calibration files written.")

    # Load the court model once. A dry-run always needs it (court is the whole
    # point); a normal run needs it unless the court stage is skipped.
    model, names = (None, None)
    if not args.skip_court or args.dry_run:
        model, names = load_checkpoint(args.checkpoint, device)

    records = [process_video(v, model, names, device, args) for v in videos]
    print_summary(records, args.summary_json)
    print("Done.")


if __name__ == "__main__":
    main()
