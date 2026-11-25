"""Deployable analysis modes using the team's selected models and feature contracts.

Each run owns its artifacts. A combined run can also be rendered as six focused
views without running the models again; those views use the same predictions.
No ground-truth annotations are read by this module.
"""
from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path

MODES = ("combined", "court", "tracking", "pose", "racket", "shuttle", "phases")
MODEL_PATHS = {
    "court": {"models/court_final.pt"},
    "tracking": {"models/yolo11n.pt"},
    "pose": {"models/yolo11n.pt", "models/pose_landmarker_lite.task"},
    "racket": {"models/yolo11n.pt", "models/pose_landmarker_lite.task", "models/racket_center_v006.pt"},
    "shuttle": {"models/shuttle_full_b8/best.pt"},
}
MODEL_PATHS["combined"] = MODEL_PATHS["court"] | MODEL_PATHS["racket"] | MODEL_PATHS["shuttle"] | {"models/phase_tcn.pt"}
MODEL_PATHS["phases"] = MODEL_PATHS["combined"]


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")


def track_players(video_path, output_path, device):
    """Track only: do not run pose or racket estimation for this mode."""
    import cv2
    from src.detection.tracker import PlayerTracker
    from src.detection.target_selector import load_selection_config, select_targets

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not decode the input video")
    tracker, config = PlayerTracker(device=device), load_selection_config()
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    records, number = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for target in select_targets(tracker.track_people(frame), config, frame_size):
                records.append(dict(frame=number, player_id=target["track_id"], bbox=target["bbox"],
                                    confidence=target["confidence"], keypoints=[]))
            number += 1
    finally:
        cap.release()
    if not number:
        raise ValueError("Input video contains no readable frames")
    write_jsonl(output_path, records)


def run_analysis(mode, video_path, workspace, device="cpu", progress=None):
    """Run only the dependencies required by ``mode`` and return overlay + diagnostics."""
    if mode not in MODES:
        raise ValueError(f"Unsupported analysis mode: {mode}")
    import cv2
    from src.combined_demo import (
        run_shuttle_tracker, shuttle_jsonl_to_labels_csv, _map_shuttle_from_csv,
        run_phase_inference, PHASE_MODEL, RACKET_WEIGHTS, SHUTTLE_WEIGHTS,
    )
    from src.demo_pipeline import (
        get_court_calibration, get_pose_data, canonicalize_pose, map_court_coords,
    )
    from src.detection.shuttle import DEFAULT_SMOOTHING_SPAN

    work, video_path = Path(workspace), Path(video_path)
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise FileExistsError("Analysis workspace must be empty")
    notify = progress or (lambda stage: None)
    court_info, calibration_path = None, None
    pose_path = work / "pose.jsonl"
    needs_court = mode in {"combined", "phases", "court"}
    needs_pose = mode in {"combined", "phases", "pose", "racket"}
    needs_racket = mode in {"combined", "phases", "racket"}

    if needs_court:
        from src.court.net import load_checkpoint
        notify("court")
        model, names = load_checkpoint("models/court_final.pt", device)
        calibration_path, court_info = get_court_calibration(
            "video", video_path, model, names, device, frames=15, save_overlay=False,
            overwrite=True, loo_gate_m=.35, output_path=work / "court.json")
        del model
    if needs_pose:
        notify("pose_and_racket" if needs_racket else "pose")
        get_pose_data("video", video_path, calibration_path, save_preview_video=False,
                      overwrite=True, racket_weights=str(RACKET_WEIGHTS) if needs_racket else None,
                      racket_device=device, tracker_device=device, output_path=pose_path)
    elif mode == "tracking":
        notify("tracking")
        track_players(video_path, pose_path, device)
    if pose_path.exists():
        notify("canonicalize_and_map")
        canonicalize_pose("video", video_path, pose_path, calibration_path)
        if calibration_path:
            map_court_coords("video", pose_path, calibration_path)
    if mode in {"combined", "phases", "shuttle"}:
        notify("shuttle")
        tracking_path = run_shuttle_tracker(video_path, work / "shuttle", SHUTTLE_WEIGHTS, device)
        labels = shuttle_jsonl_to_labels_csv(tracking_path, work / "labels.csv")
        if pose_path.exists():
            _map_shuttle_from_csv("video", pose_path, labels, DEFAULT_SMOOTHING_SPAN)
    if mode in {"combined", "phases"}:
        notify("phase")
        phases = run_phase_inference(pose_path, PHASE_MODEL, 15, 5, device)
        write_jsonl(work / "phases.jsonl", [dict(frame=frame, player_id=pid, phase=phase)
                    for (frame, pid), phase in sorted(phases.items())])

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    summary = summarize(work, mode, total_frames, court_info)
    (work / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    notify("render")
    overlay = work / "overlay.mp4"
    render_analysis(video_path, overlay, work, mode)
    return overlay, summary


def summarize(work, mode, total_frames, court_info=None):
    """Coverage counts are diagnostics, never model accuracy scores."""
    records = [r for r in read_jsonl(Path(work) / "pose.jsonl") if r["player_id"] != -1]
    shuttle = read_jsonl(Path(work) / "shuttle/shuttle_tracking.jsonl")
    phases = read_jsonl(Path(work) / "phases.jsonl")
    body = [r for r in records if any(p["name"] != "racket_center" for p in r.get("keypoints", []))]
    racket = [r for r in records if any(p["name"] == "racket_center" and p.get("visibility", 0) >= .5
              and p.get("presence", 0) >= .5 for p in r.get("keypoints", []))]
    court_ok = (Path(work) / "court.json").is_file()
    requested = set(MODES[1:]) if mode == "combined" else {mode}
    counts = dict(court=int(court_ok), tracking=len(records), pose=len(body), racket=len(racket),
                  shuttle=sum(r.get("status") == "detected" for r in shuttle), phases=len(phases))
    modules = {}
    for name, count in counts.items():
        dependencies = mode == "phases" or (mode == "racket" and name in {"tracking", "pose"}) or (mode == "pose" and name == "tracking")
        active = name in requested or dependencies
        modules[name] = dict(status=("available" if count else "unavailable") if active else "not_requested",
                             records=count if active else 0)
    warnings = []
    if court_ok:
        calibration = json.loads((Path(work) / "court.json").read_text(encoding="utf-8"))
        warnings.extend(str(w).replace("re-click more carefully", "review the predicted court points").replace("mis-click", "misaligned prediction") for w in calibration.get("warnings", []))
    if mode in {"combined", "court", "phases"} and not court_ok:
        message = "Court calibration rejected: court lines and court mapping unavailable."
        warnings.append(message + (" Phase estimates have less context." if mode in {"combined", "phases"} else ""))
    if mode == "tracking" and not records:
        warnings.append("No player tracks were produced.")
    if mode in {"combined", "pose", "racket", "phases"} and not body:
        warnings.append("No player poses were produced; pose, racket and phase overlays unavailable.")
    if mode in {"combined", "racket"} and not racket:
        warnings.append("No racket centres met the display confidence threshold.")
    if mode in {"combined", "shuttle", "phases"} and not counts["shuttle"]:
        warnings.append("No shuttle positions met the display confidence threshold.")
    if mode in {"combined", "phases"} and not phases:
        warnings.append("No complete temporal windows were available for phase predictions.")
    return dict(analysis_mode=mode, frames=total_frames, pose_records=len(records),
                modules=modules, warnings=warnings, court=court_info,
                player_slots=sorted({r["player_id"] for r in records}),
                phase_counts=dict(Counter(r["phase"] for r in phases)) if mode in {"combined", "phases"} else {},
                coverage_note="Counts describe available predictions, not correctness or evaluation accuracy.")


def render_analysis(video_path, output_path, workspace, mode="combined"):
    """Render one focused view or all six layers from the same prediction artifacts."""
    if mode not in MODES:
        raise ValueError(f"Unsupported analysis mode: {mode}")
    import cv2
    from src.combined_demo import _draw_player, _draw_shuttle, PHASE_COLORS, SLOT_NAMES, WARMING_UP
    from src.court.calibration import load_calibration
    from src.court.draw import MiniMap, overlay_court, paste_inset
    from src.pose.estimator import CONNECTIONS
    from src.detection.shuttle import load_shuttle_track

    work = Path(workspace)
    by_frame = {}
    for record in read_jsonl(work / "pose.jsonl"):
        if record["player_id"] != -1:
            by_frame.setdefault(record["frame"], []).append(record)
    phases = {(r["frame"], r["player_id"]): r["phase"] for r in read_jsonl(work / "phases.jsonl")}
    calibration = load_calibration(str(work / "court.json")) if (work / "court.json").is_file() else None
    shuttle = load_shuttle_track(str(work / "labels.csv")) if (work / "labels.csv").is_file() else {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not read source video for rendering")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(3)), int(cap.get(4))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise OSError("Could not create analysis video")
    minimap = MiniMap() if calibration and mode == "combined" else None
    trail, number = deque(maxlen=max(2, round(fps))), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            players = by_frame.get(number, [])
            if calibration and mode in {"court", "combined"}:
                overlay_court(frame, calibration["H"])
            if mode in {"combined", "tracking", "pose", "racket", "phases"}:
                for player in players:
                    phase = phases.get((number, player["player_id"]))
                    if mode == "combined":
                        _draw_player(frame, player, phase, .5)
                    else:
                        x1, y1, x2, y2 = (int(round(v)) for v in player["bbox"])
                        color = PHASE_COLORS.get(phase, (85, 225, 155)) if mode == "phases" else (85, 225, 155)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        slot = SLOT_NAMES.get(player["player_id"], f"P{player['player_id']}")
                        label = f"{slot}: {phase or WARMING_UP}" if mode == "phases" else slot
                        cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
                        points = {p["name"]: p for p in player.get("keypoints", []) if p.get("visibility", 0) >= .5 and p.get("presence", 0) >= .5}
                        if mode == "pose":
                            for a, b in CONNECTIONS:
                                if a in points and b in points:
                                    cv2.line(frame, (round(points[a]["x"]), round(points[a]["y"])),
                                             (round(points[b]["x"]), round(points[b]["y"])), (0, 240, 255), 2, cv2.LINE_AA)
                            for p in points.values():
                                cv2.circle(frame, (round(p["x"]), round(p["y"])), 3, (220, 100, 245), -1, cv2.LINE_AA)
                    if mode in {"racket", "combined"}:
                        for p in player.get("keypoints", []):
                            if p["name"] == "racket_center" and p.get("visibility", 0) >= .5 and p.get("presence", 0) >= .5:
                                center = (round(p["x"]), round(p["y"]))
                                cv2.circle(frame, center, 10, (0, 185, 255), 2, cv2.LINE_AA)
                                if mode == "racket":
                                    cv2.putText(frame, "racket", (center[0]+12, center[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 185, 255), 1, cv2.LINE_AA)
            if mode in {"combined", "shuttle"}:
                point = shuttle.get(number)
                trail.append((round(point[0]), round(point[1])) if point is not None else None)
                _draw_shuttle(frame, trail)
            if minimap:
                paste_inset(frame, minimap.render([(p["player_id"], tuple(p["court_xy"]) if p.get("court_xy") else None) for p in players]))
            cv2.rectangle(frame, (14, 14), (225, 49), (18, 25, 22), -1)
            cv2.putText(frame, "RALLYMIRROR / " + mode.upper(), (24, 37), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 245, 230), 1, cv2.LINE_AA)
            if mode in {"court", "combined"} and not calibration:
                cv2.putText(frame, "Court calibration unavailable", (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            number += 1
    finally:
        cap.release()
        writer.release()
    if not number:
        raise ValueError("Source video has no readable frames")
    return number
