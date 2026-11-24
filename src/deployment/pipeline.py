"""One isolated video run, same trained feature contracts as combined_demo."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from .preflight import check_models
from .settings import ROOT


def inspect_video(path, max_seconds=60):
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30).stdout)
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise ValueError("Submit an MP4 containing exactly one video stream")
    stream = videos[0]
    duration = float(info["format"].get("duration", 0))
    num, den = map(int, stream["avg_frame_rate"].split("/"))
    fps = num / den if den else 0
    if not 0 < duration <= max_seconds or not 1 <= fps <= 60:
        raise ValueError(f"Video must be 1?60 fps and at most {max_seconds} seconds")
    # The existing TCN's shuttle normalization was trained on ~1280x720 footage.
    # Do not silently resize and alter coordinate semantics for other inputs.
    if (stream["width"], stream["height"]) != (1280, 720):
        raise ValueError("This model package currently requires 1280x720 input; export that resolution first")
    if stream.get("tags", {}).get("rotate", "0") != "0" or any(
            item.get("rotation", 0) != 0 for item in stream.get("side_data_list", [])):
        raise ValueError("Apply video rotation before uploading")
    return dict(duration_seconds=duration, fps=fps, width=1280, height=720)


def process(input_path, workspace, output_path, device="auto", max_seconds=60, progress=None):
    # Each invocation runs in its own subprocess with repository cwd. Never chdir
    # in the API process, share tracker state, or reuse another job's cached pose.
    started = time.perf_counter()
    models = check_models()
    input_path, workspace, output_path = map(lambda p: Path(p).resolve(), (input_path, workspace, output_path))
    workspace.mkdir(parents=True, exist_ok=False)
    metadata = inspect_video(input_path, max_seconds)
    import torch
    from src.combined_demo import run
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    times = {}
    active = [None, time.perf_counter()]
    def notify(stage):
        now = time.perf_counter()
        if active[0]:
            times[active[0]] = round(now - active[1], 3)
        active[:] = [stage, now]
        print("STAGE " + stage, flush=True)
        if progress:
            progress(stage)
    args = argparse.Namespace(checkpoint=str(ROOT / "models/court_final.pt"), court_frames=15,
        overwrite=True, loo_gate=.35, max_frames=None, shuttle_jsonl=None,
        shuttle_threshold=.5, shuttle_span=5, window_size=15, smoothing=5,
        mode="both", threshold=.5, no_minimap=False)
    # Use the repository default shuttle span rather than inventing a new feature.
    from src.detection.shuttle import DEFAULT_SMOOTHING_SPAN
    args.shuttle_span = DEFAULT_SMOOTHING_SPAN
    with torch.inference_mode():
        overlay = run("video", args, device, input_path=input_path, workspace=workspace, progress=notify)
    notify("encode")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name("result.encoding.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(overlay),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True, timeout=300)
    temporary.replace(output_path)
    notify("complete")
    calibration = workspace / "court.json"
    warnings = []
    if not calibration.exists():
        warnings.append("Court calibration rejected: lines, court mapping and court ROI unavailable; phase quality may drop")
    if calibration.exists():
        court_data = json.loads(calibration.read_text(encoding="utf-8-sig"))
        warnings.extend(court_data.get("warnings", []))
    pose_records = sum(1 for line in (workspace / "pose.jsonl").open(encoding="utf-8") if line.strip())
    if not pose_records:
        warnings.append("No player poses were produced; pose, racket and phase overlays unavailable")
    report = dict(pose_records=pose_records, input=metadata, device=device, models=models, stage_seconds=times,
        total_seconds=round(time.perf_counter()-started, 3), warnings=warnings,
        output="H.264 MP4, silent, original frame rate and resolution",
        feature_contract="18 keypoints; original shuttle normalization 960x544; existing temporal logic")
    output_path.with_name("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()
    process(args.input, args.workspace, args.output, args.device, args.max_seconds)

if __name__ == "__main__":
    main()
