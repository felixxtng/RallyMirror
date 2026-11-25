"""Produce seven browser-ready views from ONE real combined inference run.

Run with the worker's Python environment. FFmpeg and ffprobe must be on PATH.
Input must be a short 1280x720 raw rally without overlays (up to 60 seconds).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

FRONTEND = Path(__file__).resolve().parents[1]
BACKEND = FRONTEND / "backend"


def encode(source, destination):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True, timeout=300)


def export_views(video, run_directory):
    """Export focused layers from saved model outputs; do not rerun prediction."""
    from src.analysis_modes import MODES, render_analysis, summarize
    from src.deployment.pipeline import inspect_video
    report = json.loads((run_directory / "report.json").read_text(encoding="utf-8"))
    work = run_directory / "work"
    phase_frames = Counter(json.loads(line)["frame"] for line in (work / "phases.jsonl").read_text().splitlines() if line.strip())
    preferred_frame = round(report["frames"] * .75)
    preview_frame = max(phase_frames, key=lambda f: (phase_frames[f], -abs(f-preferred_frame))) if phase_frames else min(preferred_frame, max(0, report["frames"]-1))
    preview_time = preview_frame / report["input"]["fps"]
    public = FRONTEND / "public/media"
    public.mkdir(parents=True, exist_ok=True)
    metadata = inspect_video(video)
    source_clip = public / "source-analysis.mp4"
    encode(video, source_clip)
    rows = []
    for mode in MODES:
        destination = public / f"{mode}-analysis.mp4"
        if mode == "combined":
            shutil.copy2(run_directory / "result.mp4", destination)
        else:
            overlay = run_directory / f"{mode}-overlay.mp4"
            render_analysis(video, overlay, work, mode)
            encode(overlay, destination)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(preview_time), "-i", str(destination),
            "-frames:v", "1", "-q:v", "3", str(public / f"{mode}-analysis.jpg")], check=True, timeout=30)
        focused = summarize(work, mode, report["frames"], report.get("court"))
        focused.update(input=report["input"], device=report["device"], models=report["models"],
                       stage_seconds=report["stage_seconds"], total_seconds=report["total_seconds"],
                       output=report["output"], inference_mode="combined",
                       provenance="All seven views use one combined run's predictions. Processing time is for that shared run.",
                       feature_contract=report.get("feature_contract"))
        (public / f"{mode}-report.json").write_text(json.dumps(focused, indent=2), encoding="utf-8")
        rows.append(dict(mode=mode, video=destination.name, bytes=destination.stat().st_size,
                         sha256=hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()))
    provenance = dict(created=datetime.now(timezone.utc).isoformat(), source_clip=video.name,
        source_sha256=hashlib.file_digest(video.open("rb"), "sha256").hexdigest(),
        runtime_source="felixxtng/RallyMirror", runtime_commit="017a366aed9ec4091881b678412e14e2eca518f8",
        inference="fresh automatic inference with the six selected checkpoints; no manual ground-truth labels",
        caveat="Demonstration footage, not a held-out evaluation or an accuracy benchmark.",
        input=metadata, views=rows)
    (public / "demo-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def configure():
    sys.path.insert(0, str(BACKEND))
    os.chdir(BACKEND)
    cache = FRONTEND / ".runtime"
    for folder in (cache / "Ultralytics", cache / "matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-directory", type=Path, default=FRONTEND / ".work" / "demo-runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    video, directory = args.video.resolve(), args.run_directory.resolve()
    configure()
    from src.deployment.pipeline import process
    directory.mkdir(parents=True, exist_ok=False)
    process(video, directory / "work", directory / "result.mp4", args.device, analysis_mode="combined")
    print(json.dumps(export_views(video, directory), indent=2))


if __name__ == "__main__":
    main()
