"""Render frozen court-model predictions on every frame of a video."""
import json
import math
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from .draw import overlay_court
from .learning_data import POINT_NAMES, image_tensor
from .learning_model import decode_points
from .learning_train import choose_device, fit_predictions, load_model, tint_service_regions


def predict_video(args):
    """Write a new H.264 MP4, carrying source audio when it is present.

    Infer each frame independently. Do not carry a previous fit through a
    rejected frame, and do not imply that the plausibility check is accuracy.
    """
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}. Choose another --output directory")
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise RuntimeError("Video export needs: python -m pip install imageio-ffmpeg") from exc
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        video.release()
        raise ValueError(f"Cannot open video: {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    ok, first = video.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        video.release()
        raise ValueError("Cannot read video frames or frame rate")
    height, width = first.shape[:2]
    args.output.mkdir(parents=True)
    partial = args.output / "prediction.partial.mp4"
    output = args.output / "prediction.mp4"
    command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", "-i", str(args.video.resolve()),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:a", "aac", "-af", "apad",
               "-movflags", "+faststart", "-shortest", str(partial)]
    threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(min(threads, 4))
    frames = accepted = 0
    problem = None
    encoder = None
    with tempfile.TemporaryFile() as errors:
        try:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with (args.output / "frames.jsonl").open("w", encoding="utf-8") as log, torch.inference_mode():
                frame = first
                while frame is not None:
                    points = decode_points(model(image_tensor(frame, checkpoint["width"],
                                                               checkpoint["height"])[None].to(device)))[0].cpu().numpy()
                    points *= [width / checkpoint["width"], height / checkpoint["height"]]
                    fit = fit_predictions(points, (width, height),
                                          max(8., 6 * width / checkpoint["width"],
                                              6 * height / checkpoint["height"]))
                    view = frame.copy()
                    if fit is not None:
                        H, _ = fit
                        tint_service_regions(view, H)
                        overlay_court(view, H)
                        accepted += 1
                    else:
                        cv2.putText(view, "Court fit unavailable", (20, height - 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 100, 255), 2, cv2.LINE_AA)
                    cv2.line(view, tuple(np.rint(points[8]).astype(int)),
                             tuple(np.rint(points[9]).astype(int)), (255, 0, 255), 2)
                    for name, xy in zip(POINT_NAMES, points):
                        x, y = map(round, xy)
                        color = (255, 0, 255) if name in ("NL-TOP", "NR-TOP") else (0, 0, 255)
                        cv2.circle(view, (x, y), 5, color, -1)
                        cv2.putText(view, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                    encoder.stdin.write(view.tobytes())
                    log.write(json.dumps({"frame": frames, "court_fit_accepted": fit is not None,
                                          "points": dict(zip(POINT_NAMES, points.tolist()))}) + "\n")
                    frames += 1
                    if frames % 60 == 0:
                        print(f"Rendered {frames} frames", flush=True)
                    ok, frame = video.read()
                    if not ok:
                        frame = None
        except Exception as exc:
            problem = exc
        finally:
            video.release()
            if encoder is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
                code = encoder.wait()
            else:
                code = -1
            torch.set_num_threads(threads)
        errors.seek(0)
        details = errors.read().decode("utf-8", errors="replace")[-4000:]
    if problem is not None or code != 0:
        raise RuntimeError(f"Video export did not finish: {details or str(problem)}") from problem
    check = cv2.VideoCapture(str(partial))
    encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if check.isOpened() else 0
    check.release()
    if encoded_frames != frames:
        raise RuntimeError(f"Incomplete video export: encoded {encoded_frames} of {frames} frames")
    partial.rename(output)
    summary = {"source": str(args.video.resolve()), "checkpoint": str(args.checkpoint.resolve()),
               "output": str(output.resolve()), "fps": fps, "frames": frames,
               "duration_seconds": frames / fps, "court_fits_accepted": accepted,
               "court_fits_rejected": frames - accepted,
               "method": "Independent frozen-model inference on every frame; no temporal smoothing",
               "audio": "Source audio included when present",
               "note": "Fit acceptance is a geometry plausibility check, not an accuracy score."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}: {frames} frames; {accepted} accepted court fits.")
    return summary
