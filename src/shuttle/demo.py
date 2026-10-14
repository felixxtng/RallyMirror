"""Run shuttle inference independently of the existing player pipeline."""

import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import letterbox, letterbox_transform, stack_frames, to_original_point
from .model import ShuttleNet
from .train import choose_device


class ShuttleDetector:
    def __init__(self, weights, threshold=0.5, device="auto"):
        self.device = choose_device(device)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.width, self.height = checkpoint["width"], checkpoint["height"]
        self.model = ShuttleNet(checkpoint["base_channels"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.threshold = threshold

    @torch.inference_mode()
    def predict(self, previous, current, following):
        """Return original-image coordinates or None; always search the full frame."""
        height, width = current.shape[:2]
        t = letterbox_transform(width, height, self.width, self.height)
        frames = [letterbox(frame, t) for frame in [previous, current, following]]
        tensor = stack_frames(frames).unsqueeze(0).to(self.device)
        heatmap = self.model(tensor).sigmoid()[0, 0].cpu().numpy()
        # Padding is not an observable part of the video.
        allowed = np.zeros_like(heatmap, dtype=bool)
        x, y = t["pad_x"], t["pad_y"]
        allowed[y:y + t["resized_height"], x:x + t["resized_width"]] = True
        binary = ((heatmap >= self.threshold) & allowed).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        best = None
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < 3:
                continue
            mask = components == component
            score = float(heatmap[mask].max())
            if best is None or score > best["score"]:
                ys, xs = np.nonzero(mask)
                weights = heatmap[mask]
                hx = float(np.average(xs, weights=weights))
                hy = float(np.average(ys, weights=weights))
                ox, oy = to_original_point(hx, hy, t)
                best = {"x": round(min(width - 1, max(0, ox)), 2),
                        "y": round(min(height - 1, max(0, oy)), 2), "score": score}
        return best


def run_demo(video_path, weights, output_dir, threshold=0.5, device="auto"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "shuttle_demo.mp4"
    output_data = output_dir / "shuttle_tracking.jsonl"
    if output_video.exists() or output_data.exists():
        raise FileExistsError("Choose a fresh output directory to avoid replacing an earlier result")
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        video.release()
        raise ValueError(f"Cannot open {video_path}")
    writer = None
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        success, current = video.read()
        if not success or fps <= 0:
            raise ValueError("Video is empty or has invalid FPS")
        detector = ShuttleDetector(weights, threshold, device)
        height, width = current.shape[:2]
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot create {output_video}")
        previous = current
        has_next, following = video.read()
        trace = deque(maxlen=max(2, round(fps)))
        frame_number = 0
        with output_data.open("w") as file:
            while True:
                detection = detector.predict(previous, current, following if has_next else current)
                status = "detected" if detection is not None else "missing"
                row = {"frame": frame_number, "time_seconds": frame_number / fps,
                       "status": status, "x": None, "y": None, "score": None}
                if detection is not None:
                    row.update(detection)
                    point = (round(detection["x"]), round(detection["y"]))
                else:
                    point = None
                trace.append(point)
                canvas = current.copy()
                points = list(trace)
                for first, second in zip(points, points[1:]):
                    if first is not None and second is not None:
                        cv2.line(canvas, first, second, (0, 200, 0), 2)
                if point is not None:
                    cv2.circle(canvas, point, 10, (0, 255, 0), 2)
                cv2.putText(canvas, f"SHUTTLE: {status.upper()}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if detection else (0, 180, 255), 2)
                writer.write(canvas)
                file.write(json.dumps(row) + "\n")
                if frame_number % 100 == 0:
                    print(f"Processed {frame_number + 1} frames", flush=True)
                if not has_next:
                    break
                previous, current = current, following
                has_next, following = video.read()
                frame_number += 1
    finally:
        video.release()
        if writer is not None:
            writer.release()
    print(f"Saved {output_video}\nSaved {output_data}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("outputs/shuttle/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shuttle_demo"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    run_demo(args.video, args.weights, args.output, args.threshold, args.device)


if __name__ == "__main__":
    main()
