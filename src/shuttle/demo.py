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
from .tracker import TemporalTracker, TrackerConfig


class ShuttleDetector:
    def __init__(self, weights, threshold=0.5, device="auto"):
        self.device = choose_device(device)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.width, self.height = checkpoint["width"], checkpoint["height"]
        self.model = ShuttleNet(checkpoint["base_channels"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.threshold = threshold

    def predict(self, previous, current, following):
        """Compatible highest-confidence detection (x, y, score), or None."""
        candidates = self.predict_candidates(previous, current, following)
        return {k: candidates[0][k] for k in ("x", "y", "score")} if candidates else None

    @torch.inference_mode()
    def predict_candidates(self, previous, current, following):
        """All components, confidence sorted; area is measured in heatmap pixels."""
        height, width = current.shape[:2]
        t = letterbox_transform(width, height, self.width, self.height)
        frames = [letterbox(frame, t) for frame in [previous, current, following]]
        tensor = stack_frames(frames).unsqueeze(0).to(self.device)
        heatmap = self.model(tensor).sigmoid()[0, 0].cpu().numpy()
        return extract_candidates(heatmap, t, width, height, self.threshold)


def extract_candidates(heatmap, t, width, height, threshold=0.5):
    """Exclude padding and tiny components; preserve weighted-centroid convention."""
    allowed = np.zeros_like(heatmap, dtype=bool)
    x, y = t["pad_x"], t["pad_y"]
    allowed[y:y+t["resized_height"], x:x+t["resized_width"]] = True
    binary = ((heatmap >= threshold) & allowed).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        mask = components == component
        ys, xs = np.nonzero(mask)
        values = heatmap[mask]
        ox, oy = to_original_point(float(np.average(xs, weights=values)),
                                   float(np.average(ys, weights=values)), t)
        candidates.append({"x": round(min(width-1, max(0, ox)), 2),
                           "y": round(min(height-1, max(0, oy)), 2),
                           "score": float(values.max()), "area": area})
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def run_demo(video_path, weights, output_dir, threshold=0.5, device="auto",
             temporal_filter=False, tracker_config=None, offline_refine=False,
             offline_config=None, candidate_cache=None, comparison_video=False):
    if temporal_filter and offline_refine:
        raise ValueError("Choose temporal filtering or offline refinement")
    if (candidate_cache or comparison_video or offline_config is not None) and not offline_refine:
        raise ValueError("Offline options require --offline-refine")
    if offline_refine:
        if tracker_config is not None:
            raise ValueError("Use offline_config for offline refinement")
        from .offline_demo import run_offline_demo
        return run_offline_demo(video_path, weights, output_dir, threshold, device,
                                offline_config, candidate_cache, comparison_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "shuttle_demo.mp4"
    output_data = output_dir / "shuttle_tracking.jsonl"
    metadata_path = output_dir / "shuttle_metadata.json"
    if output_video.exists() or output_data.exists() or metadata_path.exists():
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
        tracker = TemporalTracker(width, height, fps, tracker_config) if temporal_filter else None
        metadata = {"video": str(Path(video_path).resolve()), "width": width, "height": height,
                    "fps": fps, "frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "frame_origin": 0, "coordinate_system": "original_pixels",
                    "temporal_filter": temporal_filter}
        if tracker:
            from dataclasses import asdict
            metadata["tracker_config"] = asdict(tracker.config)
        metadata_path.write_text(json.dumps(metadata, indent=2))
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
                if tracker:
                    candidates = detector.predict_candidates(previous, current, following if has_next else current)
                    raw = {k: candidates[0][k] for k in ("x", "y", "score")} if candidates else None
                    detection = tracker.update(candidates, frame_number)
                else:
                    detection = detector.predict(previous, current, following if has_next else current)
                status = "detected" if detection is not None else "missing"
                row = {"frame": frame_number, "time_seconds": frame_number / fps,
                       "status": status, "x": None, "y": None, "score": None}
                if tracker:
                    row.update(raw_detection=raw, candidates=candidates, tracking_reason=tracker.reason)
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--temporal-filter", action="store_true")
    mode.add_argument("--offline-refine", action="store_true")
    parser.add_argument("--offline-config", type=Path, help="JSON overrides for OfflineConfig defaults")
    parser.add_argument("--candidate-cache", type=Path,
                        help="Offline only: reuse all candidates from JSONL with metadata; skips model inference")
    parser.add_argument("--comparison-video", action="store_true",
                        help="Offline only: also render original/temporal/offline panels")
    parser.add_argument("--tracker-config", type=Path, help="JSON overrides for TrackerConfig defaults")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    config = TrackerConfig(**json.loads(args.tracker_config.read_text())) if args.tracker_config else None
    from .offline_tracker import OfflineConfig
    offline_config = OfflineConfig(**json.loads(args.offline_config.read_text())) if args.offline_config else None
    if (args.candidate_cache or args.comparison_video or offline_config is not None) and not args.offline_refine:
        parser.error("Offline options require --offline-refine")
    if args.offline_refine and config is not None:
        parser.error("--tracker-config applies to temporal mode; use --offline-config")
    run_demo(args.video, args.weights, args.output, args.threshold, args.device,
             args.temporal_filter, config, args.offline_refine, offline_config,
             args.candidate_cache, args.comparison_video)


if __name__ == "__main__":
    main()
