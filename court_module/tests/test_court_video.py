"""Actual video export, per-frame inference and rejected-fit behaviour."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.court.learning_video import predict_video


@unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "Optional video encoder not installed")
class CourtVideoTests(unittest.TestCase):
    def test_video_contains_all_frames_audio_and_does_not_reuse_rejected_fit(self):
        from imageio_ffmpeg import get_ffmpeg_exe
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=blue:s=160x96:r=5:d=1", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=0.2", "-c:v", "libx264", "-c:a", "aac",
                            str(source)], check=True, capture_output=True)
            before = source.read_bytes()
            args = SimpleNamespace(checkpoint=root / "own.pt", video=source,
                                   output=root / "annotated", device="cpu")
            model = unittest.mock.Mock(return_value=torch.zeros((1, 12, 16, 16)))
            with patch("src.court.learning_video.load_model", return_value=(model, {"width": 64, "height": 64})), \
                    patch("src.court.learning_video.fit_predictions", side_effect=[(np.eye(3), 10)] + [None] * 4):
                result = predict_video(args)
            self.assertEqual(model.call_count, 5)
            self.assertEqual(result["frames"], 5)
            self.assertEqual(result["court_fits_accepted"], 1)
            self.assertEqual(result["court_fits_rejected"], 4)
            frames = [json.loads(line) for line in (args.output / "frames.jsonl").read_text().splitlines()]
            self.assertEqual([f["court_fit_accepted"] for f in frames], [True] + [False] * 4)
            output = args.output / "prediction.mp4"
            cap = cv2.VideoCapture(str(output))
            decoded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self.assertEqual(frame.shape[:2], (96, 160))
                decoded += 1
            cap.release()
            self.assertEqual(decoded, 5)
            subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-i", str(output),
                            "-map", "0:a:0", "-f", "null", "-"], check=True, capture_output=True)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "Output exists"):
                predict_video(args)

            # Optional source-audio mapping must also work on silent clips.
            silent = root / "silent.mp4"
            subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-i", str(source),
                            "-c:v", "copy", "-an", str(silent)], check=True, capture_output=True)
            args.video, args.output = silent, root / "silent_annotated"
            with patch("src.court.learning_video.load_model", return_value=(model, {"width": 64, "height": 64})), \
                    patch("src.court.learning_video.fit_predictions", return_value=None):
                self.assertEqual(predict_video(args)["frames"], 5)


if __name__ == "__main__":
    unittest.main()
