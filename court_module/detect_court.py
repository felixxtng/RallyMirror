"""Run from RallyMirror: python court_module/detect_court.py path/to/video.mp4."""
import argparse
from pathlib import Path
import sys

# Support the separate court_module/src and the project's existing src namespace.
MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT.parent))
sys.path.insert(0, str(MODULE_ROOT))

import cv2
import numpy as np

from src.court.calibration import save_calibration
from src.court.detector import detect_court
from src.court.draw import overlay_court


def main():
    parser = argparse.ArgumentParser(description="Automatically fit a green badminton court on one video frame")
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=MODULE_ROOT / "outputs")
    parser.add_argument("--min-score", type=float, default=0.58)
    parser.add_argument("--overwrite", action="store_true", help="replace results for this clip/frame")
    args = parser.parse_args()
    if args.frame < 0 or not 0 <= args.min_score <= 1:
        parser.error("Frame must be nonnegative and min-score must be between 0 and 1")
    out = args.out_dir / args.video.stem / f"frame{args.frame:06d}"
    if out.exists() and not args.overwrite:
        parser.error(f"{out} already exists; choose another frame or use --overwrite")
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened():
            parser.error(f"Cannot open {args.video}")
        video.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = video.read()
    finally:
        video.release()
    if not ok:
        parser.error(f"Cannot read frame {args.frame}")
    out.mkdir(parents=True, exist_ok=True)
    # Remove an old calibration when explicitly rerunning; failed fits must not
    # leave an earlier successful calibration pretending to be current.
    for name in ("calibration.json", "overlay.jpg", "topdown.jpg",
                 "region.png", "white_mask.png", "candidates.png"):
        (out / name).unlink(missing_ok=True)
    def write_image(name, image):
        if not cv2.imwrite(str(out / name), image):
            raise OSError(f"Cannot save {out / name}")
    write_image("source.jpg", frame)
    try:
        result, diagnostics, error = detect_court(frame, args.min_score)
    except ValueError as exc:
        parser.exit(1, f"Detection failed: {exc}\nSource saved in {out}\n")
    for name, image in diagnostics.items():
        write_image(f"{name}.png", image)
    if result is None:
        parser.exit(1, f"Detection failed: {error}\nDiagnostics saved in {out}\n")
    result.update(video=str(args.video.resolve()), frame_index=args.frame)
    save_calibration(out / "calibration.json", result)
    H = np.array(result["H_image_to_court"])
    write_image("overlay.jpg", overlay_court(frame.copy(), H))
    # Top-down metres -> pixels, near end at the bottom, with a small border.
    transform = np.array([[50., 0, 25], [0, -50., 695], [0, 0, 1]])
    write_image("topdown.jpg", cv2.warpPerspective(frame, transform @ H, (355, 720)))
    print(f"Saved {out}")
    print(f"Line support: {result['stats']['line_support_score']:.3f} (not an accuracy percentage)")
    print("Open overlay.jpg: green court lines should align with the painted white lines.")


if __name__ == "__main__":
    main()
