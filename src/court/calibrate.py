"""Click-to-calibrate tool: click known court points on one frame, save the homography.

    python -m src.court.calibrate data/sample/clip.mp4        # or a .jpg/.png image
    python -m src.court.calibrate data/sample/clip.mp4 --frame 30 --points DL-B0,DR-B0,DL-B1,DR-B1,DL-SS0
    python -m src.court.calibrate --list-points

Keys: click = place current point | mouse wheel or + / - = zoom | right-drag = pan | 0 = reset zoom |
      s = skip (hidden point) | u = undo | f = finish early (needs 4+ points) | Esc = abort.
Use a frame where the court is clearly visible and nobody stands on the lines.
For a fixed phone camera you only need to do this once per clip.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.court.calibration import build_calibration, save_calibration
from src.court.court_model import (COURT_POINTS, DEFAULT_CALIBRATION_POINTS,
                                   POINT_DESCRIPTIONS)
from src.court.draw import make_court_canvas, overlay_court, paste_inset
from src.court.zoom import ZoomView
from src.video.reader import open_video

WINDOW = "RallyMirror court calibration (see terminal for keys)"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def grab_frame(video_path, frame_index):
    """Return one frame: the image itself for image files, else frame N of a video."""
    if Path(video_path).suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(video_path))
        if image is None:
            raise ValueError(f"Could not read image: {video_path}")
        return image
    video = open_video(video_path)
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video.read()
    finally:
        video.release()
    if not success:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _label(img, text, y, scale):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)


def collect_clicks(frame, names):
    height, width = frame.shape[:2]
    base = min(1.0, 1280 / width, 720 / height)          # shrink big frames to fit the screen
    win_w, win_h = int(width * base), int(height * base)
    zoom = ZoomView(width, height, win_w, win_h)
    state = {"mouse": (win_w // 2, win_h // 2), "pan": None, "click": None}
    entries = []                                          # (name, (x, y) | None if skipped)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:                 # zoom about the last known cursor spot
            zoom.zoom_at(*state["mouse"], 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8)
            return
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = zoom.to_image(x, y)          # back to original image pixels
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["pan"] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            state["pan"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["pan"] is not None:
            zoom.pan_by_display(state["pan"][0] - x, state["pan"][1] - y)
            state["pan"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while len(entries) < len(names):
        name = names[len(entries)]
        clicked = [(n, p) for n, p in entries if p is not None]
        display = cv2.warpAffine(frame, zoom.matrix(), (win_w, win_h), flags=cv2.INTER_LINEAR)
        for n, (x, y) in clicked:
            dx, dy = zoom.to_display(x, y)
            if 0 <= dx < win_w and 0 <= dy < win_h:
                cv2.circle(display, (int(dx), int(dy)), 5, (255, 120, 0), -1)
                cv2.putText(display, n, (int(dx) + 8, int(dy) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 120, 0), 2, cv2.LINE_AA)
        paste_inset(display, make_court_canvas(highlight=name, marked=[n for n, _ in clicked]))
        _label(display, f"Click {name}: {POINT_DESCRIPTIONS[name]}", 28, 0.7)
        _label(display, f"{len(clicked)} placed | wheel or +/- zoom, right-drag pan, 0 reset | "
                        "s=skip u=undo f=finish Esc=abort", 56, 0.5)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF
        if state["click"] is not None:
            entries.append((name, state["click"]))
            state["click"] = None
        elif key in (ord("+"), ord("=")):
            zoom.zoom_at(*state["mouse"], 1.25)
        elif key in (ord("-"), ord("_")):
            zoom.zoom_at(*state["mouse"], 0.8)
        elif key == ord("0"):
            zoom.reset()
        elif key == ord("s"):
            entries.append((name, None))
        elif key == ord("u") and entries:
            entries.pop()
        elif key == ord("f") and len(clicked) >= 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration aborted.")
    return [(n, p) for n, p in entries if p is not None]


def review_and_confirm(frame, calibration):
    """Show the projected court on the frame; Enter saves, Esc discards."""
    h, w = frame.shape[:2]
    disp = min(1.0, 1280 / w, 720 / h)
    font = max(0.6, w / 1700)
    view = overlay_court(frame.copy(), np.array(calibration["H_image_to_court"]))
    stats = calibration["stats"]
    text = f"reproj mean {stats['mean_reproj_px']:.1f}px"
    if "loo_mean_m" in stats:
        text += f" | leave-one-out {stats['loo_mean_m']:.2f} m"
    _label(view, text, int(34 * font), font)
    _label(view, "Green lines should sit on the real lines.  Enter=save  Esc=discard",
           int(70 * font), font * 0.8)
    cv2.imshow(WINDOW, cv2.resize(view, None, fx=disp, fy=disp) if disp < 1 else view)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (10, 13):
            return True
        if key == 27:
            return False


def main():
    parser = argparse.ArgumentParser(description="Click court points to calibrate a clip")
    parser.add_argument("video", nargs="?", help="video or image file")
    parser.add_argument("--frame", type=int, default=0, help="frame to click on (videos only)")
    parser.add_argument("--out", help="default: calibration/<video name>.json")
    parser.add_argument("--points", help="comma-separated point names (see --list-points)")
    parser.add_argument("--list-points", action="store_true")
    args = parser.parse_args()

    if args.list_points:
        for name in COURT_POINTS:
            print(f"{name:8s} {tuple(round(v, 2) for v in COURT_POINTS[name])}  {POINT_DESCRIPTIONS[name]}")
        return
    if not args.video:
        parser.error("video is required")
    names = args.points.split(",") if args.points else DEFAULT_CALIBRATION_POINTS
    unknown = [n for n in names if n not in COURT_POINTS]
    if unknown:
        parser.error(f"unknown point names {unknown}; see --list-points")

    frame = grab_frame(args.video, args.frame)
    picked = collect_clicks(frame, names)
    if len(picked) < 4:
        cv2.destroyAllWindows()
        raise SystemExit("Need at least 4 points.")
    h, w = frame.shape[:2]
    calibration = build_calibration([n for n, _ in picked], [p for _, p in picked],
                                    args.video, args.frame, (w, h))
    print("Stats:", calibration["stats"])
    for warning in calibration["warnings"]:
        print("WARNING:", warning)
    save = review_and_confirm(frame, calibration)
    cv2.destroyAllWindows()
    if save:
        out = args.out or f"calibration/{Path(args.video).stem}.json"
        save_calibration(out, calibration)
        print(f"Saved {out}")
    else:
        print("Discarded.")


if __name__ == "__main__":
    main()
