"""Use a trained CourtNet: image/video -> court points -> calibration JSON.

    python -m src.court.predict data/court_labelling/venueC_01.jpg --checkpoint outputs/court_net.pt
    python -m src.court.predict data/sample/clip.mp4 --frames 15        # FIXED camera only

Writes calibration_pred/<name>.json (same format as the click tool, so
src.court.demo can use it) and an overlay image in outputs/court_pred/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.court.augment import resize_matrix, warp_image
from src.court.calibrate import grab_frame
from src.court.calibration import save_calibration
from src.court.draw import overlay_court
from src.court.fit import calibration_from_points
from src.court.homography import apply_homography, invert
from src.court.net import INPUT_HEIGHT, INPUT_WIDTH, decode_heatmaps, load_checkpoint


def pick_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def predict_points(model, image_bgr, device="cpu"):
    """Returns (points (K, 2) in ORIGINAL image pixels, confidence (K,))."""
    height, width = image_bgr.shape[:2]
    M = resize_matrix(width, height, INPUT_WIDTH, INPUT_HEIGHT)
    rgb = cv2.cvtColor(warp_image(image_bgr, M, INPUT_WIDTH, INPUT_HEIGHT), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        points_in, conf = decode_heatmaps(model(tensor))
    return apply_homography(invert(M), points_in[0]), conf[0]


def draw_prediction(image, calib, names, points, conf, conf_thresh=0.3):
    out = overlay_court(image.copy(), np.array(calib["H_image_to_court"]), (0, 0, 255), 2)
    for name, (x, y), c in zip(names, points, conf):
        if c >= conf_thresh:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(out, name, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Predict a court calibration with the trained model")
    parser.add_argument("media", help="image or video")
    parser.add_argument("--checkpoint", default="outputs/court_net.pt")
    parser.add_argument("--frames", type=int, default=1,
                        help="videos: number of frames to sample; the per-point median is used "
                             "(only valid if the camera does not move)")
    parser.add_argument("--conf", type=float, default=0.3, help="minimum point confidence")
    parser.add_argument("--out-dir", default="calibration_pred")
    parser.add_argument("--overlay-dir", default="outputs/court_pred")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, names = load_checkpoint(args.checkpoint, device)
    media = Path(args.media)
    is_video = media.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if is_video:
        cap = cv2.VideoCapture(str(media))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_ids = [int(i) for i in np.linspace(0, max(total - 1, 0), max(args.frames, 1))]
    else:
        frame_ids = [0]

    all_points, all_conf, first = [], [], None
    for frame_id in dict.fromkeys(frame_ids):
        frame = grab_frame(media, frame_id)
        first = frame if first is None else first
        points, conf = predict_points(model, frame, device)
        all_points.append(points)
        all_conf.append(conf)
    all_points, all_conf = np.array(all_points), np.array(all_conf)
    points = np.zeros((len(names), 2))
    conf = np.zeros(len(names))
    for k in range(len(names)):
        good = all_conf[:, k] >= args.conf
        if good.any():
            points[k], conf[k] = np.median(all_points[good, k], axis=0), np.median(all_conf[good, k])

    height, width = first.shape[:2]
    info = {}
    calib = calibration_from_points(names, points, conf, (width, height), args.conf,
                                    video=str(media), info=info)
    if calib is None:
        raise SystemExit(f"Could not fit a trustworthy court ({info.get('reason')}). "
                         "Use the manual click tool for this one.")
    print("Stats:", calib["stats"], "| points used:", len(calib["points"]), "of", calib["n_detected"], "detected")
    for warning in calib["warnings"]:
        print("WARNING:", warning)
    out_path = Path(args.out_dir) / f"{media.stem}.json"
    save_calibration(out_path, calib)
    overlay_path = Path(args.overlay_dir) / f"{media.stem}.jpg"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), draw_prediction(first, calib, names, points, conf, args.conf))
    print(f"Saved {out_path}\nSaved {overlay_path}  (red lines should sit on the real lines)")


if __name__ == "__main__":
    main()
