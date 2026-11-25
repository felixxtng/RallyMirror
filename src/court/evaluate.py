"""Test the trained model on held-out images and compare with your manual calibrations.

    python -m src.court.evaluate --dataset dataset/court --checkpoint checkpoints/court_net.pt

For each image of the chosen split it reports
  * point error: pixels between predicted and clicked points (as % of image width),
  * court error: metres of error on floor positions after fitting a homography to the
    PREDICTED points, compared with your manual calibration (the ground truth),
  * a saved overlay: green = your manual calibration, red = the model's.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.court.court_data import load_labels
from src.court.draw import overlay_court
from src.court.homography import apply_homography
from src.court.metrics import angle_summary, floor_grid_in_image, half_errors
from src.court.net import load_checkpoint
from src.court.predict import calibration_from_points, pick_device, predict_points


def _mean_of(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate the court model")
    parser.add_argument("--dataset", default="dataset/court")
    parser.add_argument("--checkpoint", default="checkpoints/court_net.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--success-m", type=float, default=0.3,
                        help="an image counts as a success if mean court error is below this (metres)")
    parser.add_argument("--out", default="outputs/court_eval")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, names = load_checkpoint(args.checkpoint, device)
    _, entries = load_labels(args.dataset)
    entries = [e for e in entries if e["split"] == args.split]
    if not entries:
        raise SystemExit(f"No images in split '{args.split}'")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in entries:
        image = cv2.imread(str(Path(args.dataset) / "images" / entry["file"]))
        points, conf = predict_points(model, image, device)
        truth = np.array([entry["points"][n] for n in names])
        visible = np.array([entry["visible"][n] for n in names])
        point_err = np.linalg.norm(points[visible] - truth[visible], axis=1) / entry["size"][0] * 100
        found = float((conf[visible] >= args.conf).mean()) if visible.any() else 0.0
        H_true = np.array(entry["H_image_to_court"])
        info = {}
        calib = calibration_from_points(names, points, conf, tuple(entry["size"]), args.conf, info=info)
        court_mean = court_max = near_mean = far_mean = None
        overlay = overlay_court(image.copy(), H_true, (0, 255, 0), 2)
        if calib is not None:
            H_pred = np.array(calib["H_image_to_court"])
            pixels, court = floor_grid_in_image(H_true, entry["size"])
            errors = np.linalg.norm(apply_homography(H_pred, pixels) - court, axis=1)
            court_mean, court_max = float(errors.mean()), float(errors.max())
            halves = half_errors(errors, court)
            near_mean, far_mean = halves["near"][0], halves["far"][0]
            overlay = overlay_court(overlay, H_pred, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{entry['name']}.jpg"), overlay)
        rows.append({"image": entry["name"], "angle": entry.get("angle", "standard"), "point_error_pct_width": float(point_err.mean()) if len(point_err) else None,
                     "points_found": found, "court_error_mean_m": court_mean, "court_error_max_m": court_max, "court_error_near_m": near_mean, "court_error_far_m": far_mean,
                     "fitted": calib is not None})
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        text = (f"REJECTED: {info.get('reason')}" if calib is None else
                f"court error mean {court_mean:.2f} m (near {fmt(near_mean)}, far {fmt(far_mean)}), "
                f"max {court_max:.2f} m [{len(calib['points'])} points used]")
        point_text = "n/a" if not len(point_err) else f"{point_err.mean():.2f}% of width"
        print(f"{entry['name']:20s} point error {point_text} | found {found:4.0%} | {text}")

    fitted = [r for r in rows if r["fitted"]]
    summary = {"split": args.split, "images": len(rows), "fitted": len(fitted),
               "success_rate": sum(r["court_error_mean_m"] <= args.success_m for r in fitted) / len(rows),
               "mean_court_error_m": float(np.mean([r["court_error_mean_m"] for r in fitted])) if fitted else None,
               "median_court_error_m": float(np.median([r["court_error_mean_m"] for r in fitted])) if fitted else None,
               "mean_near_error_m": _mean_of(fitted, "court_error_near_m"),
               "mean_far_error_m": _mean_of(fitted, "court_error_far_m"),
               "success_threshold_m": args.success_m, "per_image": rows}
    summary["by_angle"] = angle_summary(rows, args.success_m)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{summary['split']}: {summary['images']} images, court fitted on {summary['fitted']}, "
          f"success (< {args.success_m} m) {summary['success_rate']:.0%}, "
          f"mean court error {summary['mean_court_error_m']} m, median {summary['median_court_error_m']} m "
          f"(near half {summary['mean_near_error_m']}, far half {summary['mean_far_error_m']})")
    print("\nBy camera angle (name files venueX_<angle>_NN to tag an angle):")
    for angle, g in summary["by_angle"].items():
        mean = "n/a" if g["mean_error_m"] is None else f"{g['mean_error_m']:.2f} m"
        print(f"  {angle:10s} images {g['images']} | fitted {g['fitted']} | under {args.success_m} m: "
              f"{g['success']} | rejected {g['rejected']} | mean error of fitted {mean}")
    print(f"Overlays saved in {out_dir} (green = your manual label, red = model)")


if __name__ == "__main__":
    main()
