"""Court-space error measures (numpy only, so they are easy to test)."""
import numpy as np

from src.court.court_model import COURT_LENGTH, COURT_WIDTH, NET_Y
from src.court.homography import invert


def floor_grid_in_image(H_image_to_court, size, nx=7, ny=15):
    """Court-floor points that fall inside the image under the ground-truth calibration.
    Returns (pixels, court_metres)."""
    xs, ys = np.meshgrid(np.linspace(0, COURT_WIDTH, nx), np.linspace(0, COURT_LENGTH, ny))
    court = np.c_[xs.ravel(), ys.ravel()]
    H_inv = invert(H_image_to_court)
    homogeneous = np.c_[court, np.ones(len(court))] @ H_inv.T
    centre = np.array([COURT_WIDTH / 2, COURT_LENGTH / 2, 1.0]) @ H_inv.T
    front = np.sign(homogeneous[:, 2]) == np.sign(centre[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    inside = (front & np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= size[0] - 1)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] <= size[1] - 1))
    return pixels[inside], court[inside]


def half_errors(errors, court):
    """Mean and max error (metres) for the near half (y < net, the B0 end) and the far half.
    A value is None if no floor points fall in that half."""
    errors, court = np.asarray(errors), np.asarray(court)
    near = court[:, 1] < NET_Y
    result = {}
    for label, mask in (("near", near), ("far", ~near)):
        result[label] = ((float(errors[mask].mean()), float(errors[mask].max()))
                         if mask.any() else (None, None))
    return result


def angle_summary(rows, success_m=0.3):
    """Group per-image results by camera angle.

    rows: dicts with "angle", "fitted" and "court_error_mean_m".
    Returns {angle: {images, fitted, rejected, success, mean_error_m}}; a rejected image
    counts as a failure and mean_error_m covers only the fitted images.
    """
    summary = {}
    for angle in sorted({r.get("angle", "standard") for r in rows}):
        group = [r for r in rows if r.get("angle", "standard") == angle]
        fitted = [r for r in group if r["fitted"]]
        errors = [r["court_error_mean_m"] for r in fitted]
        summary[angle] = {
            "images": len(group), "fitted": len(fitted), "rejected": len(group) - len(fitted),
            "success": sum(e <= success_m for e in errors),
            "mean_error_m": float(np.mean(errors)) if errors else None,
        }
    return summary
