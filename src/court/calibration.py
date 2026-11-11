"""Build, save and load a court calibration (image -> court homography)."""
import json
from pathlib import Path

import numpy as np

from src.court.court_model import COURT_POINTS
from src.court.homography import (compute_homography, leave_one_out_errors_m,
                                  reprojection_error_px)

# Rough "is this calibration trustworthy?" thresholds used for warnings only.
WARN_REPROJ_PX = 6.0
WARN_LOO_M = 0.35


def build_calibration(names, image_points, video="", frame_index=0, image_size=None):
    """names: court point names; image_points: matching (x, y) pixel clicks."""
    unknown = [n for n in names if n not in COURT_POINTS]
    if unknown:
        raise ValueError(f"Unknown court point names: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("Each court point can only be used once")
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    court_points = np.array([COURT_POINTS[n] for n in names], dtype=float)
    H = compute_homography(image_points, court_points)

    reproj = reprojection_error_px(H, image_points, court_points)
    stats = {"n_points": len(names),
             "mean_reproj_px": float(reproj.mean()),
             "max_reproj_px": float(reproj.max())}
    warnings = []
    if len(names) >= 5:
        loo = leave_one_out_errors_m(image_points, court_points)
        stats["loo_mean_m"] = float(loo.mean())
        stats["loo_max_m"] = float(loo.max())
        if loo.mean() > WARN_LOO_M:
            warnings.append(f"Leave-one-out error is {loo.mean():.2f} m: re-click more carefully")
    else:
        warnings.append("Only 4 points: no way to estimate accuracy, click 5+ if you can")
    if reproj.max() > WARN_REPROJ_PX:
        warnings.append(f"Max reprojection error is {reproj.max():.1f}px: check for a mis-click")

    return {"video": str(video), "frame_index": int(frame_index),
            "image_size": list(image_size) if image_size else None,
            "points": [{"name": n, "image_xy": [float(p[0]), float(p[1])],
                        "court_xy": list(COURT_POINTS[n])}
                       for n, p in zip(names, image_points)],
            "H_image_to_court": H.tolist(), "stats": stats, "warnings": warnings}


def save_calibration(path, calibration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["H"] = np.array(data["H_image_to_court"], dtype=float)
    return data
