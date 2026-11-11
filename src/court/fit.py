"""Turn detected court points into a calibration, rejecting fits that cannot be trusted.

A wrong court is worse than no court (it silently corrupts every player position), so
this refuses to return a calibration unless the detections are consistent and plausible.
numpy only, so it can be unit tested without the network.
"""
import numpy as np

from src.court.calibration import build_calibration
from src.court.court_model import COURT_LENGTH, COURT_POINTS, COURT_WIDTH
from src.court.homography import apply_homography, invert, ransac_homography


def near_end_is_lower(H_image_to_court):
    """True if the B0 end (the end nearest the camera, by our labelling rule) appears
    lower in the image than the far end. False means the fit is upside down."""
    ends = apply_homography(invert(H_image_to_court),
                            [[COURT_WIDTH / 2, 0.0], [COURT_WIDTH / 2, COURT_LENGTH]])
    return bool(np.all(np.isfinite(ends)) and ends[0][1] > ends[1][1])


def calibration_from_points(names, points, conf, image_size, conf_thresh=0.3, ransac_frac=0.006,
                            video="", frame_index=0, min_inliers=6, check_orientation=True,
                            info=None):
    """Confident points -> RANSAC homography -> calibration dict, or None if not trustworthy.

    ransac_frac: inlier distance as a fraction of the image diagonal (0.006 = ~9 px at 720p).
    min_inliers: fewer consistent points than this and the fit is rejected.
    info: optional dict that receives "reason" (why it was rejected) and "inliers".
    """
    info = info if info is not None else {}
    keep = [i for i in range(len(names)) if conf[i] >= conf_thresh]
    if len(keep) < 4:
        info["reason"] = f"only {len(keep)} confident points"
        return None
    diagonal = float(np.hypot(*image_size))
    court = np.array([COURT_POINTS[names[i]] for i in keep])
    H, inliers = ransac_homography(points[keep], court, ransac_frac * diagonal)
    if H is None:
        info["reason"] = "no consistent set of points"
        return None
    info["inliers"] = int(inliers.sum())
    if info["inliers"] < min_inliers:
        info["reason"] = f"only {info['inliers']} consistent points (need {min_inliers})"
        return None
    if check_orientation and not near_end_is_lower(H):
        info["reason"] = "court upside down (B0 end is not the end nearest the camera)"
        return None
    chosen = [keep[j] for j in np.flatnonzero(inliers)]
    calib = build_calibration([names[i] for i in chosen], points[chosen], video, frame_index, image_size)
    calib["source"] = "model"
    calib["n_detected"] = len(keep)
    return calib
