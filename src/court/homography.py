"""Image <-> court homography (numpy only, so it is easy to unit test).

A homography is only valid for points ON THE GROUND PLANE. Use feet, never the
shuttle or a player's hands/head.
"""
import numpy as np


def _normalise(points):
    centre = points.mean(axis=0)
    mean_dist = np.sqrt(((points - centre) ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-9:
        raise ValueError("All points are identical")
    s = np.sqrt(2) / mean_dist
    T = np.array([[s, 0, -s * centre[0]], [0, s, -s * centre[1]], [0, 0, 1]])
    homogeneous = np.c_[points, np.ones(len(points))] @ T.T
    return homogeneous[:, :2], T


def _check_not_collinear(points, name):
    centred = points - points.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[-1] < 1e-6 * max(sv[0], 1e-12):
        raise ValueError(f"{name} points are (almost) collinear")


def compute_homography(src, dst):
    """Least-squares homography H with dst ~ H @ src (normalised DLT, >= 4 points)."""
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError("src and dst need the same number of points")
    if len(src) < 4:
        raise ValueError("Need at least 4 point pairs")
    _check_not_collinear(src, "Source")
    _check_not_collinear(dst, "Destination")
    s, Ts = _normalise(src)
    d, Td = _normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_homography(H, points):
    """Map (n, 2) or (2,) points through H. Points at infinity become NaN."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = pts.reshape(-1, 2)
    mapped = np.c_[pts, np.ones(len(pts))] @ np.asarray(H).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = mapped[:, :2] / mapped[:, 2:3]
    out[np.abs(mapped[:, 2]) < 1e-12] = np.nan
    return out[0] if single else out


def invert(H):
    return np.linalg.inv(np.asarray(H, dtype=float))


def reprojection_error_px(H_image_to_court, image_points, court_points):
    """Per-point pixel distance after projecting court points back into the image."""
    back = apply_homography(invert(H_image_to_court), court_points)
    return np.linalg.norm(back - np.asarray(image_points, dtype=float), axis=1)


def leave_one_out_errors_m(image_points, court_points):
    """Honest accuracy check: fit without point i, then measure its error in metres.

    Needs >= 5 points. Returns one error (metres) per point.
    """
    image_points = np.asarray(image_points, dtype=float)
    court_points = np.asarray(court_points, dtype=float)
    n = len(image_points)
    if n < 5:
        raise ValueError("Leave-one-out needs at least 5 points")
    errors = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        H = compute_homography(image_points[keep], court_points[keep])
        errors.append(np.linalg.norm(apply_homography(H, image_points[i]) - court_points[i]))
    return np.array(errors)


def _triple_area(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2


def _subset_is_well_spread(points):
    """Reject 4-point subsets where any 3 points are (nearly) collinear."""
    from itertools import combinations
    return all(_triple_area(*trio) > 1e-3 for trio in combinations(points, 3))


def ransac_homography(image_points, court_points, thresh_px, max_subsets=800, seed=0):
    """Robust image->court homography from possibly wrong detections.

    Tries 4-point subsets, keeps the one whose homography reprojects the most points
    within `thresh_px` pixels, then refits on all of those inliers.
    Returns (H, inlier_mask) or (None, None) if no valid fit exists.
    """
    from itertools import combinations
    import random

    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    court_points = np.asarray(court_points, dtype=float).reshape(-1, 2)
    n = len(image_points)
    if n < 4:
        return None, None
    subsets = list(combinations(range(n), 4))
    if len(subsets) > max_subsets:
        subsets = random.Random(seed).sample(subsets, max_subsets)

    best = (-1, np.inf, None)          # (inliers, mean error, mask)
    for subset in subsets:
        idx = list(subset)
        if not _subset_is_well_spread(court_points[idx]):
            continue
        try:
            H = compute_homography(image_points[idx], court_points[idx])
            errors = reprojection_error_px(H, image_points, court_points)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(errors)):
            continue
        mask = errors <= thresh_px
        count, mean_err = int(mask.sum()), float(errors[mask].mean()) if mask.any() else np.inf
        if count > best[0] or (count == best[0] and mean_err < best[1]):
            best = (count, mean_err, mask)
    if best[2] is None or best[0] < 4:
        return None, None
    mask = best[2]
    try:
        H = compute_homography(image_points[mask], court_points[mask])
    except (ValueError, np.linalg.LinAlgError):
        return None, None
    return H, mask
