"""Turn your saved calibrations into a labelled dataset for training.

    python -m src.court.export_dataset --calibration-dir data/calibration --out data/dataset/court

For every calibration/<name>.json it:
  1. loads the labelled image (or video frame),
  2. projects ALL court points in --points into that image using the saved homography
     (so points you did not click are filled in), marking points outside the frame as
     not visible,
  3. saves the image and writes data/dataset/court/labels.json.

Name your files "<venue>_<number>" (venueA_01, venueB_03 ...). The text before the first
underscore is the venue, and the train/val/test split is made BY VENUE so the test
score is honest (a model must never be tested on a venue it trained on).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.court.calibrate import grab_frame
from src.court.calibration import load_calibration
from src.court.court_model import COURT_POINTS
from src.court.homography import invert

DEFAULT_POINTS = ["DL-B0", "DR-B0", "DL-B1", "DR-B1", "DL-SS0", "DR-SS0",
                  "DL-SS1", "DR-SS1", "C-B0", "C-SS0", "C-SS1", "C-B1"]


def venue_of(name):
    """Venue = text before the first underscore, ignoring upper/lower case
    (so venueF_01 and VenueF_02 are the same venue)."""
    return name.split("_")[0].lower()


def angle_of(name):
    """Camera-angle tag: the middle part of a three-part name, else "standard".
    venueF_low_04 -> "low", venueC_phone_01 -> "phone", venueA_01 -> "standard"."""
    parts = name.split("_")
    return parts[1].lower() if len(parts) >= 3 else "standard"


def assign_splits(venues, val_venues=None, test_venues=None):
    """Map each venue to 'train' / 'val' / 'test'. Returns (mapping, warnings)."""
    venues = sorted({v.lower() for v in venues})
    warnings = []
    val_venues = {v.lower() for v in (val_venues or [])}
    test_venues = {v.lower() for v in (test_venues or [])}
    unknown = (val_venues | test_venues) - set(venues)
    if unknown:
        raise ValueError(f"Unknown venues: {sorted(unknown)}. Found: {venues}")
    if val_venues & test_venues and len(venues) > 2:
        raise ValueError("A venue cannot be both validation and test")
    if not val_venues and not test_venues:
        if len(venues) >= 3:
            test_venues = {venues[-1]}
            val_venues = {venues[-2]}
        elif len(venues) == 2:
            test_venues = val_venues = {venues[-1]}
            warnings.append("Only 2 venues: the second venue is used for BOTH validation and "
                            "test. Fine for a first run, but label more venues for real results.")
        else:
            warnings.append("Only 1 venue: everything is 'train'. You cannot measure "
                            "generalisation until you label a second venue.")
    mapping = {}
    for venue in venues:
        if venue in test_venues:
            mapping[venue] = "test"
        elif venue in val_venues:
            mapping[venue] = "val"
        else:
            mapping[venue] = "train"
    if "train" not in mapping.values():
        raise ValueError("No venue left for training; adjust --val-venues/--test-venues")
    return mapping, warnings


def project_court_points(H_image_to_court, names, image_size):
    """Project named court points into the image. Returns {name: (xy, visible)}.

    A point counts as visible only if it lands inside the frame and in FRONT of the
    camera (a homography also maps points behind the camera to garbage positions).
    """
    H_inv = invert(H_image_to_court)
    width, height = image_size
    court = np.array([COURT_POINTS[n] for n in names], dtype=float)
    homogeneous = np.c_[court, np.ones(len(court))] @ H_inv.T
    w = homogeneous[:, 2]
    # the true court corners lie in front of the camera, so use the sign of the centre
    centre = np.array([*np.mean(court, axis=0), 1.0]) @ H_inv.T
    in_front = np.sign(w) == np.sign(centre[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = homogeneous[:, :2] / w[:, None]
    result = {}
    for name, point, front in zip(names, xy, in_front):
        inside = (front and np.all(np.isfinite(point))
                  and 0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1)
        result[name] = ([float(point[0]), float(point[1])] if np.all(np.isfinite(point)) else [0.0, 0.0],
                        bool(inside))
    return result


def find_media(video, media_dir):
    path = Path(str(video).replace("\\", "/"))
    if path.exists():
        return path
    if media_dir:
        matches = list(Path(media_dir).rglob(path.name))
        if matches:
            return matches[0]
    return None


def export(calibration_dir, out_dir, names, val_venues, test_venues, media_dir, extra_frames):
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(calibration_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"No calibration .json files in {calibration_dir}")

    entries, skipped = [], []
    for file in files:
        calib = load_calibration(file)
        media = find_media(calib["video"], media_dir)
        if media is None:
            skipped.append(f"{file.name}: cannot find {calib['video']} (use --media-dir)")
            continue
        is_video = media.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        frame_ids = [calib["frame_index"]]
        if is_video and extra_frames > 0:
            cap = cv2.VideoCapture(str(media))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total > 0:
                frame_ids += [int(i) for i in np.linspace(0, total - 1, extra_frames + 2)[1:-1]]
        for frame_id in dict.fromkeys(frame_ids):
            try:
                frame = grab_frame(media, frame_id)
            except ValueError as error:
                skipped.append(f"{file.name}: {error}")
                continue
            height, width = frame.shape[:2]
            if calib.get("image_size") and tuple(calib["image_size"]) != (width, height):
                skipped.append(f"{file.name}: size changed since labelling")
                continue
            stem = file.stem if frame_id == calib["frame_index"] else f"{file.stem}_f{frame_id}"
            image_name = f"{stem}.jpg"
            cv2.imwrite(str(out_dir / "images" / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            projected = project_court_points(calib["H"], names, (width, height))
            entries.append({
                "file": image_name, "name": stem, "venue": venue_of(file.stem),
                "size": [width, height], "angle": angle_of(file.stem),
                "points": {n: projected[n][0] for n in names},
                "visible": {n: projected[n][1] for n in names},
                "H_image_to_court": calib["H"].tolist(),
            })

    mapping, warnings = assign_splits([e["venue"] for e in entries], val_venues, test_venues)
    for entry in entries:
        entry["split"] = mapping[entry["venue"]]
    (out_dir / "labels.json").write_text(
        json.dumps({"point_names": names, "images": entries}, indent=1), encoding="utf-8")

    counts = {s: sum(e["split"] == s for e in entries) for s in ("train", "val", "test")}
    print(f"Exported {len(entries)} images -> {out_dir}")
    print("Venue split:", mapping, "| images per split:", counts)
    angles = {}
    for entry in entries:
        angles.setdefault(entry["angle"], {}).setdefault(entry["split"], 0)
        angles[entry["angle"]][entry["split"]] += 1
    print("Images per camera angle and split:", angles)
    avg_visible = np.mean([sum(e["visible"].values()) for e in entries])
    print(f"Average visible points per image: {avg_visible:.1f} of {len(names)}")
    for text in warnings + skipped:
        print("WARNING:", text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export calibrations to a training dataset")
    parser.add_argument("--calibration-dir", default="data/calibration")
    parser.add_argument("--out", default="data/dataset/court")
    parser.add_argument("--points", help="comma-separated point names (default: the 12-point list)")
    parser.add_argument("--val-venues", help="comma-separated venues used for validation")
    parser.add_argument("--test-venues", help="comma-separated venues used for testing")
    parser.add_argument("--media-dir", help="folder to search if a labelled file has moved")
    parser.add_argument("--extra-frames", type=int, default=0,
                        help="FIXED-CAMERA videos only: extra frames per clip using the same label")
    args = parser.parse_args()
    point_names = args.points.split(",") if args.points else DEFAULT_POINTS
    bad = [n for n in point_names if n not in COURT_POINTS]
    if bad:
        parser.error(f"unknown point names {bad}")
    split = lambda text: text.split(",") if text else None
    export(args.calibration_dir, args.out, point_names, split(args.val_venues),
           split(args.test_venues), args.media_dir, args.extra_frames)
