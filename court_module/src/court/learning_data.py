"""Manually annotated court images; coordinates always refer to saved originals."""
import json
import re
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .court_model import DEFAULT_CALIBRATION_POINTS, POINT_DESCRIPTIONS

LEGACY_POINT_NAMES = tuple(DEFAULT_CALIBRATION_POINTS)
POINT_NAMES = LEGACY_POINT_NAMES + ("NL-TOP", "NR-TOP", "C-SS0", "C-SS1")
GROUND_POINT_NAMES = tuple(n for n in POINT_NAMES if n not in ("NL-TOP", "NR-TOP"))
LEARNING_DESCRIPTIONS = {**POINT_DESCRIPTIONS,
                         "NL-TOP": "left end of the TOP net edge at the post (not the floor)",
                         "NR-TOP": "right end of the TOP net edge at the post (not the floor)"}
ARCHITECTURE = "court_keypoints_12_v2"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def load_manifest(root):
    path = Path(root) / "annotations.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") == 1:
        raise ValueError("Eight-point labels need upgrading: run court_learning.py upgrade-labels")
    if data.get("version") != 2 or data.get("point_names") != list(POINT_NAMES):
        raise ValueError("Unsupported annotation format or point order")
    return data


def save_manifest(root, data):
    path = Path(root) / "annotations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def blank_points():
    return {name: {"status": "unlabelled", "xy": None} for name in POINT_NAMES}


def upgrade_manifest(root):
    """Back up exact original bytes before adding four unlabelled points."""
    root = Path(root)
    path = root / "annotations.json"
    original = path.read_bytes()
    data = json.loads(original)
    if data.get("version") == 2:
        load_manifest(root)
        print("Labels already use 12 points; no changes made.")
        return
    if data.get("version") != 1 or data.get("point_names") != list(LEGACY_POINT_NAMES):
        raise ValueError("Cannot upgrade an unknown annotation format")
    backup = root / "annotations.eight_points.backup.json"
    suffix = 1
    while backup.exists():
        backup = root / f"annotations.eight_points.backup.{suffix}.json"
        suffix += 1
    backup.write_bytes(original)
    for record in data["images"]:
        for name in POINT_NAMES[len(LEGACY_POINT_NAMES):]:
            record["points"][name] = {"status": "unlabelled", "xy": None}
        record.setdefault("court_id", f"{record['group']}_court")
        record.setdefault("view_id", f"{record['group']}_main")
    data.update(version=2, point_names=list(POINT_NAMES))
    save_manifest(root, data)
    print(f"Preserved existing labels; added four blank points per image. Backup: {backup}")


def audit_splits(records):
    """Keep recordings AND physical courts/views out of multiple splits."""
    for field in ("group", "court_id", "view_id"):
        assignment = {}
        for r in records:
            value = r.get(field)
            if not value:
                raise ValueError(f"{r['id']}: missing {field}")
            if value in assignment and assignment[value] != r["split"]:
                raise ValueError(f"{field} {value} occurs in multiple splits")
            assignment[value] = r["split"]
    views = {}
    for r in records:
        if r["view_id"] in views and views[r["view_id"]] != r["court_id"]:
            raise ValueError(f"View {r['view_id']} refers to different physical courts; use globally unique view IDs")
        views[r["view_id"]] = r["court_id"]


def complete(record):
    return all(record["points"][n]["status"] in ("visible", "hidden") for n in POINT_NAMES)


def prepare(source, root, group, split, count, court_id=None, view_id=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", group):
        raise ValueError("Group must contain only letters, numbers, underscores or hyphens")
    if split not in ("train", "val", "test") or count <= 0:
        raise ValueError("Choose train/val/test and a positive image count")
    source, root = Path(source), Path(root)
    files = sorted(p for p in source.iterdir() if p.suffix.lower() == ".mp4") if source.is_dir() else [source]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("No input video files found")
    if count < len(files):
        raise ValueError(f"Use --count at least {len(files)} to sample every input video")
    data = load_manifest(root) if (root / "annotations.json").exists() else {
        "version": 2, "point_names": list(POINT_NAMES), "images": []}
    court_id = court_id or f"{group}_court"
    view_id = view_id or f"{group}_main"
    audit_splits(data["images"] + [{"id": "new_source", "group": group, "court_id": court_id,
                                    "view_id": view_id, "split": split}])
    if any(r["group"] == group and r["split"] != split for r in data["images"]):
        raise ValueError(f"Group {group} already belongs to another split; keep a whole match together")
    indexed = {r["id"]: r for r in data["images"]}
    added = 0
    for number, path in enumerate(files):
        video = cv2.VideoCapture(str(path))
        try:
            if not video.isOpened():
                raise ValueError(f"Cannot open {path}")
            frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            n = count // len(files) + int(number < count % len(files))
            if frames < n:
                raise ValueError(f"{path} has fewer than {n} frames")
            for index in np.linspace(0, frames - 1, n).round().astype(int):
                image_id = f"{group}/{path.stem}_frame{index:06d}"
                if image_id in indexed:
                    if indexed[image_id]["video"] != str(path.resolve()):
                        raise ValueError(f"Video name collision: {image_id}")
                    continue
                video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"Cannot read frame {index} in {path}")
                relative = f"images/{image_id}.jpg"
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    raise ValueError(f"Unregistered image exists: {output}; choose a different group name")
                if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Cannot write {output}")
                record = {"id": image_id, "image": relative, "video": str(path.resolve()),
                          "frame": int(index), "width": frame.shape[1], "height": frame.shape[0],
                          "group": group, "court_id": court_id, "view_id": view_id,
                          "split": split, "unusable": False, "points": blank_points()}
                data["images"].append(record)
                indexed[image_id] = record
                save_manifest(root, data)
                added += 1
                print(f"Prepared {image_id}", flush=True)
        finally:
            video.release()
    print(f"Added {added} images. Total: {len(data['images'])}. Labels: {root / 'annotations.json'}")


def import_sources(csv_path, root):
    """Import hand-selected photos or individual video frames from a source roster.

    Rows identify recordings, physical courts and camera views explicitly. IDs
    record the user's provenance; they do not automatically verify diversity.
    """
    csv_path, root = Path(csv_path), Path(root)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"source", "group", "court_id", "view_id", "split", "frame"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV needs columns: {', '.join(sorted(required))}")
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    if not rows:
        raise ValueError("Source CSV is empty")
    data = load_manifest(root) if (root / "annotations.json").exists() else {
        "version": 2, "point_names": list(POINT_NAMES), "images": []}
    # Check the complete roster before making changes.
    for number, row in enumerate(rows, 2):
        for field in ("group", "court_id", "view_id"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", row[field]):
                raise ValueError(f"CSV row {number}: {field} needs a valid nonempty ID")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError(f"CSV row {number}: invalid split")
        source = Path(row["source"])
        row["path"] = source.resolve() if source.is_absolute() else (csv_path.parent / source).resolve()
        if not row["path"].is_file():
            raise ValueError(f"CSV row {number}: file not found: {row['path']}")
        is_video = row["path"].suffix.lower() == ".mp4"
        if is_video and (not row["frame"].isdigit()):
            raise ValueError(f"CSV row {number}: MP4 source requires a nonnegative frame number")
        if not is_video and row["path"].suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            raise ValueError(f"CSV row {number}: use an image file or MP4")
        row["frame_index"] = int(row["frame"]) if is_video else None
        if not is_video and row["frame"]:
            raise ValueError(f"CSV row {number}: leave frame blank for an image")
    audit_splits(data["images"] + [{"id": r["source"], **{k: r[k] for k in ("group", "court_id", "view_id", "split")}} for r in rows])
    known = {(r.get("source", r.get("video")), r.get("frame")): r for r in data["images"]}
    hashes = {r["image_hash"]: r for r in data["images"] if "image_hash" in r}
    added = 0
    for row in rows:
        key = (str(row["path"]), row["frame_index"])
        if key in known:
            previous = known[key]
            if any(previous[k] != row[k] for k in ("group", "court_id", "view_id", "split")):
                raise ValueError(f"Source already imported with different identity/split: {row['source']}")
            continue
        if row["frame_index"] is None:
            image = cv2.imread(str(row["path"]))
        else:
            video = cv2.VideoCapture(str(row["path"]))
            try:
                video.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
                ok, image = video.read()
                if not ok:
                    image = None
            finally:
                video.release()
        if image is None:
            raise ValueError(f"Cannot read {row['source']}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Exact duplicate image: {row['source']} matches {hashes[digest]['id']}")
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", row["path"].stem)
        token = hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest()[:10]
        image_id = f"{row['group']}/{stem}_{token}"
        relative = f"images/{image_id}.jpg"
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ValueError(f"Unregistered image exists: {output}")
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Cannot write {output}")
        record = {"id": image_id, "image": relative, "source": key[0], "frame": key[1],
                  "width": image.shape[1], "height": image.shape[0], "image_hash": digest,
                  **{k: row[k] for k in ("group", "court_id", "view_id", "split")},
                  "unusable": False, "points": blank_points()}
        data["images"].append(record)
        known[key], hashes[digest] = record, record
        save_manifest(root, data)
        added += 1
        print(f"Imported {image_id} ({row['court_id']}, {row['view_id']})", flush=True)
    print(f"Added {added} images. Repeat imports preserve labels.")


def sync_images(root):
    """Register photos dropped into images/ without moving or relabelling them.

    Automatic IDs identify files, not verified courts. This shortcut is for a
    training collection only; recording/court identities must be reviewed before
    making held-out splits. Validate the whole batch before saving any changes.
    """
    root = Path(root).resolve()
    image_dir = root / "images"
    if not image_dir.is_dir():
        raise ValueError(f"Put your court JPG/PNG photos in {image_dir} first")
    data = load_manifest(root) if (root / "annotations.json").exists() else {
        "version": 2, "point_names": list(POINT_NAMES), "images": []}
    registered, hashes = set(), {}
    for row in data["images"]:
        path = (root / row["image"]).resolve()
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (row["height"], row["width"]):
            raise ValueError(f"Missing or wrong-size registered image: {row['image']}")
        registered.add(path)
        hashes[hashlib.sha256(image.tobytes()).hexdigest()] = row["image"]
    pending, duplicates = [], []
    ids = {r["id"] for r in data["images"]}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(image_dir.resolve()):
            raise ValueError(f"Image must be inside {image_dir}: {path}")
        if resolved in registered:
            continue
        relative = path.relative_to(root).as_posix()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {relative}; fix or remove it, then sync again")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in hashes:
            duplicates.append((relative, hashes[digest]))
            continue
        token = hashlib.sha256(relative.encode()).hexdigest()[:16]
        image_id = f"added/{token}"
        if image_id in ids:
            raise ValueError(f"Image ID collision: {relative}")
        pending.append({"id": image_id, "image": relative,
                        "source": str(resolved), "frame": None,
                        "width": image.shape[1], "height": image.shape[0],
                        "image_hash": digest, "group": "unverified_training_collection",
                        "court_id": f"unverified_photo_{token}",
                        "view_id": f"unverified_photo_{token}",
                        "court_identity_status": "unverified_photo",
                        "source_group_status": "collection_id_not_verified_match",
                        "split": "train", "unusable": False, "points": blank_points()})
        ids.add(image_id)
        hashes[digest] = relative
    audit_splits(data["images"] + pending)
    if pending or not (root / "annotations.json").exists():
        data["images"].extend(pending)
        save_manifest(root, data)
    for duplicate, original in duplicates:
        print(f"Skipped exact duplicate: {duplicate} matches {original}")
    print(f"Added {len(pending)} images; {len(data['images'])} registered in total. Existing labels preserved.")
    print("New images are training data. Automatic IDs do not verify distinct courts.")
    return len(pending)


def representative_indices(records):
    """One usable image per fixed camera view; extra rallies do not add views."""
    by_view = {}
    for i, row in enumerate(records):
        view = row["view_id"]
        if view not in by_view or records[by_view[view]]["unusable"] and not row["unusable"]:
            by_view[view] = i
    return list(by_view.values())


def validate_record(record):
    for name in POINT_NAMES:
        point = record["points"][name]
        if point["status"] not in ("unlabelled", "visible", "hidden"):
            raise ValueError(f"{record['id']}: invalid status for {name}")
        if point["status"] == "visible":
            xy = np.asarray(point["xy"], dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or not (
                    0 <= xy[0] < record["width"] and 0 <= xy[1] < record["height"]):
                raise ValueError(f"{record['id']}: invalid coordinates for {name}")


def image_tensor(image, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255


def target_heatmaps(points, width, height, sigma=1.5):
    """Normalised Gaussian targets; hidden points are excluded from the loss."""
    yy, xx = np.mgrid[:height, :width]
    targets = np.zeros((len(POINT_NAMES), height, width), np.float32)
    mask = np.zeros(len(POINT_NAMES), np.float32)
    for i, point in enumerate(points):
        if np.isfinite(point).all():
            heatmap = np.exp(-((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / (2 * sigma ** 2))
            targets[i] = heatmap / max(heatmap.sum(), 1e-12)
            mask[i] = 1
    return torch.from_numpy(targets), torch.from_numpy(mask)


class CourtDataset(Dataset):
    def __init__(self, root, split, width=640, height=360, augment=False, one_per_view=False):
        self.root, self.width, self.height, self.augment = Path(root), width, height, augment
        if width < 32 or height < 32 or width % 8 or height % 8:
            raise ValueError("Input dimensions must be multiples of 8 and at least 32")
        data = load_manifest(root)
        self.records = []
        # Audit all split assignments, including unfinished images.
        audit_splits(data["images"])
        for r in data["images"]:
            validate_record(r)
            if r["split"] == split and not r["unusable"] and complete(r):
                if sum(r["points"][n]["status"] == "visible" for n in GROUND_POINT_NAMES) >= 4:
                    self.records.append(r)
        if one_per_view:
            self.records = [self.records[i] for i in representative_indices(self.records)]
        self.groups = {r["group"] for r in self.records}
        self.courts = {r["court_id"] for r in self.records}
        self.views = {r["view_id"] for r in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image"]))
        if image is None or image.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Missing or wrong-size image: {record['image']}")
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        points = np.full((len(POINT_NAMES), 2), np.nan, np.float32)
        for i, name in enumerate(POINT_NAMES):
            point = record["points"][name]
            if point["status"] == "visible":
                points[i] = np.array(point["xy"]) * [self.width / record["width"], self.height / record["height"]]
        if self.augment:
            # Transform the image AND every visible point together. Net-top
            # points are image annotations, so image-space augmentation is valid.
            angle = np.random.uniform(-3, 3)
            matrix = cv2.getRotationMatrix2D((self.width/2, self.height/2), angle, np.random.uniform(0.94, 1.06))
            matrix[:, 2] += [np.random.uniform(-0.03, 0.03)*self.width,
                             np.random.uniform(-0.03, 0.03)*self.height]
            image = cv2.warpAffine(image, matrix, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101)
            visible = np.isfinite(points).all(axis=1)
            points[visible] = np.c_[points[visible], np.ones(visible.sum())] @ matrix.T
            outside = (points[:, 0] < 0) | (points[:, 0] >= self.width) | (points[:, 1] < 0) | (points[:, 1] >= self.height)
            points[outside] = np.nan
            # Photometric augmentation does not alter point identities or coordinates.
            gain, offset = np.random.uniform(0.8, 1.2), np.random.uniform(-15, 15)
            image = np.clip(image.astype(float) * gain + offset, 0, 255).astype(np.uint8)
        targets, mask = target_heatmaps(points / 4, self.width // 4, self.height // 4)
        return image_tensor(image, self.width, self.height), targets, mask, torch.from_numpy(points)
