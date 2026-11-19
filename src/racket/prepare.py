"""Register sources without extracting every frame or touching shuttle labels."""
import argparse
import hashlib
from pathlib import Path
import re

import cv2

from .schema import NAMES, file_hash, group_id, read_json, save_json
from .source import image_files


def register(source, name, group, split, root, fps=30, start=0, end=None):
    source, root = Path(source).resolve(), Path(root)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not group_id(group):
        raise ValueError("Use a safe clip name and nonempty source group")
    target = root / "clips" / name
    if target.exists():
        raise FileExistsError(f"{target} exists; use labeler to resume")
    images = None
    if source.is_dir():
        paths = image_files(source)
        if not paths:
            raise ValueError("Image sequence is empty")
        image = cv2.imread(str(paths[0]))
        if image is None:
            raise ValueError("Unreadable first image")
        height, width = image.shape[:2]
        count = len(paths)
        images = [p.name for p in paths]
        source_hash = hashlib.sha256("".join(file_hash(p) for p in paths).encode()).hexdigest()
        kind = "images"
    else:
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open {source}")
            width, height = (int(capture.get(prop)) for prop in
                             (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        source_hash, kind = file_hash(source), "video"
    stop = count if end is None else end
    if not (width > 0 and height > 0 and fps > 0 and 0 <= start < stop <= count):
        raise ValueError("Invalid source dimensions, FPS, or frame range")
    for existing in (root / "clips").glob("*/meta.json"):
        meta = read_json(existing)
        same = (group_id(meta["source_group"]) == group_id(group)
                or meta["source_hash"] == source_hash or meta["source"] == str(source))
        if same and meta["split"] != split:
            raise ValueError(f"Split leakage with {existing.parent.name}: assign the same split")
    meta = {"schema_version": 1, "keypoint_names": NAMES, "source": str(source),
            "source_kind": kind, "source_hash": source_hash, "source_group": group,
            "split": split, "width": width, "height": height, "fps": fps,
            "start_frame": start, "frame_count": stop - start,
            "target": "foreground player's racket", "coordinate_system": "original_pixels"}
    if images is not None:
        meta["images"] = images
    target.mkdir(parents=True)
    save_json(target / "meta.json", meta)
    save_json(target / "annotations.json", {"schema_version": 1, "keypoint_names": NAMES, "frames": {}})
    print(f"Registered {name}: {stop-start} frames, {split}, {group}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--name")
    parser.add_argument("--source-group")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--root", type=Path, default=Path("data/racket"))
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--from-shuttle", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("data/shuttle/videos"))
    parser.add_argument("--test-group", default="match06")
    args = parser.parse_args()
    if args.from_shuttle:
        for path in sorted(args.from_shuttle.glob("*/meta.json")):
            meta = read_json(path)
            split = "test" if group_id(meta["source_group"]) == group_id(args.test_group) else meta["split"]
            source = args.videos / Path(meta["video"].replace("\\", "/")).name
            target = args.root / "clips" / path.parent.name
            if target.exists():
                current = read_json(target / "meta.json")
                if (current["source"] != str(source.resolve()) or current["split"] != split
                        or group_id(current["source_group"]) != group_id(meta["source_group"])):
                    raise ValueError(f"Existing registration disagrees: {target}")
                print(f"Preserved existing registration: {target}")
                continue
            register(source, path.parent.name, meta["source_group"], split, args.root,
                     start=meta["start_frame"], end=meta["start_frame"] + meta["frame_count"])
    else:
        if not all([args.source, args.name, args.source_group, args.split]):
            parser.error("Supply source, --name, --source-group and --split")
        register(args.source, args.name, args.source_group, args.split, args.root,
                 args.fps, args.start_frame, args.end_frame)


if __name__ == "__main__":
    main()
