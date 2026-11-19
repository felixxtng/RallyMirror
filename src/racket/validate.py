"""Validate annotations and detect leakage across all three splits."""
import argparse
from collections import Counter
from pathlib import Path

from .schema import NAMES, CENTER_NAMES, SPLITS, anchors_current, approved, file_hash, group_id, load_clip, point_errors, read_json
from .source import FrameSource


def validate(root, check_frames=False, verify_sources=False):
    errors, counts = [], Counter()
    if not (Path(root) / "clips").is_dir():
        return [f"No registered clips directory: {root}"], {}
    groups, sources, paths = {}, {}, {}
    for clip in sorted((Path(root) / "clips").iterdir()):
        if not clip.is_dir():
            continue
        try:
            meta, annotations = load_clip(clip)
            split = meta["split"]
            if split not in SPLITS:
                errors.append(f"{clip.name}: invalid split")
            for mapping, identity in ((groups, group_id(meta["source_group"])),
                                      (sources, meta["source_hash"]), (paths, meta["source"])):
                if identity in mapping and mapping[identity] != split:
                    errors.append(f"{clip.name}: split leakage for {identity}")
                mapping[identity] = split
            names = meta.get("keypoint_names")
            if names not in (NAMES, CENTER_NAMES) or annotations.get("keypoint_names") != names:
                errors.append(f"{clip.name}: inconsistent keypoint order")
            if meta.get("schema_version") not in (1, 2) or annotations.get("schema_version") != meta.get("schema_version"):
                errors.append(f"{clip.name}: unsupported schema version")
            if meta.get("coordinate_system") != "original_pixels":
                errors.append(f"{clip.name}: unsupported coordinate system")
            if not (meta["width"] > 0 and meta["height"] > 0 and meta["fps"] > 0
                    and meta["frame_count"] > 0 and meta["start_frame"] >= 0):
                errors.append(f"{clip.name}: invalid dimensions, FPS or frame range")
            if not Path(meta["source"]).exists():
                errors.append(f"{clip.name}: source missing")
            elif verify_sources and meta["source_kind"] == "video" and file_hash(meta["source"]) != meta["source_hash"]:
                errors.append(f"{clip.name}: source contents changed")
            rows = annotations["frames"]
            generic = meta.get("annotation_mode") == "selected_racket"
            if generic and (names != CENTER_NAMES or annotations.get("annotation_mode") != "selected_racket"):
                errors.append(f"{clip.name}: generic player mode requires head_center and matching annotation mode")
            if meta.get("schema_version") == 2:
                from .players import validate_clip
                instance_errors, instance_counts = validate_clip(meta, annotations)
                errors.extend(f"{clip.name}/{e}" for e in instance_errors)
                counts["clips_"+split] += 1
                counts["frames_"+split] += meta["frame_count"]
                counts["approved_"+split] += instance_counts["approved_instances"]
                counts["approved_frames_"+split] += instance_counts["approved_frames"]
                for player in ("player_1", "player_2"):
                    counts[f"approved_{player}_{split}"] += instance_counts[f"approved_{player}"]
                counts["unfinished_instances"] += instance_counts["unfinished_instances"]
                counts["unfinished"] += instance_counts["unfinished_frames"]
                if check_frames:
                    source = FrameSource(meta)
                    try:
                        for frame in range(meta["frame_count"]):
                            source.read(frame)
                    finally:
                        source.close()
                continue
            for key, row in rows.items():
                if not isinstance(row.get("approved"), bool):
                    errors.append(f"{clip.name}/{key}: approval must be boolean")
                if str(row.get("frame")) != key or not 0 <= int(key) < meta["frame_count"]:
                    errors.append(f"{clip.name}/{key}: invalid frame number")
                if [p.get("name") for p in row.get("keypoints", [])] != names:
                    errors.append(f"{clip.name}/{key}: missing points or inconsistent order")
                if row.get("provenance") not in {"manual", "copied", "interpolated", "migrated", "model"}:
                    errors.append(f"{clip.name}/{key}: invalid provenance")
                if generic:
                    from .selected import association_errors
                    errors.extend(f"{clip.name}/{key}: {e}" for e in association_errors(row, meta))
                elif row.get("racket_id") != "foreground":
                    errors.append(f"{clip.name}/{key}: expected foreground racket identity")
                if row.get("state") not in {"unlabelled", "draft", "visible", "invisible", "unavailable"}:
                    errors.append(f"{clip.name}/{key}: invalid annotation state")
                if row.get("state") in {"visible", "invisible", "unavailable"} or row.get("approved"):
                    errors.extend(f"{clip.name}/{key}: {e}" for e in point_errors(row, meta["width"], meta["height"], names))
                else:
                    # Drafts may omit points, but any entered coordinate must be valid.
                    for p in row.get("keypoints", []):
                        if p.get("v") is not None:
                            trial = dict(row, state="visible", keypoints=[
                                p if name == p["name"] else {"name": name, "x": None, "y": None, "v": 0}
                                for name in names])
                            errors.extend(f"{clip.name}/{key}: {e}" for e in point_errors(trial, meta["width"], meta["height"], names)
                                          if e != "visible racket has no located points")
                if row.get("provenance") == "interpolated":
                    endpoints = row.get("anchors", {})
                    if len(endpoints) != 2 or not min(map(int, endpoints)) < int(key) < max(map(int, endpoints)):
                        errors.append(f"{clip.name}/{key}: invalid interpolation endpoints")
                    if row.get("approved") and not anchors_current(row, rows):
                        errors.append(f"{clip.name}/{key}: approved interpolation is stale; regenerate/review")
                counts["approved_"+split] += int(approved(row, meta, rows))
            counts["clips_"+split] += 1
            counts["frames_"+split] += meta["frame_count"]
            counts["unfinished"] += meta["frame_count"] - sum(approved(r, meta, rows) for r in rows.values())
            if check_frames:
                source = FrameSource(meta)
                try:
                    for frame in range(meta["frame_count"]):
                        source.read(frame)
                finally:
                    source.close()
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            errors.append(f"{clip.name}: {exc}")
    return errors, dict(counts)


def validate_export(root, require_splits=False):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    errors, counts = [], Counter()
    seen_groups, seen_sources, seen_images = {}, {}, {}
    names = manifest.get("keypoint_names")
    if names not in (NAMES, CENTER_NAMES):
        errors.append("Export keypoint order disagrees")
    for sample in manifest["samples"]:
        split = sample["split"]
        if split not in SPLITS:
            errors.append("Invalid export split")
        for seen, identity in ((seen_groups, group_id(sample["source_group"])),
                               (seen_sources, sample["source_hash"]),
                               (seen_images, sample["image_hash"])):
            if identity in seen and seen[identity] != split:
                errors.append(f"Export split leakage: {identity}")
            seen[identity] = split
        if not (root / sample["image"]).is_file():
            errors.append(f"Missing exported image {sample['image']}")
        elif file_hash(root / sample["image"]) != sample["image_hash"]:
            errors.append(f"Changed exported image {sample['image']}")
        errors.extend(point_errors(sample["annotation"], sample["width"], sample["height"], names))
        if sample["annotation"].get("state") == "unavailable":
            errors.append("Unknown locations must be excluded, not exported as negatives")
        if sample["annotation"].get("approved") is not True:
            errors.append("Unapproved sample in export")
        counts[split] += 1
    if require_splits:
        for split in sorted(SPLITS):
            if not counts[split]:
                errors.append(f"No approved examples in {split}")
    return errors, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/racket"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--check-frames", action="store_true")
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--require-splits", action="store_true")
    args = parser.parse_args()
    errors, counts = (validate_export(args.export, args.require_splits) if args.export
                      else validate(args.root, args.check_frames, args.verify_sources))
    print(counts)
    for error in errors:
        print("ERROR:", error)
    if errors:
        raise SystemExit(1)
    print("Validation passed. Unlabelled frames are excluded, not negative examples.")


if __name__ == "__main__":
    main()
