"""One selected racket per frame, labelled 'player', without asserting ownership."""
import argparse
import copy
from pathlib import Path

from .schema import CENTER_NAMES, blank, file_hash, load_clip, read_json, save_json


def blank_selected(frame):
    row = blank(frame, CENTER_NAMES)
    row.update(label="player", racket_id="selected_racket", player_id=None,
               track_id=None, player_bbox=None, association_status="unassigned")
    return row


def prepare(source_root, root, clips):
    source_root, root = Path(source_root).resolve(), Path(root).resolve()
    if source_root == root or source_root in root.parents or root in source_root.parents:
        raise ValueError("Use a separate sibling root; existing annotations must remain untouched")
    for name in clips:
        if Path(name).name != name or name in (".", "..") or not name:
            raise ValueError("Invalid clip name")
        meta, _ = load_clip(source_root / "clips" / name)
        target = root / "clips" / name
        if target.exists():
            current, _ = load_clip(target)
            if current.get("annotation_mode") != "selected_racket" or any(current[k] != meta[k] for k in
                    ("source_hash", "source", "frame_count", "start_frame", "source_group", "split")):
                raise ValueError(f"Existing clip disagrees: {target}")
            print(f"Preserved existing player labels: {name}")
            continue
        meta = copy.deepcopy(meta)
        for key in ("players", "player_ids", "identity_scope", "identity_note"):
            meta.pop(key, None)
        meta.update(schema_version=1, keypoint_names=CENTER_NAMES, annotation_mode="selected_racket",
                    target="one selected racket string-bed centre per frame", label="player",
                    ownership="unassigned until explicitly associated with a player crop",
                    previous_annotations=str(source_root / "clips" / name / "annotations.json"))
        target.mkdir(parents=True)
        save_json(target / "meta.json", meta)
        save_json(target / "annotations.json", {"schema_version": 1, "keypoint_names": CENTER_NAMES,
                  "annotation_mode": "selected_racket", "label": "player", "frames": {}})
        print(f"Fresh player annotations: {name}, {meta['frame_count']} frames; old files preserved")


def association_errors(row, meta):
    from .players import binding_errors
    errors = binding_errors(row, meta["width"], meta["height"])
    if row.get("label") != "player" or row.get("racket_id") != "selected_racket":
        errors.append("Expected generic player label and selected_racket slot")
    if row.get("association_status") not in {"unassigned", "verified"}:
        errors.append("Invalid association status")
    if row.get("association_status") == "unassigned" and row.get("player_id") is not None:
        errors.append("Unassigned racket cannot claim a player identity")
    if row.get("association_status") == "verified" and row.get("player_bbox") is None:
        errors.append("Verified association requires a player box")
    if row.get("state") == "invisible" and row.get("association_status") != "verified":
        errors.append("Anonymous missing racket must be unavailable, not a whole-image negative")
    return errors


def bind(clip, mapping):
    """Import explicitly verified crops for the clicked racket, with optional IDs."""
    clip, mapping = Path(clip), Path(mapping)
    meta, ann = load_clip(clip)
    if meta.get("annotation_mode") != "selected_racket":
        raise ValueError("Use a generic player clip")
    value = read_json(mapping)
    if value.get("coordinate_system") != "original_pixels" or value.get("frame_indexing") != "clip_local_zero_based":
        raise ValueError("Declare original_pixels and clip_local_zero_based")
    seen = set()
    for entry in value["frames"]:
        frame = entry["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < meta["frame_count"] or frame in seen:
            raise ValueError("Invalid or duplicate frame")
        seen.add(frame)
        if entry.get("verified") is not True:
            raise ValueError("Only explicitly verified racket-to-player crop associations may be imported")
        row = ann["frames"].setdefault(str(frame), blank_selected(frame))
        row.update(player_bbox=entry["bbox"], player_id=entry.get("player_id"), track_id=entry.get("track_id"),
                   association_status="verified")
        errors = association_errors(row, meta)
        if errors:
            raise ValueError("; ".join(errors))
    ann.setdefault("association_imports", []).append({"path": str(mapping.resolve()), "hash": file_hash(mapping)})
    save_json(clip / "annotations.json", ann)
    print(f"Attached {len(seen)} verified player crops; centre labels unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--source-root", type=Path, default=Path("data/racket_center_dual"))
    preparation.add_argument("--root", type=Path, default=Path("data/racket_player"))
    preparation.add_argument("--clips", nargs="+", default=["rally05", "rally09"])
    binding = sub.add_parser("bind")
    binding.add_argument("clip", type=Path)
    binding.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_root, args.root, args.clips)
    else:
        bind(args.clip, args.mapping)


if __name__ == "__main__":
    main()
