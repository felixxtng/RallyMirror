"""Two independent, clip-local player/racket identities per frame (schema v2).

Player 1 starts on the near side; player 2 starts on the far side. IDs follow
the people throughout a clip, not whichever person is lower in a later frame.
Tracker association is explicit, optional, and never guessed from point distance.
"""
import argparse
import copy
import math
from collections import Counter
from collections.abc import MutableMapping
from pathlib import Path

from .schema import CENTER_NAMES, approved, blank, file_hash, load_clip, point_errors, read_json, save_json

PLAYER_IDS = ("player_1", "player_2")


def blank_player(frame, player):
    if player not in PLAYER_IDS:
        raise ValueError("Unknown player identity")
    row = blank(frame, CENTER_NAMES)
    row.update(player_id=player, racket_id=f"{player}_racket", track_id=None, player_bbox=None)
    return row


def blank_frame(frame):
    return {"frame": frame, "rackets": {p: blank_player(frame, p) for p in PLAYER_IDS}}


class PlayerRows(MutableMapping):
    """Mutable per-player view; editing one instance never replaces its partner."""
    def __init__(self, frames, player):
        self.frames, self.player = frames, player

    def __getitem__(self, key):
        return self.frames[key]["rackets"][self.player]

    def __setitem__(self, key, value):
        self.frames.setdefault(key, blank_frame(int(key)))["rackets"][self.player] = value

    def __delitem__(self, key):
        self.frames[key]["rackets"][self.player] = blank_player(int(key), self.player)

    def __iter__(self):
        return iter(self.frames)

    def __len__(self):
        return len(self.frames)


def frame_approved(row, meta):
    return all(approved(row["rackets"][p], meta, {}) for p in PLAYER_IDS)


def migrate(source_root, root):
    source_root, root = Path(source_root).resolve(), Path(root).resolve()
    if source_root == root or source_root in root.parents or root in source_root.parents:
        raise ValueError("Use a separate sibling directory for the two-player dataset")
    paths = sorted((source_root / "clips").glob("*/meta.json"))
    if not paths:
        raise ValueError("No source clips found")
    counts = Counter()
    for path in paths:
        meta, original = load_clip(path.parent)
        if meta["schema_version"] != 1 or meta["keypoint_names"] != CENTER_NAMES:
            raise ValueError("Migrate from the single-player head-centre dataset")
        clip = root / "clips" / path.parent.name
        if clip.exists():
            current, ann = load_clip(clip)
            if current["schema_version"] != 2 or any(current[k] != meta[k] for k in
                    ("source_hash", "source", "split", "source_group", "start_frame", "frame_count")):
                raise ValueError(f"Existing two-player registration disagrees: {clip}")
        else:
            current = copy.deepcopy(meta)
            current.update(schema_version=2, target="both players' racket string-bed centres",
                           player_ids=list(PLAYER_IDS), identity_scope="clip",
                           players={"player_1": {"initial_side": "near", "display_name": None},
                                    "player_2": {"initial_side": "far", "display_name": None}},
                           migrated_from=str(path.parent),
                           identity_note="Legacy foreground labels assumed player_1; verify if source has camera cuts or side swaps")
            ann = {"schema_version": 2, "keypoint_names": CENTER_NAMES,
                   "player_ids": list(PLAYER_IDS), "frames": {}}
            clip.mkdir(parents=True)
            save_json(clip / "meta.json", current)
        # Both slots always exist. A blank slot means unlabelled, never absent.
        for frame in range(meta["frame_count"]):
            key = str(frame)
            ann["frames"].setdefault(key, blank_frame(frame))
            if key not in original["frames"]:
                continue
            existing = ann["frames"][key]["rackets"]["player_1"]
            if existing["state"] != "unlabelled" or existing.get("approved"):
                continue
            converted = copy.deepcopy(original["frames"][key])
            converted.update(player_id="player_1", racket_id="player_1_racket", track_id=existing.get("track_id"), player_bbox=existing.get("player_bbox"),
                             origin_single_player={"clip": str(path.parent), "frame": frame})
            ann["frames"][key]["rackets"]["player_1"] = converted
            counts["player_1_labels_copied"] += int(approved(converted, current, {}))
        save_json(clip / "annotations.json", ann)
        counts["clips"] += 1
    print(dict(counts))
    return dict(counts)


def binding_errors(row, width, height):
    errors = []
    track = row.get("track_id")
    if track is not None and (isinstance(track, bool) or not isinstance(track, (int, str)) or track == ""):
        errors.append("track_id must be an integer, nonempty string, or null")
    box = row.get("player_bbox")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box):
            errors.append("player_bbox must be finite original-pixel [x1,y1,x2,y2]")
        elif not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            errors.append("player_bbox outside original image or empty")
    return errors


def validate_clip(meta, ann):
    errors, counts = [], Counter()
    if meta.get("keypoint_names") != CENTER_NAMES or ann.get("keypoint_names") != CENTER_NAMES:
        errors.append("Two-player schema requires head_center only")
    if meta.get("player_ids") != list(PLAYER_IDS) or ann.get("player_ids") != list(PLAYER_IDS):
        errors.append("Expected exactly player_1 and player_2")
    if meta.get("identity_scope") != "clip":
        errors.append("Expected clip-local stable player identities")
    if set(meta.get("players", {})) != set(PLAYER_IDS):
        errors.append("Missing player identity definitions")
    frames = ann["frames"]
    for key, frame in frames.items():
        if str(frame.get("frame")) != key or not 0 <= int(key) < meta["frame_count"]:
            errors.append(f"{key}: invalid frame index")
        if set(frame.get("rackets", {})) != set(PLAYER_IDS):
            errors.append(f"{key}: expected two racket slots")
            continue
        tracks = []
        for player, row in frame["rackets"].items():
            prefix = f"{key}/{player}: "
            if row.get("frame") != int(key) or row.get("player_id") != player or row.get("racket_id") != f"{player}_racket":
                errors.append(prefix + "instance identity/frame mismatch")
            if not isinstance(row.get("approved"), bool):
                errors.append(prefix + "invalid approval flag")
            if row.get("provenance") not in {"manual", "migrated", "model", "copied"}:
                errors.append(prefix + "invalid provenance; interpolation is not supported")
            if row.get("state") not in {"unlabelled", "draft", "visible", "invisible", "unavailable"}:
                errors.append(prefix + "invalid state")
            if [p.get("name") for p in row.get("keypoints", [])] != CENTER_NAMES:
                errors.append(prefix + "expected one head_center")
            if row.get("approved") or row.get("state") in {"visible", "invisible", "unavailable"}:
                errors.extend(prefix + e for e in point_errors(row, meta["width"], meta["height"], CENTER_NAMES))
            else:
                for p in row.get("keypoints", []):
                    if p.get("v") is not None:
                        trial = dict(row, state="visible" if p["v"] else "unavailable")
                        errors.extend(prefix + e for e in point_errors(trial, meta["width"], meta["height"], CENTER_NAMES))
            errors.extend(prefix + e for e in binding_errors(row, meta["width"], meta["height"]))
            if row.get("track_id") is not None:
                tracks.append(str(row["track_id"]))
            done = approved(row, meta, {})
            counts[f"approved_{player}"] += int(done)
            counts["approved_instances"] += int(done)
        if len(tracks) != len(set(tracks)):
            errors.append(f"{key}: both rackets assigned to the same tracker ID")
        counts["approved_frames"] += int(frame_approved(frame, meta))
    counts["unfinished_instances"] = 2*meta["frame_count"] - counts["approved_instances"]
    counts["unfinished_frames"] = meta["frame_count"] - counts["approved_frames"]
    return errors, dict(counts)


def progress(root):
    totals = Counter()
    result = []
    for path in sorted((Path(root) / "clips").glob("*/meta.json")):
        meta, ann = load_clip(path.parent)
        if meta["schema_version"] != 2:
            raise ValueError("Use the two-player dataset root")
        item = {"clip": path.parent.name, "split": meta["split"], "frames": meta["frame_count"], "players": {}}
        for player in PLAYER_IDS:
            rows = PlayerRows(ann["frames"], player)
            done = {int(k) for k, r in rows.items() if approved(r, meta, rows)}
            item["players"][player] = {"reviewed": len(done),
                "located": sum(r["state"] == "visible" for k, r in rows.items() if int(k) in done),
                "first_unfinished": next((f for f in range(meta["frame_count"]) if f not in done), None)}
            totals[f"approved_{player}"] += len(done)
            for key, row in rows.items():
                if "proposal" not in row:
                    continue
                if int(key) not in done:
                    totals["model_pending"] += 1
                elif row["state"] == "unavailable":
                    totals["model_excluded_unknown"] += 1
                elif row["state"] == "invisible":
                    totals["model_confirmed_absent"] += 1
                else:
                    p, q = row["keypoints"][0], row["proposal"]["point"]
                    same = all(p[k] == q[k] for k in ("x", "y", "v"))
                    totals["model_accepted_unchanged" if same else "model_corrected"] += 1
        complete = sum(frame_approved(f, meta) for f in ann["frames"].values())
        item["both_reviewed"] = complete
        totals["complete_clips"] += int(complete == meta["frame_count"])
        totals["frames"] += meta["frame_count"]
        result.append(item)
        if any(p["reviewed"] for p in item["players"].values()):
            print(item)
    totals["clips"] = len(result)
    totals["unfinished_instances"] = totals["frames"]*2 - totals["approved_player_1"] - totals["approved_player_2"]
    print("TWO-PLAYER TOTALS", dict(totals))
    return {"clips": result, "totals": dict(totals)}


def bind(clip, path):
    """Attach explicit teammate associations. Validate all edits before saving."""
    clip, path = Path(clip), Path(path)
    meta, ann = load_clip(clip)
    if meta["schema_version"] != 2:
        raise ValueError("Use a two-player clip")
    from .tracking_adapter import normalize_tracking
    mappings = normalize_tracking(read_json(path))
    seen = set()
    count = 0
    for frame in mappings["frames"]:
        index = frame["frame"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < meta["frame_count"] or index in seen:
            raise ValueError("Invalid or duplicate mapping frame")
        seen.add(index)
        for player, identity in frame["players"].items():
            if player not in PLAYER_IDS:
                raise ValueError("Unknown player in mapping")
            row = ann["frames"].setdefault(str(index), blank_frame(index))["rackets"][player]
            if "track_id" in identity:
                row["track_id"] = identity["track_id"]
            if "bbox" in identity:
                row["player_bbox"] = identity["bbox"]
            count += 1
    errors, _ = validate_clip(meta, ann)
    if errors:
        raise ValueError("\n".join(errors))
    ann.setdefault("tracking_imports", []).append({"path": str(path.resolve()), "hash": file_hash(path)})
    save_json(clip / "annotations.json", ann)
    print(f"Attached {count} player associations; point coordinates and approvals unchanged")
    return count


def crop_bounds(box, width, height, padding=.75):
    if box is None:
        return None
    if binding_errors({"player_bbox": box}, width, height) or not math.isfinite(padding) or padding < 0:
        raise ValueError("Invalid player crop")
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    return [max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    migration = sub.add_parser("migrate")
    migration.add_argument("--source-root", type=Path, default=Path("data/racket_center"))
    migration.add_argument("--root", type=Path, default=Path("data/racket_center_dual"))
    status = sub.add_parser("status")
    status.add_argument("--root", type=Path, default=Path("data/racket_center_dual"))
    binding = sub.add_parser("bind")
    binding.add_argument("clip", type=Path)
    binding.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "migrate":
        migrate(args.source_root, args.root)
    elif args.command == "status":
        progress(args.root)
    else:
        bind(args.clip, args.mapping)


if __name__ == "__main__":
    main()
