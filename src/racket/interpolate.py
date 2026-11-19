"""Propose short-gap interpolation; nothing is approved automatically."""
import argparse
import copy
import csv
import math
from pathlib import Path

from .schema import approved, digest, load_clip, save_json


def interpolate(meta, annotations, max_gap=10, max_speed=1500, max_length_ratio=1.5):
    if meta.get("keypoint_names") == ["head_center"]:
        raise ValueError("Head-centre workflow uses image-based model proposals, not interpolation")
    rows = annotations["frames"]
    anchors = sorted(int(k) for k, row in rows.items()
                     if row.get("provenance") == "manual" and approved(row, meta, rows))
    for left, right in zip(anchors, anchors[1:]):
        a, b = rows[str(left)], rows[str(right)]
        if right - left > max_gap or a["racket_id"] != b["racket_id"]:
            continue
        if a["state"] != b["state"] or a["state"] == "invisible":
            # Absence between two keyframes does not prove absence throughout the gap.
            continue
        if [p["v"] for p in a["keypoints"]] != [p["v"] for p in b["keypoints"]]:
            continue
        reasons = []
        for p, q in zip(a["keypoints"], b["keypoints"]):
            if p["v"]:
                speed = math.hypot(q["x"]-p["x"], q["y"]-p["y"]) * meta["fps"] / (right-left)
                if speed > max_speed:
                    reasons.append("fast_motion")
        for i, j in ((0, 1), (1, 2)):
            lengths = [math.hypot(r["keypoints"][j]["x"]-r["keypoints"][i]["x"],
                                  r["keypoints"][j]["y"]-r["keypoints"][i]["y"])
                       for r in (a, b) if r["keypoints"][i]["v"] and r["keypoints"][j]["v"]]
            if len(lengths) == 2 and max(lengths) / max(min(lengths), 1) > max_length_ratio:
                reasons.append("geometry_change")
        for number in range(left + 1, right):
            existing = rows.get(str(number))
            if existing and existing.get("state") != "unlabelled" and (
                    existing.get("provenance") != "interpolated" or approved(existing, meta, rows)):
                continue
            row = copy.deepcopy(a)
            row.update(frame=number, provenance="interpolated", approved=False,
                       anchors={str(left): digest(a), str(right): digest(b)},
                       review_reasons=sorted(set(reasons or ["interpolation_needs_review"])))
            t = (number-left)/(right-left)
            for p, q, result in zip(a["keypoints"], b["keypoints"], row["keypoints"]):
                if p["v"]:
                    result["x"] = round(p["x"] * (1-t) + q["x"] * t, 3)
                    result["y"] = round(p["y"] * (1-t) + q["y"] * t, 3)
            rows[str(number)] = row
    return annotations


def review_queue(meta, annotations):
    rows = annotations["frames"]
    result = []
    for frame in range(meta["frame_count"]):
        row = rows.get(str(frame))
        if not row or not approved(row, meta, rows):
            reasons = row.get("review_reasons", ["incomplete_or_unreviewed"]) if row else ["no_reliable_interpolation"]
            result.append({"frame": frame, "source_frame": frame + meta["start_frame"],
                           "reason": ";".join(reasons)})
    return result


def write_queue(clip, meta, annotations):
    path = Path(clip) / "review_queue.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "source_frame", "reason"])
        writer.writeheader()
        writer.writerows(review_queue(meta, annotations))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-speed", type=float, default=1500)
    parser.add_argument("--max-length-ratio", type=float, default=1.5)
    args = parser.parse_args()
    if args.max_gap < 1 or args.max_speed <= 0 or args.max_length_ratio < 1:
        parser.error("Require positive gap/speed and length ratio >= 1")
    meta, annotations = load_clip(args.clip)
    interpolate(meta, annotations, args.max_gap, args.max_speed, args.max_length_ratio)
    save_json(args.clip / "annotations.json", annotations)
    path = write_queue(args.clip, meta, annotations)
    print(f"Saved proposals and {len(review_queue(meta, annotations))} review frames to {path}")


if __name__ == "__main__":
    main()
