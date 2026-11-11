"""CLI for preparing, manually labelling, training and evaluating court points."""
import argparse
from pathlib import Path
import sys
import csv

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT.parent))
sys.path.insert(0, str(MODULE_ROOT))

from src.court.learning_data import (DEFAULT_ROOT, GROUND_POINT_NAMES, POINT_NAMES, complete,
                                     import_sources, load_manifest, prepare, sync_images, upgrade_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="sample images without generating any point labels")
    prep.add_argument("source", type=Path, help="MP4 file or directory of MP4 files")
    prep.add_argument("--group", required=True, help="match/recording ID; a whole group stays in one split")
    prep.add_argument("--split", choices=["train", "val", "test"], required=True)
    prep.add_argument("--count", type=int, default=1, help="total sample images; prefer one selected frame per fixed view")
    prep.add_argument("--court-id", required=True, help="physical court/venue ID, shared across matches on that court")
    prep.add_argument("--view-id", required=True, help="globally unique fixed camera view ID")
    upgrade = commands.add_parser("upgrade-labels", help="back up eight-point labels and add four blank points")
    importing = commands.add_parser("import-sources", help="import a roster of diverse court images/selected frames")
    importing.add_argument("csv", type=Path)
    syncing = commands.add_parser("sync-images", help="register new photos in the dataset images folder; preserve existing labels")
    roster = commands.add_parser("make-roster", help="list incoming images in a CSV for you to enter court identities")
    roster.add_argument("source", type=Path)
    roster.add_argument("--output", type=Path, required=True)
    label = commands.add_parser("label", help="open the manual click annotation window")
    label.add_argument("--all-frames", action="store_true", help="also show repeat images of each fixed view")
    status = commands.add_parser("status", help="show annotation progress")
    training = commands.add_parser("train", help="create random weights and train on manual labels")
    training.add_argument("--output", type=Path, default=MODULE_ROOT / "outputs" / "court_training")
    training.add_argument("--epochs", type=int, default=40)
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--width", type=int, default=640)
    training.add_argument("--height", type=int, default=360)
    training.add_argument("--lr", type=float, default=0.001)
    modes = training.add_mutually_exclusive_group()
    modes.add_argument("--overfit", action="store_true", help="pilot only: check on training data, without a held-out score")
    modes.add_argument("--train-only", action="store_true", help="train now, save last.pt, and leave held-out evaluation for later")
    training.add_argument("--all-frames", action="store_true", help="use repeat images instead of one complete image per fixed view")
    evaluation = commands.add_parser("evaluate", help="report manual-label point error")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluation.add_argument("--all-frames", action="store_true")
    prediction = commands.add_parser("predict", help="detect points using your own trained checkpoint")
    prediction.add_argument("checkpoint", type=Path)
    prediction.add_argument("video", type=Path)
    prediction.add_argument("--frame", type=int, default=0)
    prediction.add_argument("--output", type=Path, required=True)
    video_prediction = commands.add_parser("predict-video", help="save an MP4 with model predictions on every frame")
    video_prediction.add_argument("checkpoint", type=Path)
    video_prediction.add_argument("video", type=Path)
    video_prediction.add_argument("--output", type=Path, required=True)
    for command in (prep, label, status, training, evaluation, upgrade, importing, syncing):
        command.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    for command in (training, evaluation, prediction, video_prediction):
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.source, args.root, args.group, args.split, args.count, args.court_id, args.view_id)
        elif args.command == "upgrade-labels":
            upgrade_manifest(args.root)
        elif args.command == "import-sources":
            import_sources(args.csv, args.root)
        elif args.command == "sync-images":
            sync_images(args.root)
        elif args.command == "make-roster":
            if not args.source.is_dir():
                raise ValueError("Source must be an image directory")
            images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            if not images:
                raise ValueError("Put your selected court photos in that folder first")
            if args.output.exists():
                raise ValueError("Roster already exists; choose a new output name")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["source", "group", "court_id", "view_id", "split", "frame"])
                for image in images:
                    writer.writerow([str(image.resolve()), "", "", "", "train", ""])
            print(f"Created {args.output}: {len(images)} images. Fill group, court_id and view_id before importing.")
        elif args.command == "label":
            from src.court.learning_labeler import CourtLabeler
            CourtLabeler(args.root, one_per_view=not args.all_frames).run()
        elif args.command == "status":
            data = load_manifest(args.root)
            for split in ("train", "val", "test"):
                rows = [r for r in data["images"] if r["split"] == split]
                eligible = sum(not r["unusable"] and complete(r) and
                               sum(r["points"][n]["status"] == "visible" for n in GROUND_POINT_NAMES) >= 4 for r in rows)
                excluded = sum(r["unusable"] for r in rows)
                groups = sorted({r["group"] for r in rows})
                usable = [r for r in rows if not r["unusable"]]
                labelled = [r for r in usable if complete(r) and sum(r["points"][n]["status"] == "visible" for n in GROUND_POINT_NAMES) >= 4]
                print(f"{split}: {len(rows)} images, {eligible} complete/usable, {excluded} excluded")
                print(f"  usable: {len({r['court_id'] for r in usable})} court IDs, {len({r['view_id'] for r in usable})} view IDs; labelled views: {len({r['view_id'] for r in labelled})}; source groups: {len(groups)}")
            print("Court/view counts come from your source IDs, not automatic diversity verification.")
        elif args.command == "predict-video":
            from src.court.learning_video import predict_video
            predict_video(args)
        else:
            from src.court import learning_train
            getattr(learning_train, args.command)(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
