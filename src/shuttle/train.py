"""Train using labelled rally clips; hold out whole source matches."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ShuttleDataset
from .model import ShuttleNet, heatmap_loss


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for frames, target in loader:
            frames, target = frames.to(device), target.to(device)
            loss = heatmap_loss(model(frames), target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.detach().cpu()) * frames.shape[0]
            count += frames.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/shuttle/clips"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shuttle"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    torch.manual_seed(42)
    np.random.seed(42)
    train = ShuttleDataset(args.root, "train", augment=True)
    val = ShuttleDataset(args.root, "val")
    if not len(train) or not len(val):
        parser.error("Prepare and label at least one train clip and one val clip first")
    overlap = train.groups & val.groups
    if overlap:
        parser.error(f"Same source match in train and val: {sorted(overlap)}. Hold out a whole match.")
    if len(train.dimensions | val.dimensions) != 1:
        parser.error("Prepare every clip with the same width and height")
    train_visible = sum(row["visible"] == 1 for _, _, row in train.samples)
    val_visible = sum(row["visible"] == 1 for _, _, row in val.samples)
    if not train_visible or not val_visible:
        parser.error("Both splits need some visible shuttle labels")
    width, height = next(iter(train.dimensions))
    device = choose_device(args.device)
    print(f"Device: {device}; input: {width}x{height}", flush=True)
    print(f"Train: {len(train)} labels ({train_visible} visible) across {len(train.clips)} clips", flush=True)
    print(f"Val: {len(val)} labels ({val_visible} visible) across {len(val.clips)} clips", flush=True)
    if train_visible == len(train):
        print("No absent-shuttle training frames: label off-screen and hidden frames with N.", flush=True)
    model = ShuttleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # num_workers=0 also works with OpenCV and Apple's MPS backend on macOS.
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists():
        parser.error("Output already contains a training run. Choose a new --output directory.")
    summary = {"train_clips": train.clips, "val_clips": val.clips,
               "train_groups": sorted(train.groups), "val_groups": sorted(val.groups),
               "train_labels": len(train), "val_labels": len(val),
               "width": width, "height": height, "epochs": args.epochs,
               "batch_size": args.batch_size, "lr": args.lr,
               "heatmap_radius": 3, "loss": "balanced BCE with hard background pixels"}
    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    best = float("inf")
    with (args.output / "history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            val_loss = run_epoch(model, val_loader, device)
            writer.writerow([epoch, train_loss, val_loss])
            file.flush()
            checkpoint = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "width": width, "height": height, "base_channels": 16,
                          "epoch": epoch, "val_loss": val_loss, "method": "three-frame RGB heatmap"}
            torch.save(checkpoint, args.output / "last.pt")
            if val_loss < best:
                best = val_loss
                torch.save(checkpoint, args.output / "best.pt")
            print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.5f} val={val_loss:.5f}", flush=True)
    print(f"Saved {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
