"""Train CourtNet from scratch on the exported dataset.

    python -m src.court.train --dataset data/dataset/court --out outputs/court_net.pt

Watch two numbers: the training loss (should fall steadily) and the validation pixel
error on a venue the model never trained on (this is what tells you if it works).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.court.court_data import CourtDataset, load_labels
from src.court.net import CourtNet, save_checkpoint
from src.court.predict import pick_device, predict_points


def heatmap_loss(logits, target):
    """MSE on sigmoid output, with the few target pixels weighted up (else it learns 'all zero')."""
    prob = torch.sigmoid(logits)
    weight = 1.0 + 20.0 * target
    return (weight * (prob - target) ** 2).sum(dim=(2, 3)).mean()


def validation_errors(model, dataset, device, conf_thresh=0.3):
    """Mean pixel error (as % of image width) over visible points, plus the share found."""
    model.eval()
    errors, found, total = [], 0, 0
    for entry, image in zip(dataset.entries, dataset.images):
        points, conf = predict_points(model, image, device)
        truth = np.array([entry["points"][n] for n in dataset.names])
        visible = np.array([entry["visible"][n] for n in dataset.names])
        width = entry["size"][0]
        errors += list(np.linalg.norm(points[visible] - truth[visible], axis=1) / width * 100)
        found += int((conf[visible] >= conf_thresh).sum())
        total += int(visible.sum())
    return (float(np.mean(errors)) if errors else float("nan")), (found / total if total else 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train the court keypoint network from scratch")
    parser.add_argument("--dataset", default="data/dataset/court")
    parser.add_argument("--out", default="outputs/court_net.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--repeat", type=int, default=10,
                        help="augmented passes over the training set per epoch (small data)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=24, help="network width")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-on-all", action="store_true",
                        help="use EVERY labelled image for training (no validation or test set) and "
                             "save the final epoch. Only for the last model: it cannot be scored honestly.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    names, _ = load_labels(args.dataset)
    train_set = CourtDataset(args.dataset, "all" if args.train_on_all else "train", train=True,
                             repeat=args.repeat, seed=args.seed)
    if args.train_on_all:
        val_set = CourtDataset(args.dataset, "none", train=False)       # empty on purpose
        print("Training on ALL labelled images: there is no validation or test set, so the "
              "final epoch is saved and this model cannot be scored honestly.")
    else:
        val_set = CourtDataset(args.dataset, "val", train=False)
        if len(val_set.entries) == 0:
            val_set = CourtDataset(args.dataset, "test", train=False)
            print("No 'val' images, so using the 'test' images for validation (fine for a first run; "
                  "label 3+ venues to get a separate validation venue).")
    if len(train_set.entries) == 0:
        raise SystemExit("No training images. Run src.court.export_dataset first.")
    print(f"device={device} | train images={len(train_set.entries)} (x{args.repeat} augmented) "
          f"| val images={len(val_set.entries)} | points={len(names)}")

    model = CourtNet(len(names), args.base).to(device)
    model.head.bias.data.fill_(-4.0)                      # start by predicting "no point"
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=len(train_set) >= 2 * args.batch)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best, start = float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, heatmaps, _ in loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            loss = heatmap_loss(model(images), heatmaps)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(losses))
        message = f"epoch {epoch:3d}/{args.epochs} loss {train_loss:8.2f}"
        if args.train_on_all:
            if epoch == args.epochs:
                save_checkpoint(args.out, model, names, args.base,
                                {"epoch": epoch, "score": train_loss, "trained_on_all": True})
                message += "  <- saved final model"
        elif epoch % args.val_every == 0 or epoch == args.epochs:
            if len(val_set.entries):
                error, found = validation_errors(model, val_set, device)
                message += f" | val error {error:5.2f}% of width, points found {found:4.0%}"
                score = error if np.isfinite(error) else float("inf")
            else:
                score = train_loss
            if score < best:
                best = score
                save_checkpoint(args.out, model, names, args.base, {"epoch": epoch, "score": score})
                message += "  <- saved best"
        print(message + f"  [{time.time() - start:.0f}s]")
    print(f"Done. Saved: {args.out}")


if __name__ == "__main__":
    main()
