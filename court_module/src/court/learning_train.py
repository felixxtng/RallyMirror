"""Training, held-out evaluation and inference for manually labelled court points."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .court_model import COURT_LENGTH, COURT_POINTS, COURT_WIDTH
from .homography import apply_homography
from .draw import overlay_court
from .learning_data import ARCHITECTURE, CourtDataset, GROUND_POINT_NAMES, POINT_NAMES, image_tensor
from .learning_model import CourtNet, decode_points, keypoint_loss


def choose_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_error, correct, count = 0.0, 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, target, mask, points in loader:
            images, target, mask = images.to(device), target.to(device), mask.to(device)
            logits = model(images)
            loss = keypoint_loss(logits, target, mask)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            selected = mask.bool()
            error = torch.linalg.vector_norm(decode_points(logits.detach()) - points.to(device), dim=-1)[selected]
            n = error.numel()
            total_loss += float(loss.detach()) * n
            total_error += float(error.sum())
            correct += int((error <= 10).sum())
            count += n
    if count == 0:
        raise ValueError("No visible labelled points available")
    return {"loss": total_loss / count, "mean_error_input_px": total_error / count,
            "fraction_within_10_input_px": correct / count, "points": count}


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = choose_device(args.device)
    training = CourtDataset(args.root, "train", args.width, args.height, augment=not args.overfit,
                            one_per_view=not args.all_frames)
    validation = CourtDataset(args.root, "val", args.width, args.height, one_per_view=not args.all_frames)
    if not len(training):
        raise ValueError("No complete training images with at least four visible points. Run label first.")
    missing = [name for name in POINT_NAMES if not any(r["points"][name]["status"] == "visible" for r in training.records)]
    if missing:
        raise ValueError(f"No visible training labels for: {', '.join(missing)}. Add clear examples for every point.")
    if not args.overfit and not args.train_only and not len(validation):
        raise ValueError("Label validation courts, or use --train-only to train now and evaluate on different courts later.")
    if args.output.exists():
        raise ValueError(f"Training output already exists: {args.output}. Choose a new --output directory.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    model = CourtNet().to(device)  # Fresh random initialization; no weights loaded.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    check_data = CourtDataset(args.root, "train", args.width, args.height, one_per_view=not args.all_frames) if args.overfit else validation
    check_loader = None if args.train_only else DataLoader(check_data, batch_size=args.batch_size, num_workers=0)
    args.output.mkdir(parents=True)
    mode = "training_only_not_validated" if args.train_only else "pilot_overfit_not_held_out" if args.overfit else "held_out_validation"
    summary = {"initialization": "random", "pretrained_weights": False, "architecture": ARCHITECTURE,
               "mode": mode, "point_names": list(POINT_NAMES), "width": args.width, "height": args.height,
               "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
               "train_groups": sorted(training.groups), "val_groups": [] if args.overfit or args.train_only else sorted(validation.groups),
               "train_courts": sorted(training.courts), "train_views": sorted(training.views),
               "val_courts": [] if args.overfit or args.train_only else sorted(validation.courts),
               "val_views": [] if args.overfit or args.train_only else sorted(validation.views),
               "selection": "all_frames" if args.all_frames else "one_image_per_view",
               "training_annotations": training.records, "validation_annotations": [] if args.overfit or args.train_only else validation.records}
    (args.output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device}. Train images: {len(training)}. Courts: {len(training.courts)}. Views: {len(training.views)}. Mode: {mode}", flush=True)
    best = float("inf")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "check_loss", "check_mean_error_input_px", "check_fraction_within_10_input_px"])
        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, device, optimizer)
            check = None if check_loader is None else run_epoch(model, check_loader, device)
            writer.writerow([epoch, tr["loss"], check["loss"] if check else "", check["mean_error_input_px"] if check else "", check["fraction_within_10_input_px"] if check else ""])
            file.flush()
            checkpoint = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "architecture": ARCHITECTURE, "point_names": list(POINT_NAMES),
                          "width": args.width, "height": args.height, "mode": mode,
                          "epoch": epoch, "check_metrics": check, "pretrained_weights": False}
            torch.save(checkpoint, args.output / "last.pt")
            if check is not None and check["mean_error_input_px"] < best:
                best = check["mean_error_input_px"]
                torch.save(checkpoint, args.output / "best.pt")
            report = f", check error={check['mean_error_input_px']:.1f} input px" if check else " (training only)"
            print(f"Epoch {epoch}/{args.epochs}: loss={tr['loss']:.4f}{report}", flush=True)
    print(f"Saved {args.output / ('last.pt' if args.train_only else 'best.pt')}")
    if args.overfit:
        print("Pilot results measure memorisation of training images, not generalisation.")
    if args.train_only:
        print("No validation or test score produced. Evaluate on different courts later.")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("point_names") != list(POINT_NAMES):
        raise ValueError("This pipeline needs a newly trained 12-point checkpoint. The old 8-point pilot is preserved but incompatible.")
    model = CourtNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate(args):
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = CourtDataset(args.root, args.split, checkpoint["width"], checkpoint["height"], one_per_view=not args.all_frames)
    if not len(dataset):
        raise ValueError(f"No complete usable images in {args.split}")
    run_file = args.checkpoint.parent / "run.json"
    identity_verified = False
    if args.split in ("val", "test"):
        if not run_file.exists():
            raise ValueError("Keep run.json beside the checkpoint so held-out groups can be audited")
        run = json.loads(run_file.read_text(encoding="utf-8"))
        if dataset.groups & set(run["train_groups"]):
            raise ValueError("Evaluation groups overlap the training groups")
        if args.split == "test" and dataset.groups & set(run["val_groups"]):
            raise ValueError("Test groups were used for validation/model selection")
        if dataset.courts & set(run["train_courts"]) or dataset.views & set(run["train_views"]):
            raise ValueError("Evaluation courts/camera views overlap training")
        if args.split == "test" and (dataset.courts & set(run["val_courts"]) or dataset.views & set(run["val_views"])):
            raise ValueError("Test courts/camera views were used for validation/model selection")
        identity_verified = all(r.get("court_identity_status") == "verified"
                                for r in dataset.records + run["training_annotations"] + run["validation_annotations"])
    metrics = run_epoch(model, DataLoader(dataset, batch_size=2, num_workers=0), device)
    scope = "training_images" if args.split == "train" else "held_out_courts" if identity_verified else "labelled_images_court_independence_unverified"
    print(json.dumps({"split": args.split, "evaluation_scope": scope,
                      "images": len(dataset), "groups": sorted(dataset.groups),
                      "input_size": [checkpoint["width"], checkpoint["height"]], **metrics}, indent=2))
    if args.split == "train":
        print("Training-set performance is not held-out accuracy.")
    elif not identity_verified:
        print("Point error on labelled images only: physical court independence has not been verified.")


def fit_predictions(points, image_size, tolerance):
    """Reject inconsistent or collapsed predictions before exporting geometry.

    This is a plausibility check, not an independent accuracy measurement.
    """
    model_points = np.float32([COURT_POINTS[n] for n in GROUND_POINT_NAMES])
    # Raised net endpoints are learned and displayed, but NEVER treated as
    # ground-plane correspondences. Wrong net heights cannot warp the floor.
    ground_points = np.float32(points)[[POINT_NAMES.index(n) for n in GROUND_POINT_NAMES]]
    H, inliers = cv2.findHomography(model_points, ground_points, cv2.RANSAC, tolerance)
    if H is None or inliers is None or int(inliers.sum()) < 6 or np.linalg.matrix_rank(H) < 3:
        return None
    corners = apply_homography(H, [[0, 0], [COURT_WIDTH, 0],
                                   [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]])
    width, height = image_size
    if not np.isfinite(corners).all():
        return None
    polygon = corners.astype(np.float32)
    if (not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < width * height * 0.10
            or corners[:, 0].min() < -width * 0.05 or corners[:, 0].max() > width * 1.05
            or corners[:, 1].min() < -height * 0.05 or corners[:, 1].max() > height * 1.05
            or corners[:2, 1].mean() <= corners[2:, 1].mean()):
        return None
    inverse = np.linalg.inv(H)
    if abs(inverse[2, 2]) < 1e-10:
        return None
    return inverse / inverse[2, 2], int(inliers.sum())


def predict(args):
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    video = cv2.VideoCapture(str(args.video))
    try:
        if not video.isOpened() or args.frame < 0:
            raise ValueError("Cannot open video or frame index is negative")
        video.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = video.read()
    finally:
        video.release()
    if not ok:
        raise ValueError("Cannot read the selected frame")
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}. Choose another --output directory")
    width, height = checkpoint["width"], checkpoint["height"]
    with torch.no_grad():
        points = decode_points(model(image_tensor(frame, width, height)[None].to(device)))[0].cpu().numpy()
    points *= [frame.shape[1] / width, frame.shape[0] / height]
    # This is a geometric fit to NETWORK predictions, not a line detector.
    fit = fit_predictions(points, (frame.shape[1], frame.shape[0]),
                          max(8.0, 6 * frame.shape[1] / width, 6 * frame.shape[0] / height))
    warnings = ["Inspect predictions against independent labels; a heatmap maximum is not confidence.",
                "This model has no visibility head; hidden/off-screen points can be unreliable."]
    calibration = None
    if fit is not None:
        H, inliers = fit
        calibration = {"H_image_to_court": H.tolist(), "image_size": [frame.shape[1], frame.shape[0]],
                       "video": str(args.video.resolve()), "frame_index": args.frame,
                       "method": ARCHITECTURE, "stats": {"ground_ransac_inliers": inliers},
                       "ground_point_names": list(GROUND_POINT_NAMES),
                       "net_top_points_image": {n: points[POINT_NAMES.index(n)].tolist() for n in ("NL-TOP", "NR-TOP")},
                       "service_regions_court_m": service_regions(),
                       "warnings": warnings}
    args.output.mkdir(parents=True)
    view = frame.copy()
    if calibration:
        tint_service_regions(view, np.array(calibration["H_image_to_court"]))
        overlay_court(view, np.array(calibration["H_image_to_court"]))
        (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    cv2.line(view, tuple(np.rint(points[8]).astype(int)), tuple(np.rint(points[9]).astype(int)), (255, 0, 255), 2)
    for name, xy in zip(POINT_NAMES, points):
        x, y = map(round, xy)
        color = (255, 0, 255) if name in ("NL-TOP", "NR-TOP") else (0, 0, 255)
        cv2.circle(view, (x, y), 5, color, -1)
        cv2.putText(view, name, (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if not cv2.imwrite(str(args.output / "prediction.jpg"), view):
        raise OSError("Cannot write prediction image")
    (args.output / "points.json").write_text(json.dumps({"points": dict(zip(POINT_NAMES, points.tolist())),
                                                        "checkpoint_mode": checkpoint["mode"], "warnings": warnings}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}. {'Court fit created.' if calibration else 'No valid court fit; inspect point predictions.'}")
    if checkpoint["mode"] == "pilot_overfit_not_held_out":
        print("This checkpoint is a pilot memorisation run, not a validated detector.")
    elif checkpoint["mode"] == "training_only_not_validated":
        print("This checkpoint has not yet been validated on different courts.")


def service_regions():
    """Singles service boxes in floor coordinates; actual server side needs rally context."""
    from .court_model import X_LINES, Y_LINES
    return {f"{end}_{side}": {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}
            for end, y0, y1 in (("near", 0., Y_LINES["SS0"]), ("far", Y_LINES["SS1"], COURT_LENGTH))
            for side, x0, x1 in (("left", X_LINES["SL"], X_LINES["C"]), ("right", X_LINES["C"], X_LINES["SR"]))}


def tint_service_regions(frame, H_image_to_court):
    """Show the four singles service regions; no inference of server or score."""
    H = np.linalg.inv(H_image_to_court)
    tint = frame.copy()
    for name, box in service_regions().items():
        polygon = apply_homography(H, [[box["x_min"], box["y_min"]], [box["x_max"], box["y_min"]],
                                       [box["x_max"], box["y_max"]], [box["x_min"], box["y_max"]]])
        color = (230, 130, 40) if name.endswith("left") else (40, 210, 230)
        cv2.fillConvexPoly(tint, np.rint(polygon).astype(np.int32), color)
    cv2.addWeighted(tint, 0.12, frame, 0.88, 0, dst=frame)
    for name, box in service_regions().items():
        centre = apply_homography(H, [(box["x_min"]+box["x_max"])/2, (box["y_min"]+box["y_max"])/2])
        cv2.putText(frame, name.upper(), tuple(np.rint(centre).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 2, cv2.LINE_AA)
