"""Train the TCN phase classifier on annotated clips.

Pipeline (see module docstrings in annotation.py / tcn.py / classifier.py):

    annotations + per-clip pose JSONL
        -> quality gate (drop clips whose pose tracking mostly failed)
        -> windowed samples (build_windowed_training_dataset)
        -> split BY CLIP (hold out whole clips for validation)
        -> train TCN, report per-class precision/recall + confusion matrix
        -> save checkpoint

Design choices for the small-data regime (~15 clips):
- Split by clip, never by window: windows from the same clip are temporally
  near-duplicates, so a random window split would leak and inflate accuracy.
- No class weighting yet: we first REPORT the class imbalance (per-class recall +
  support) so it is visible before we decide how to correct it.
- Clips below a pose-survival threshold are excluded automatically (e.g. a
  frame where the tracker locked onto a spectator), logged explicitly. When the
  upstream tracker is fixed and a clip regenerates, it rejoins with no code
  change.

Run::

    .\\.venv\\Scripts\\python.exe -m src.temporal.train \\
        --annotations data/annotations --clips data/shuttle/clips \\
        --window-size 30 --epochs 60

torch is imported lazily so importing this module (e.g. for the data-assembly
unit tests) does not require torch.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.temporal import schema
from src.temporal.annotation import AnnotationSet, build_windowed_training_dataset
from src.temporal.contracts import load_pose_jsonl


# ---------------------------------------------------------------------------
# Data assembly (pure logic -- unit tested, no torch)
# ---------------------------------------------------------------------------

@dataclass
class ClipData:
    clip_id: str
    X: List[List[List[float]]]      # windowed samples: [n, window, feature_len]
    y: List[int]                    # phase indices
    raw_poses: int = 0
    valid_poses: int = 0

    @property
    def survival(self) -> float:
        return (self.valid_poses / self.raw_poses) if self.raw_poses else 0.0


def discover_clips(annotations_dir: str, clips_dir: str) -> List[str]:
    """Clip ids that have BOTH an annotation file and a pose JSONL."""
    ids = []
    for p in sorted(glob.glob(os.path.join(annotations_dir, "*.jsonl"))):
        cid = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join(clips_dir, cid, "pose.jsonl")):
            ids.append(cid)
    return ids


def load_clip(
    clip_id: str,
    annotations_dir: str,
    clips_dir: str,
    window_size: int,
    stride: int,
    label_from: str,
) -> ClipData:
    """Load one clip's pose + annotation into windowed samples, recording the
    pose-survival stats used by the quality gate."""
    anno_path = os.path.join(annotations_dir, f"{clip_id}.jsonl")
    pose_path = os.path.join(clips_dir, clip_id, "pose.jsonl")

    raw = sum(1 for ln in open(pose_path, encoding="utf-8") if ln.strip())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # suppress per-line skip warnings
        poses = load_pose_jsonl(pose_path)

    anno = AnnotationSet.load(anno_path)
    X, y = build_windowed_training_dataset(
        anno, poses, clip_id, window_size=window_size, stride=stride,
        label_from=label_from,
    )
    return ClipData(clip_id, X, y, raw_poses=raw, valid_poses=len(poses))


@dataclass
class Dataset:
    train_clips: List[ClipData]
    val_clips: List[ClipData]
    excluded: List[Tuple[str, str]] = field(default_factory=list)  # (clip, reason)

    def _stack(self, clips):
        X, y = [], []
        for c in clips:
            X.extend(c.X)
            y.extend(c.y)
        return X, y

    def train_xy(self):
        return self._stack(self.train_clips)

    def val_xy(self):
        return self._stack(self.val_clips)


def load_kept_clips(
    annotations_dir: str,
    clips_dir: str,
    window_size: int,
    stride: int = 1,
    label_from: str = "last",
    min_survival: float = 0.5,
) -> Tuple[List[ClipData], List[Tuple[str, str]]]:
    """Discover, window, and quality-gate clips. Returns (kept, excluded).

    Shared by both the single-split ``assemble_dataset`` and ``cross_validate``
    so the discovery/gating rules can never drift between them.
    """
    clip_ids = discover_clips(annotations_dir, clips_dir)
    if not clip_ids:
        raise ValueError(
            f"no clips with both an annotation and a pose.jsonl under "
            f"{annotations_dir} / {clips_dir}"
        )

    kept: List[ClipData] = []
    excluded: List[Tuple[str, str]] = []
    for cid in clip_ids:
        clip = load_clip(cid, annotations_dir, clips_dir, window_size, stride, label_from)
        if clip.survival < min_survival:
            excluded.append(
                (cid, f"pose survival {clip.survival:.0%} < {min_survival:.0%}")
            )
            continue
        if not clip.X:
            excluded.append((cid, "no training windows (no annotated+posed frames)"))
            continue
        kept.append(clip)

    if not kept:
        raise ValueError("every clip was excluded; nothing to train on")
    return kept, excluded


def assemble_dataset(
    annotations_dir: str,
    clips_dir: str,
    window_size: int,
    stride: int = 1,
    label_from: str = "last",
    min_survival: float = 0.5,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Dataset:
    """Discover, quality-gate, window, and split clips into train/val.

    The split is BY CLIP: whole clips are assigned to train or val.
    """
    kept, excluded = load_kept_clips(
        annotations_dir, clips_dir, window_size, stride, label_from, min_survival
    )

    # Split by clip, seeded for reproducibility.
    rng = random.Random(seed)
    shuffled = kept[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_clips = shuffled[:n_val]
    train_clips = shuffled[n_val:]
    return Dataset(train_clips, val_clips, excluded)


def make_folds(clips: List[ClipData], k: int, seed: int = 0) -> List[List[ClipData]]:
    """Partition clips into ``k`` roughly-equal folds, by clip (seeded).

    Returns a list of ``k`` clip lists. Each is used once as the validation
    fold while the rest train. Requires ``k`` >= 2 and at most one fold per clip.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > len(clips):
        raise ValueError(f"k={k} exceeds the number of usable clips ({len(clips)})")
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    folds: List[List[ClipData]] = [[] for _ in range(k)]
    for i, clip in enumerate(shuffled):
        folds[i % k].append(clip)
    return folds


# ---------------------------------------------------------------------------
# Metrics (hand-rolled -- no sklearn dependency)
# ---------------------------------------------------------------------------

def inverse_frequency_weights(y: List[int], num_classes: int) -> List[float]:
    """Per-class loss weights = inverse of class frequency, mean-normalised.

    A class with fewer samples gets a proportionally larger weight, so the model
    is penalised more for ignoring rare classes (e.g. ``idle``). Weights are
    normalised to average 1.0 so the overall loss scale is unchanged. A class
    absent from ``y`` gets weight 0 (it cannot be learned from this split, and 0
    avoids a divide-by-zero / inflating a class the model never sees).

    Computed from TRAINING labels only -- never validation.
    """
    counts = [0] * num_classes
    for t in y:
        counts[t] += 1
    present = [c for c in counts if c > 0]
    if not present:
        return [0.0] * num_classes
    raw = [(1.0 / c if c > 0 else 0.0) for c in counts]
    # Normalise so the mean of the non-zero weights is 1.0.
    nonzero = [w for w in raw if w > 0]
    scale = len(nonzero) / sum(nonzero)
    return [w * scale for w in raw]


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int):
    m = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def per_class_metrics(cm) -> List[Dict[str, float]]:
    """precision/recall/support per class from a confusion matrix."""
    n = len(cm)
    out = []
    for c in range(n):
        tp = cm[c][c]
        fn = sum(cm[c]) - tp
        fp = sum(cm[r][c] for r in range(n)) - tp
        support = sum(cm[c])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out.append({"precision": precision, "recall": recall, "support": support})
    return out


def format_report(cm) -> str:
    phases = schema.PHASES
    metrics = per_class_metrics(cm)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    acc = correct / total if total else 0.0

    lines = []
    lines.append(f"{'phase':12} {'precision':>9} {'recall':>7} {'support':>8}")
    for phase, m in zip(phases, metrics):
        lines.append(
            f"{phase:12} {m['precision']:>9.2f} {m['recall']:>7.2f} {int(m['support']):>8}"
        )
    lines.append(f"\naccuracy: {acc:.3f}  ({correct}/{total})")

    # Confusion matrix (rows = true, cols = pred).
    lines.append("\nconfusion matrix (rows=true, cols=pred):")
    header = " " * 12 + "".join(f"{p[:8]:>9}" for p in phases)
    lines.append(header)
    for i, phase in enumerate(phases):
        lines.append(f"{phase:12}" + "".join(f"{cm[i][j]:>9}" for j in range(len(phases))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training (torch -- lazy imported)
# ---------------------------------------------------------------------------

def train(
    annotations_dir: str = "data/annotations",
    clips_dir: str = "data/shuttle/clips",
    window_size: int = 30,
    stride: int = 1,
    label_from: str = "last",
    min_survival: float = 0.5,
    val_fraction: float = 0.2,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    patience: int = 10,
    seed: int = 0,
    checkpoint: str = "outputs/phase_tcn.pt",
    device: str = "cpu",
    use_velocity: bool = True,
    class_weights: bool = False,
) -> Dict:
    import torch

    torch.manual_seed(seed)

    data = assemble_dataset(
        annotations_dir, clips_dir, window_size, stride, label_from,
        min_survival, val_fraction, seed,
    )

    print("=== clips ===")
    for cid, reason in data.excluded:
        print(f"  EXCLUDED {cid}: {reason}")
    print(f"  train: {[c.clip_id for c in data.train_clips]}")
    print(f"  val:   {[c.clip_id for c in data.val_clips]}")

    n_valid_clips = len(data.train_clips) + len(data.val_clips)
    print(f"  no. valid clips: {n_valid_clips}")

    Xtr, ytr = data.train_xy()
    Xva, yva = data.val_xy()
    print(f"  train windows: {len(Xtr)}  val windows: {len(Xva)}  "
          f"velocity: {use_velocity}  weight decay: {weight_decay}  "
          f"class weights: {class_weights}")
    if not Xtr:
        raise ValueError("no training windows after assembly")

    clf, cm = _fit_and_evaluate(
        Xtr, ytr, Xva, yva,
        use_velocity=use_velocity, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, epochs=epochs, patience=patience,
        class_weights=class_weights, device=device, verbose=True,
    )

    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    clf.save(checkpoint)
    print(f"\nsaved checkpoint -> {checkpoint}")

    report = None
    if cm is not None:
        report = format_report(cm)
        print("\n=== validation report ===")
        print(report)
    else:
        print("\n(no validation clips -- skipping report)")

    return {
        "train_windows": len(Xtr),
        "val_windows": len(Xva),
        "excluded": data.excluded,
        "checkpoint": checkpoint,
        "report": report,
    }


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def format_cv_report(fold_cms: List[List[List[int]]]) -> str:
    """Summarise k folds: per-fold accuracy, plus mean +/- std of accuracy and
    per-class recall across folds, and an aggregate (summed) confusion matrix.
    """
    phases = schema.PHASES
    n = len(phases)

    fold_acc: List[float] = []
    fold_recall: Dict[str, List[float]] = {p: [] for p in phases}
    agg = [[0] * n for _ in range(n)]

    for cm in fold_cms:
        total = sum(sum(r) for r in cm)
        correct = sum(cm[i][i] for i in range(n))
        fold_acc.append(correct / total if total else 0.0)
        for c in range(n):
            support = sum(cm[c])
            recall = cm[c][c] / support if support else float("nan")
            if support:
                fold_recall[phases[c]].append(recall)
        for i in range(n):
            for j in range(n):
                agg[i][j] += cm[i][j]

    lines = [f"=== {len(fold_cms)}-fold cross-validation ==="]
    acc_m, acc_s = _mean_std(fold_acc)
    lines.append("per-fold accuracy: " + ", ".join(f"{a:.3f}" for a in fold_acc))
    lines.append(f"accuracy: {acc_m:.3f} +/- {acc_s:.3f}")
    lines.append("")
    lines.append(f"{'phase':12} {'recall mean':>12} {'std':>7} {'folds':>7}")
    for p in phases:
        vals = fold_recall[p]
        m, s = _mean_std(vals)
        lines.append(f"{p:12} {m:>12.3f} {s:>7.3f} {len(vals):>7}")
    lines.append("")
    lines.append("aggregate confusion matrix (summed over folds; rows=true, cols=pred):")
    header = " " * 12 + "".join(f"{p[:8]:>9}" for p in phases)
    lines.append(header)
    for i, p in enumerate(phases):
        lines.append(f"{p:12}" + "".join(f"{agg[i][j]:>9}" for j in range(n)))
    return "\n".join(lines)


def cross_validate(
    annotations_dir: str = "data/annotations",
    clips_dir: str = "data/shuttle/clips",
    k: int = 5,
    window_size: int = 30,
    stride: int = 1,
    label_from: str = "last",
    min_survival: float = 0.5,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    patience: int = 10,
    seed: int = 0,
    device: str = "cpu",
    use_velocity: bool = True,
    class_weights: bool = False,
) -> Dict:
    """k-fold cross-validation at the CLIP level.

    Each fold holds out a disjoint set of whole clips for validation and trains
    on the rest, reusing ``_fit_and_evaluate`` (identical training to ``train``).
    Reports mean +/- std across folds -- the trustworthy signal that single
    splits (which we've seen swing widely on this small data) cannot give.

    No checkpoint is saved: CV is for MEASUREMENT. Use ``train`` to fit the
    model you ship.
    """
    import torch

    torch.manual_seed(seed)
    kept, excluded = load_kept_clips(
        annotations_dir, clips_dir, window_size, stride, label_from, min_survival
    )
    print("=== clips ===")
    for cid, reason in excluded:
        print(f"  EXCLUDED {cid}: {reason}")
    print(f"  usable clips: {len(kept)}  (k={k})")

    folds = make_folds(kept, k, seed)
    fold_cms: List[List[List[int]]] = []

    for fi, val_clips in enumerate(folds, start=1):
        train_clips = [c for c in kept if c not in val_clips]
        Xtr, ytr = [], []
        for c in train_clips:
            Xtr.extend(c.X); ytr.extend(c.y)
        Xva, yva = [], []
        for c in val_clips:
            Xva.extend(c.X); yva.extend(c.y)

        print(f"\n--- fold {fi}/{k} ---")
        print(f"  val clips: {[c.clip_id for c in val_clips]}")
        print(f"  train windows: {len(Xtr)}  val windows: {len(Xva)}")
        if not Xtr or not Xva:
            print("  skipped (empty train or val)")
            continue

        _clf, cm = _fit_and_evaluate(
            Xtr, ytr, Xva, yva,
            use_velocity=use_velocity, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, epochs=epochs, patience=patience,
            class_weights=class_weights, device=device, verbose=False,
        )
        if cm is not None:
            fold_cms.append(cm)

    if not fold_cms:
        raise ValueError("no folds produced a validation result")

    report = format_cv_report(fold_cms)
    print("\n" + report)
    return {"k": k, "fold_confusion_matrices": fold_cms, "report": report}


def _fit_and_evaluate(
    Xtr, ytr, Xva, yva, *,
    use_velocity: bool, lr: float, weight_decay: float, batch_size: int,
    epochs: int, patience: int, class_weights: bool, device: str,
    verbose: bool = False,
):
    """Train one TCN on (Xtr,ytr), early-stopping on (Xva,yva); return
    ``(classifier, val_confusion_matrix_or_None)``.

    This is the single fit+eval core shared by ``train`` and ``cross_validate``
    so the two can never diverge in how a model is trained or scored.
    Inputs are COORDINATE windows; the velocity transform is applied here.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.temporal.classifier import TrainableClassifier
    from src.temporal.features import transform_window

    Xtr_feat = [transform_window(w, use_velocity) for w in Xtr]
    Xtr_t = torch.tensor(Xtr_feat, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True
    )

    has_val = len(Xva) > 0
    if has_val:
        Xva_feat = [transform_window(w, use_velocity) for w in Xva]
        Xva_t = torch.tensor(Xva_feat, dtype=torch.float32)
        yva_t = torch.tensor(yva, dtype=torch.long)

    clf = TrainableClassifier.from_new(use_velocity=use_velocity, device=device)
    model = clf.model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    weight_tensor = None
    if class_weights:
        w = inverse_frequency_weights(ytr, len(schema.PHASES))
        weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        if verbose:
            print("  class weights: " + ", ".join(
                f"{p}={wi:.2f}" for p, wi in zip(schema.PHASES, w)
            ))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(Xtr)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xva_t.to(device)), yva_t.to(device)).item()
            marker = ""
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                marker = " *"
            else:
                epochs_no_improve += 1
            if verbose:
                print(f"epoch {epoch:3}  train {train_loss:.4f}  val {val_loss:.4f}{marker}")
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(no val improvement in {patience})")
                break
        elif verbose:
            print(f"epoch {epoch:3}  train {train_loss:.4f}  (no val set)")

    if best_state is not None:
        model.load_state_dict(best_state)

    cm = None
    if has_val:
        model.eval()
        with torch.no_grad():
            preds = model(Xva_t.to(device)).argmax(dim=1).tolist()
        cm = confusion_matrix(yva, preds, len(schema.PHASES))
    return clf, cm


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train the TCN phase classifier")
    p.add_argument("--annotations", default="data/annotations")
    p.add_argument("--clips", default="data/shuttle/clips")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--label-from", default="last", choices=["last", "center", "majority"])
    p.add_argument("--min-survival", type=float, default=0.5,
                   help="exclude clips whose pose-survival rate is below this")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="outputs/phase_tcn.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-velocity", dest="velocity", action="store_false",
                   help="use coordinates only (disable per-keypoint velocity)")
    p.set_defaults(velocity=True)
    p.add_argument("--class-weights", action="store_true",
                   help="weight the loss by inverse class frequency (train "
                        "split) so rare phases like 'idle' are not ignored")
    p.add_argument("--cross-validate", action="store_true",
                   help="run k-fold cross-validation (measurement) instead of a "
                        "single train/val split; saves no checkpoint")
    p.add_argument("--k", type=int, default=5, help="number of CV folds")
    args = p.parse_args(argv)

    if args.cross_validate:
        cross_validate(
            annotations_dir=args.annotations, clips_dir=args.clips, k=args.k,
            window_size=args.window_size, stride=args.stride, label_from=args.label_from,
            min_survival=args.min_survival, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            patience=args.patience, seed=args.seed, device=args.device,
            use_velocity=args.velocity, class_weights=args.class_weights,
        )
        return

    train(
        annotations_dir=args.annotations, clips_dir=args.clips,
        window_size=args.window_size, stride=args.stride, label_from=args.label_from,
        min_survival=args.min_survival, val_fraction=args.val_fraction,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        patience=args.patience, seed=args.seed, checkpoint=args.checkpoint,
        use_velocity=args.velocity, class_weights=args.class_weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()
