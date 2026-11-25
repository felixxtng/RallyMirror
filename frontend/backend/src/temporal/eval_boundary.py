"""Boundary-aware and segment-level evaluation for phase predictions.

Frame-exact accuracy (what ``train.cross_validate`` reports) punishes a
prediction that is correct but a frame or two off at a phase TRANSITION. Since
the rally phases are adjacent in a cycle (ready -> movement -> stroke ->
recovery -> ready), most of the error concentrates at those boundaries, where
even a human annotator is unsure of the exact switch frame. These extra views
tell you whether the ~0.55 frame-exact number is a real ceiling or an artifact
of exact-frame scoring:

A. BOUNDARY-TOLERANT frame accuracy (+/-k): a frame counts correct if its
   prediction matches the true label at that frame OR at any true label within
   +/-k frames. Forgives sub-second jitter at transitions; still penalises a
   genuinely wrong phase.

B. SMOOTHED frame accuracy: apply the same causal majority filter the deployed
   ``PhasePipeline`` uses before scoring, so the number reflects what you would
   actually ship rather than the bare per-window model output.

C. SEGMENT-LEVEL detection: of the contiguous true phase segments, how many did
   the model "detect" (predict that phase for at least ``min_overlap`` of the
   segment's frames). Measures the thing you care about -- did it recognise the
   player was in recovery -- rather than frame-exact alignment.

All three operate on PER-CLIP TIMELINES: an ordered list of
``(frame, true_idx, pred_idx)`` for one validation clip. The caller
(``train``) builds these by predicting each val clip's windows and anchoring
each prediction at its window's label frame. Pure logic, no torch -- unit
testable.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

# A per-clip timeline: ordered (frame, true_idx, pred_idx) triples. Frames are
# the window label-anchor frames, strictly increasing within a clip.
Timeline = Sequence[Tuple[int, int, int]]


def frame_exact_accuracy(timelines: Sequence[Timeline]) -> float:
    """Plain per-frame accuracy over all timelines (sanity cross-check against
    the existing confusion-matrix accuracy)."""
    total = correct = 0
    for tl in timelines:
        for _f, t, p in tl:
            total += 1
            correct += (t == p)
    return correct / total if total else 0.0


def boundary_tolerant_accuracy(timelines: Sequence[Timeline], k: int = 3) -> float:
    """(A) A frame is correct if its prediction equals any TRUE label within
    +/-k frames on the same clip's timeline. ``k=0`` reduces to frame-exact.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    total = correct = 0
    for tl in timelines:
        true_by_frame = {f: t for f, t, _p in tl}
        for f, _t, p in tl:
            total += 1
            # Accept if p matches the true label at any frame in [f-k, f+k].
            hit = any(true_by_frame.get(f + d) == p for d in range(-k, k + 1)
                      if (f + d) in true_by_frame)
            correct += hit
    return correct / total if total else 0.0


def _smooth(preds: List[int], window: int) -> List[int]:
    """Causal majority filter matching ``PhasePipeline._smooth`` in spirit: each
    output is the majority over the trailing ``window`` predictions (ties ->
    the most recent). ``window<=1`` is a passthrough."""
    if window <= 1:
        return list(preds)
    out: List[int] = []
    recent: deque = deque(maxlen=window)
    for p in preds:
        recent.append(p)
        counts: Dict[int, int] = {}
        for q in recent:
            counts[q] = counts.get(q, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == p))
        out.append(best[0])
    return out


def smoothed_accuracy(timelines: Sequence[Timeline], window: int = 5) -> float:
    """(B) Frame accuracy after applying the deployed majority filter to each
    clip's prediction sequence."""
    total = correct = 0
    for tl in timelines:
        preds = _smooth([p for _f, _t, p in tl], window)
        for (_, t, _orig), p in zip(tl, preds):
            total += 1
            correct += (t == p)
    return correct / total if total else 0.0


def _segments(seq: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Contiguous runs of equal values -> list of (value, start_idx, end_idx]
    (end exclusive)."""
    runs: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            runs.append((seq[start], start, i))
            start = i
    return runs


def segment_detection(
    timelines: Sequence[Timeline],
    num_classes: int,
    min_overlap: float = 0.5,
    smooth_window: int = 1,
) -> Tuple[Dict[int, Tuple[int, int]], float]:
    """(C) Per-class segment detection rate.

    For each contiguous TRUE segment, it is "detected" if the predictions over
    that segment's frames match its class for at least ``min_overlap`` of them.
    Optionally smooth predictions first (``smooth_window``) to reflect the
    deployed output. Returns ``({class: (detected, total)}, overall_rate)``.
    """
    per_class: Dict[int, List[int]] = {c: [0, 0] for c in range(num_classes)}
    for tl in timelines:
        trues = [t for _f, t, _p in tl]
        preds = _smooth([p for _f, _t, p in tl], smooth_window)
        for cls, s, e in _segments(trues):
            hits = sum(1 for i in range(s, e) if preds[i] == cls)
            per_class[cls][1] += 1
            if (e - s) > 0 and hits / (e - s) >= min_overlap:
                per_class[cls][0] += 1
    det = sum(v[0] for v in per_class.values())
    tot = sum(v[1] for v in per_class.values())
    rate = det / tot if tot else 0.0
    return {c: (v[0], v[1]) for c, v in per_class.items()}, rate


def format_boundary_report(
    timelines: Sequence[Timeline],
    phase_names: Sequence[str],
    ks: Sequence[int] = (0, 2, 5),
    smooth_window: int = 5,
    seg_min_overlap: float = 0.5,
) -> str:
    """Render all three views into one text block for the CV report."""
    num_classes = len(phase_names)
    lines = ["=== boundary-aware evaluation (aggregated over folds) ==="]

    lines.append("frame accuracy, boundary-tolerant (+/-k frames):")
    for k in ks:
        acc = boundary_tolerant_accuracy(timelines, k)
        tag = " (frame-exact)" if k == 0 else ""
        lines.append(f"  k={k:<2} {acc:.3f}{tag}")

    sm = smoothed_accuracy(timelines, smooth_window)
    lines.append(f"frame accuracy, smoothed (majority window={smooth_window}): {sm:.3f}")

    per_class, overall = segment_detection(
        timelines, num_classes, seg_min_overlap, smooth_window)
    lines.append(f"\nsegment detection (>= {seg_min_overlap:.0%} overlap, "
                 f"smoothed window={smooth_window}):")
    lines.append(f"{'phase':12} {'detected':>9} {'total':>7} {'rate':>7}")
    for c, name in enumerate(phase_names):
        d, t = per_class[c]
        rate = d / t if t else 0.0
        lines.append(f"{name:12} {d:>9} {t:>7} {rate:>7.3f}")
    lines.append(f"{'OVERALL':12} {'':>9} {'':>7} {overall:>7.3f}")
    return "\n".join(lines)
