"""Tests for the torch-free data-assembly and metrics in train.py."""

import json
import os

import pytest

from src.temporal import schema
from src.temporal.contracts import KEYPOINT_NAMES
from src.temporal.train import (
    assemble_dataset,
    confusion_matrix,
    discover_clips,
    format_report,
    per_class_metrics,
)


def _write_annotation(path, clip_id, segments):
    with open(path, "w", encoding="utf-8") as fh:
        for pid, s, e, label in segments:
            fh.write(json.dumps({
                "clip_id": clip_id, "player_id": pid,
                "start_frame": s, "end_frame": e, "label": label,
            }) + "\n")


def _write_pose(path, n_frames, bbox=(0.0, 0.0, 200.0, 400.0),
                player_id=1, in_box=True):
    """Write a merged pose JSONL. If in_box is False, place a keypoint far
    outside the bbox so the contract rejects the record (simulating a bad
    track)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x1, y1, x2, y2 = bbox
    with open(path, "w", encoding="utf-8") as fh:
        for f in range(n_frames):
            kps = []
            for i, name in enumerate(KEYPOINT_NAMES):
                px = x1 + 10 + (i % 5)
                py = y1 + 10 + (i % 5)
                if not in_box and name == "left_hip":
                    py = y2 + 500  # far below the box -> normalizes well past 1
                kps.append({"name": name, "x": px, "y": py,
                            "visibility": 0.9, "presence": 0.9})
            fh.write(json.dumps({
                "frame": f, "player_id": player_id, "bbox": list(bbox),
                "confidence": 0.9, "keypoints": kps,
            }) + "\n")


def _make_clip(root, clip_id, n_frames, label="stroke", in_box=True):
    anno_dir = os.path.join(root, "annotations")
    clips_dir = os.path.join(root, "clips")
    os.makedirs(anno_dir, exist_ok=True)
    _write_annotation(
        os.path.join(anno_dir, f"{clip_id}.jsonl"),
        clip_id, [(1, 0, n_frames - 1, label)],
    )
    _write_pose(os.path.join(clips_dir, clip_id, "pose.jsonl"), n_frames, in_box=in_box)


# --- discovery --------------------------------------------------------------

def test_discover_requires_both_annotation_and_pose(tmp_path):
    root = str(tmp_path)
    _make_clip(root, "rallyA", 40)
    # Annotation without a pose file -> not discovered.
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    _write_annotation(os.path.join(root, "annotations", "rallyB.jsonl"),
                      "rallyB", [(1, 0, 5, "ready")])
    found = discover_clips(os.path.join(root, "annotations"),
                           os.path.join(root, "clips"))
    assert found == ["rallyA"]


# --- quality gate + split ---------------------------------------------------

def test_assemble_excludes_low_survival_clip(tmp_path):
    root = str(tmp_path)
    _make_clip(root, "good1", 60, in_box=True)
    _make_clip(root, "good2", 60, in_box=True)
    _make_clip(root, "bad", 60, in_box=False)   # all records fail contract

    data = assemble_dataset(
        os.path.join(root, "annotations"), os.path.join(root, "clips"),
        window_size=10, min_survival=0.5, val_fraction=0.5, seed=0,
    )
    excluded_ids = [cid for cid, _ in data.excluded]
    assert "bad" in excluded_ids
    all_kept = [c.clip_id for c in data.train_clips + data.val_clips]
    assert set(all_kept) == {"good1", "good2"}


def test_assemble_splits_by_clip_not_window(tmp_path):
    root = str(tmp_path)
    for i in range(5):
        _make_clip(root, f"r{i}", 50)
    data = assemble_dataset(
        os.path.join(root, "annotations"), os.path.join(root, "clips"),
        window_size=10, val_fraction=0.2, seed=0,
    )
    train_ids = {c.clip_id for c in data.train_clips}
    val_ids = {c.clip_id for c in data.val_clips}
    # No clip appears in both splits.
    assert train_ids.isdisjoint(val_ids)
    # 5 clips, 20% -> 1 val, 4 train.
    assert len(val_ids) == 1 and len(train_ids) == 4


def test_assemble_raises_when_all_excluded(tmp_path):
    root = str(tmp_path)
    _make_clip(root, "bad", 60, in_box=False)
    with pytest.raises(ValueError):
        assemble_dataset(os.path.join(root, "annotations"),
                         os.path.join(root, "clips"), window_size=10)


# --- multi-player (canonical two-slot) data --------------------------------

def _write_multiplayer_pose(path, n_frames, extra_rejected=0):
    """Write a two-slot pose JSONL: slots 1 and 2 each appear every frame, in
    DIFFERENT bboxes. Optionally append ``extra_rejected`` records with
    player_id == -1 (canonicalization strays) to confirm they are dropped.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    boxes = {1: (0.0, 0.0, 200.0, 400.0), 2: (600.0, 0.0, 800.0, 400.0)}
    with open(path, "w", encoding="utf-8") as fh:
        for f in range(n_frames):
            for pid, (x1, y1, x2, y2) in boxes.items():
                kps = [{"name": name, "x": x1 + 10 + (i % 5),
                        "y": y1 + 10 + (i % 5), "visibility": 0.9, "presence": 0.9}
                       for i, name in enumerate(KEYPOINT_NAMES)]
                fh.write(json.dumps({
                    "frame": f, "player_id": pid, "bbox": [x1, y1, x2, y2],
                    "confidence": 0.9, "keypoints": kps,
                }) + "\n")
        for j in range(extra_rejected):
            x1, y1, x2, y2 = (300.0, 0.0, 500.0, 400.0)
            kps = [{"name": name, "x": x1 + 10 + (i % 5),
                    "y": y1 + 10 + (i % 5), "visibility": 0.9, "presence": 0.9}
                   for i, name in enumerate(KEYPOINT_NAMES)]
            fh.write(json.dumps({
                "frame": j, "player_id": -1, "bbox": [x1, y1, x2, y2],
                "confidence": 0.9, "keypoints": kps,
            }) + "\n")


def _make_multiplayer_clip(root, clip_id, n_frames, extra_rejected=0):
    anno_dir = os.path.join(root, "annotations")
    clips_dir = os.path.join(root, "clips")
    os.makedirs(anno_dir, exist_ok=True)
    # Both slots annotated, with DIFFERENT labels per slot across the whole clip.
    _write_annotation(
        os.path.join(anno_dir, f"{clip_id}.jsonl"),
        clip_id,
        [(1, 0, n_frames - 1, "stroke"), (2, 0, n_frames - 1, "ready")],
    )
    _write_multiplayer_pose(
        os.path.join(clips_dir, clip_id, "pose.jsonl"), n_frames, extra_rejected)


def test_auto_detect_multiplayer_builds_per_slot_windows(tmp_path):
    from src.temporal.train import load_clip

    root = str(tmp_path)
    _make_multiplayer_clip(root, "duo", 40)
    # single_player=None -> auto-detect; two slots -> multi-player join.
    clip = load_clip("duo", os.path.join(root, "annotations"),
                     os.path.join(root, "clips"),
                     window_size=10, stride=1, label_from="last",
                     single_player=None)
    # Both slots annotated with different phases -> both stroke and ready
    # windows are produced (two independent streams, not one mixed stream).
    phases = {schema.PHASES[i] for i in clip.y}
    assert phases == {"stroke", "ready"}
    # 40 frames per slot, window 10 -> 31 windows per slot, 62 total.
    assert len(clip.X) == 62


def test_single_player_mode_rejects_multiplayer_clip(tmp_path):
    from src.temporal.train import load_clip

    root = str(tmp_path)
    _make_multiplayer_clip(root, "duo", 40)
    # Forcing single-player on a two-annotated-player clip is ambiguous and must
    # raise rather than silently mislabel (annotation.single_player_frame_labels).
    with pytest.raises(ValueError):
        load_clip("duo", os.path.join(root, "annotations"),
                  os.path.join(root, "clips"),
                  window_size=10, stride=1, label_from="last",
                  single_player=True)


def test_rejected_slot_records_are_dropped(tmp_path):
    from src.temporal.train import load_clip

    root = str(tmp_path)
    # 20 real frames x 2 slots = 40 valid; plus 5 rejected (-1) strays.
    _make_multiplayer_clip(root, "duo", 20, extra_rejected=5)
    clip = load_clip("duo", os.path.join(root, "annotations"),
                     os.path.join(root, "clips"),
                     window_size=5, stride=1, label_from="last",
                     single_player=None)
    # valid_poses counts only non-rejected records: 20 frames x 2 slots.
    assert clip.valid_poses == 40


# --- court feature at the dataset level ------------------------------------

def _write_pose_with_court(path, n_frames, player_id=1,
                           bbox=(0.0, 0.0, 200.0, 400.0), court_xy=(3.0, 3.0)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x1, y1, _x2, _y2 = bbox
    with open(path, "w", encoding="utf-8") as fh:
        for f in range(n_frames):
            kps = [{"name": name, "x": x1 + 10 + (i % 5), "y": y1 + 10 + (i % 5),
                    "visibility": 0.9, "presence": 0.9}
                   for i, name in enumerate(KEYPOINT_NAMES)]
            fh.write(json.dumps({
                "frame": f, "player_id": player_id, "bbox": list(bbox),
                "confidence": 0.9, "court_xy": list(court_xy), "keypoints": kps,
            }) + "\n")


def test_use_court_grows_window_feature_length(tmp_path):
    from src.temporal.train import load_clip
    from src.temporal.features import base_feature_len

    root = str(tmp_path)
    anno_dir = os.path.join(root, "annotations")
    os.makedirs(anno_dir, exist_ok=True)
    _write_annotation(os.path.join(anno_dir, "c.jsonl"), "c",
                      [(1, 0, 39, "stroke")])
    _write_pose_with_court(os.path.join(root, "clips", "c", "pose.jsonl"), 40)

    # Without court: base length = 2*NUM_KEYPOINTS.
    clip_no = load_clip("c", anno_dir, os.path.join(root, "clips"),
                        window_size=10, stride=1, label_from="last",
                        single_player=True, use_court=False)
    assert len(clip_no.X[0][0]) == base_feature_len(use_court=False)

    # With court: each frame carries 2 extra channels.
    clip_ct = load_clip("c", anno_dir, os.path.join(root, "clips"),
                        window_size=10, stride=1, label_from="last",
                        single_player=True, use_court=True)
    assert len(clip_ct.X[0][0]) == base_feature_len(use_court=True)
    # Court tail is normalized court_xy (3.0/6.10, 3.0/13.40).
    assert clip_ct.X[0][0][-2:] == [3.0 / 6.10, 3.0 / 13.40]


def _write_pose_with_shuttle(path, n_frames, player_id=1,
                             bbox=(0.0, 0.0, 200.0, 400.0), speed=0.07, valid=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x1, y1, _x2, _y2 = bbox
    with open(path, "w", encoding="utf-8") as fh:
        for f in range(n_frames):
            kps = [{"name": name, "x": x1 + 10 + (i % 5), "y": y1 + 10 + (i % 5),
                    "visibility": 0.9, "presence": 0.9}
                   for i, name in enumerate(KEYPOINT_NAMES)]
            fh.write(json.dumps({
                "frame": f, "player_id": player_id, "bbox": list(bbox),
                "confidence": 0.9, "shuttle_speed": speed,
                "shuttle_speed_valid": valid, "keypoints": kps,
            }) + "\n")


def test_use_shuttle_grows_window_feature_length(tmp_path):
    from src.temporal.train import load_clip
    from src.temporal.features import base_feature_len

    root = str(tmp_path)
    anno_dir = os.path.join(root, "annotations")
    os.makedirs(anno_dir, exist_ok=True)
    _write_annotation(os.path.join(anno_dir, "c.jsonl"), "c",
                      [(1, 0, 39, "idle")])
    _write_pose_with_shuttle(os.path.join(root, "clips", "c", "pose.jsonl"), 40)

    clip = load_clip("c", anno_dir, os.path.join(root, "clips"),
                     window_size=10, stride=1, label_from="last",
                     single_player=True, use_shuttle=True)
    assert len(clip.X[0][0]) == base_feature_len(use_shuttle=True)
    # Shuttle tail is (speed, valid).
    assert clip.X[0][0][-2:] == [0.07, 1.0]


def test_assemble_is_reproducible_under_seed(tmp_path):
    root = str(tmp_path)
    for i in range(6):
        _make_clip(root, f"r{i}", 40)
    kw = dict(annotations_dir=os.path.join(root, "annotations"),
              clips_dir=os.path.join(root, "clips"), window_size=10, seed=7)
    a = assemble_dataset(**kw)
    b = assemble_dataset(**kw)
    assert [c.clip_id for c in a.val_clips] == [c.clip_id for c in b.val_clips]


# --- metrics ----------------------------------------------------------------

def test_confusion_matrix_counts():
    y_true = [0, 0, 1, 2]
    y_pred = [0, 1, 1, 2]
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm[0] == [1, 1, 0]
    assert cm[1] == [0, 1, 0]
    assert cm[2] == [0, 0, 1]


def test_per_class_metrics_precision_recall():
    # class 0: 1 tp, 1 fn (predicted as 1), 0 fp -> recall .5, precision 1.
    cm = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    m = per_class_metrics(cm)
    assert m[0]["recall"] == pytest.approx(0.5)
    assert m[0]["precision"] == pytest.approx(1.0)
    assert m[0]["support"] == 2


def test_format_report_runs_over_all_phases():
    n = len(schema.PHASES)
    cm = [[0] * n for _ in range(n)]
    for i in range(n):
        cm[i][i] = 3
    text = format_report(cm)
    for phase in schema.PHASES:
        assert phase in text
    assert "accuracy: 1.000" in text


# --- class weights ----------------------------------------------------------

from src.temporal.train import inverse_frequency_weights


def test_inverse_frequency_weights_rare_class_gets_more():
    # class 0: 90 samples, class 1: 10 samples -> class 1 weighted higher.
    y = [0] * 90 + [1] * 10
    w = inverse_frequency_weights(y, 2)
    assert w[1] > w[0]
    # Ratio tracks inverse frequency: (1/10)/(1/90) = 9.
    assert w[1] / w[0] == pytest.approx(9.0)


def test_inverse_frequency_weights_mean_normalised():
    y = [0] * 90 + [1] * 10
    w = inverse_frequency_weights(y, 2)
    # Mean of non-zero weights is 1.0.
    assert sum(w) / len(w) == pytest.approx(1.0)


def test_inverse_frequency_weights_absent_class_is_zero():
    # class 2 never appears -> weight 0, no divide-by-zero.
    y = [0, 0, 1]
    w = inverse_frequency_weights(y, 3)
    assert w[2] == 0.0
    assert all(wi >= 0 for wi in w)


def test_inverse_frequency_weights_empty_is_all_zero():
    assert inverse_frequency_weights([], 5) == [0.0] * 5


# --- k-fold helpers ---------------------------------------------------------

from src.temporal.train import ClipData, make_folds, format_cv_report, _mean_std


def _clip(cid):
    return ClipData(cid, X=[[[0.0]]], y=[0], raw_poses=1, valid_poses=1)


def test_make_folds_partitions_all_clips_disjointly():
    clips = [_clip(f"r{i}") for i in range(10)]
    folds = make_folds(clips, k=5, seed=0)
    assert len(folds) == 5
    all_ids = [c.clip_id for fold in folds for c in fold]
    # Every clip appears exactly once across all folds.
    assert sorted(all_ids) == sorted(c.clip_id for c in clips)
    assert len(all_ids) == len(set(all_ids))


def test_make_folds_balanced_sizes():
    clips = [_clip(f"r{i}") for i in range(11)]
    folds = make_folds(clips, k=5, seed=0)
    sizes = sorted(len(f) for f in folds)
    # 11 into 5 folds -> sizes differ by at most 1.
    assert sizes[-1] - sizes[0] <= 1


def test_make_folds_validates_k():
    clips = [_clip("a"), _clip("b")]
    for bad in (1, 3):
        try:
            make_folds(clips, k=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for k={bad}")


def test_make_folds_is_seeded():
    clips = [_clip(f"r{i}") for i in range(8)]
    a = [[c.clip_id for c in f] for f in make_folds(clips, 4, seed=3)]
    b = [[c.clip_id for c in f] for f in make_folds(clips, 4, seed=3)]
    assert a == b


def test_mean_std():
    m, s = _mean_std([1.0, 1.0, 1.0])
    assert m == pytest.approx(1.0) and s == pytest.approx(0.0)
    m, s = _mean_std([0.0, 1.0])
    assert m == pytest.approx(0.5) and s == pytest.approx(0.5)


def test_format_cv_report_aggregates_folds():
    n = len(schema.PHASES)
    # Two folds, each a perfect diagonal confusion matrix.
    cm = [[0] * n for _ in range(n)]
    for i in range(n):
        cm[i][i] = 4
    report = format_cv_report([cm, cm])
    assert "2-fold cross-validation" in report
    assert "accuracy: 1.000" in report
    for phase in schema.PHASES:
        assert phase in report
