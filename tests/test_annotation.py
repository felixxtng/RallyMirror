"""Tests for the segment annotation data layer and training-dataset join."""

import pytest

from src.temporal import schema
from src.temporal.annotation import (
    AnnotationSet,
    Segment,
    build_training_dataset,
)
from src.temporal.contracts import NUM_KEYPOINTS


def _pose_rec(frame, player_id=1, val=0.5):
    return {
        "frame": frame,
        "player_id": player_id,
        "keypoints": [[val, val] for _ in range(NUM_KEYPOINTS)],
    }


# --- Segment ---------------------------------------------------------------

def test_segment_rejects_unknown_label():
    with pytest.raises(ValueError):
        Segment("clip", 1, 0, 10, "not_a_label")


def test_segment_rejects_reversed_frames():
    with pytest.raises(ValueError):
        Segment("clip", 1, 20, 10, "stroke")


def test_segment_valid_label_from_schema():
    seg = Segment("clip", 1, 0, 5, schema.ANNOTATION_LABELS[0])
    assert list(seg.frames()) == [0, 1, 2, 3, 4, 5]  # inclusive


# --- AnnotationSet ---------------------------------------------------------

def test_add_rejects_overlap_same_clip_player():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 10, "stroke"))
    with pytest.raises(ValueError):
        anno.add(Segment("c", 1, 5, 15, "movement"))  # overlaps 5-10


def test_add_allows_overlap_different_player():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 10, "stroke"))
    anno.add(Segment("c", 2, 5, 15, "movement"))  # different player, ok
    assert len(anno) == 2


def test_add_allows_adjacent_non_overlapping():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 10, "stroke"))
    anno.add(Segment("c", 1, 11, 20, "recovery"))
    assert len(anno) == 2


def test_save_load_round_trip(tmp_path):
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 10, "stroke"))
    anno.add(Segment("c", 1, 11, 20, "movement"))
    p = tmp_path / "anno.jsonl"
    anno.save(str(p))
    loaded = AnnotationSet.load(str(p))
    assert len(loaded) == 2
    assert loaded.segments[0].label == "stroke"


def test_load_enforces_overlap_check(tmp_path):
    p = tmp_path / "anno.jsonl"
    p.write_text(
        '{"clip_id": "c", "player_id": 1, "start_frame": 0, "end_frame": 10, "label": "stroke"}\n'
        '{"clip_id": "c", "player_id": 1, "start_frame": 5, "end_frame": 9, "label": "movement"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        AnnotationSet.load(str(p))


def test_to_frame_labels_expands_inclusive():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 2, 4, "stroke"))
    fl = anno.to_frame_labels("c", 1)
    assert fl == {2: "stroke", 3: "stroke", 4: "stroke"}


def test_label_counts():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "stroke"))      # 10 frames
    anno.add(Segment("c", 1, 10, 14, "movement"))  # 5 frames
    assert anno.label_counts() == {"stroke": 10, "movement": 5}


# --- build_training_dataset ------------------------------------------------

def test_build_training_dataset_joins_and_projects():
    anno = AnnotationSet()
    # stroke phase on frames 0-2, idle phase on frames 5-6.
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    anno.add(Segment("c", 1, 5, 6, "idle"))
    poses = [_pose_rec(f) for f in range(8)]  # frames 0..7

    X, y = build_training_dataset(anno, poses, clip_id="c")

    # Only annotated frames (0,1,2,5,6) are included -> 5 samples.
    assert len(X) == 5 and len(y) == 5
    assert all(len(vec) == 2 * NUM_KEYPOINTS for vec in X)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    idle = schema.PHASE_TO_INDEX[schema.PHASE_IDLE]
    assert y == [stroke, stroke, stroke, idle, idle]


def test_build_training_dataset_skips_unlabeled_and_wrong_clip():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 1, "stroke"))
    poses = [_pose_rec(0), _pose_rec(1), _pose_rec(2)]  # frame 2 unlabeled
    X, y = build_training_dataset(anno, poses, clip_id="c")
    assert len(X) == 2
    # Different clip id -> nothing matches.
    X2, y2 = build_training_dataset(anno, poses, clip_id="other")
    assert X2 == [] and y2 == []


def test_build_training_dataset_separates_players():
    # Multi-player (doubles-style) join: match on player_id -> single_player=False.
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 1, "stroke"))
    anno.add(Segment("c", 2, 0, 1, "idle"))
    poses = [_pose_rec(0, player_id=1), _pose_rec(0, player_id=2)]
    X, y = build_training_dataset(anno, poses, clip_id="c", single_player=False)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    idle = schema.PHASE_TO_INDEX[schema.PHASE_IDLE]
    assert sorted(y) == sorted([stroke, idle])


# --- build_windowed_training_dataset ---------------------------------------

from src.temporal.annotation import build_windowed_training_dataset
from src.temporal.contracts import pose_record_to_vector


def test_windowed_shape_and_count():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "stroke"))
    poses = [_pose_rec(f) for f in range(10)]  # frames 0..9, all labeled
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    # 10 frames, window 3, stride 1 -> 8 windows.
    assert len(X) == 8 and len(y) == 8
    for seq in X:
        assert len(seq) == 3
        assert all(len(frame) == 2 * NUM_KEYPOINTS for frame in seq)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    assert all(label == stroke for label in y)


def test_windowed_requires_all_frames_annotated():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 4, "stroke"))  # frames 0..4 labeled; 5..7 not
    poses = [_pose_rec(f) for f in range(8)]
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    # Only windows fully inside 0..4: starts 0,1,2 -> 3 windows.
    assert len(X) == 3


def test_windowed_does_not_span_frame_gap():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    anno.add(Segment("c", 1, 10, 12, "movement"))
    # Pose frames have a gap between 2 and 10.
    poses = [_pose_rec(f) for f in [0, 1, 2, 10, 11, 12]]
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    # One window from each consecutive run -> 2 windows, no straddling window.
    assert len(X) == 2
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    movement = schema.PHASE_TO_INDEX[schema.PHASE_MOVEMENT]
    assert sorted(y) == sorted([stroke, movement])


def test_windowed_label_from_last_vs_center():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 1, "movement"))
    anno.add(Segment("c", 1, 2, 2, "stroke"))
    poses = [_pose_rec(f) for f in range(3)]  # frames 0,1,2 consecutive
    movement = schema.PHASE_TO_INDEX[schema.PHASE_MOVEMENT]
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]

    # One window [0,1,2]. last -> frame 2 = stroke; center -> frame 1 = movement.
    X_last, y_last = build_windowed_training_dataset(
        anno, poses, "c", window_size=3, label_from="last")
    assert y_last == [stroke]
    X_ctr, y_ctr = build_windowed_training_dataset(
        anno, poses, "c", window_size=3, label_from="center")
    assert y_ctr == [movement]


def test_windowed_majority_label():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 1, "movement"))
    anno.add(Segment("c", 1, 2, 2, "stroke"))
    poses = [_pose_rec(f) for f in range(3)]
    movement = schema.PHASE_TO_INDEX[schema.PHASE_MOVEMENT]
    X, y = build_windowed_training_dataset(
        anno, poses, "c", window_size=3, label_from="majority")
    # labels [movement, movement, stroke] -> majority movement.
    assert y == [movement]


def test_windowed_stride_reduces_count():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "stroke"))
    poses = [_pose_rec(f) for f in range(10)]
    X1, _ = build_windowed_training_dataset(anno, poses, "c", window_size=2, stride=1)
    X2, _ = build_windowed_training_dataset(anno, poses, "c", window_size=2, stride=3)
    # 10 frames, window 2: valid starts are range(0, 9, stride).
    assert len(X1) == 9              # stride 1 -> starts 0..8
    assert len(X2) == 3              # stride 3 -> starts 0, 3, 6


def test_windowed_separates_players():
    # Multi-player (doubles-style) windowing: group by id -> single_player=False.
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    anno.add(Segment("c", 2, 0, 2, "idle"))
    poses = [_pose_rec(f, player_id=1) for f in range(3)]
    poses += [_pose_rec(f, player_id=2) for f in range(3)]
    X, y = build_windowed_training_dataset(
        anno, poses, "c", window_size=3, single_player=False)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    idle = schema.PHASE_TO_INDEX[schema.PHASE_IDLE]
    assert sorted(y) == sorted([stroke, idle])


def test_windowed_validates_params():
    anno = AnnotationSet()
    poses = [_pose_rec(0)]
    for bad in (0, -1):
        try:
            build_windowed_training_dataset(anno, poses, "c", window_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for bad window_size")
    try:
        build_windowed_training_dataset(anno, poses, "c", window_size=2, label_from="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad label_from")


# --- ignore sentinel handling ----------------------------------------------

def test_ignore_frames_excluded_from_per_frame_dataset():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    anno.add(Segment("c", 1, 3, 5, "ignore"))
    poses = [_pose_rec(f) for f in range(6)]
    X, y = build_training_dataset(anno, poses, clip_id="c")
    # Only the 3 stroke frames survive; ignore frames are dropped.
    assert len(X) == 3
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    assert all(v == stroke for v in y)


def test_ignore_frame_invalidates_windows_containing_it():
    anno = AnnotationSet()
    # frames 0-4 stroke, 5-6 ignore, 7-11 movement (all consecutive poses)
    anno.add(Segment("c", 1, 0, 4, "stroke"))
    anno.add(Segment("c", 1, 5, 6, "ignore"))
    anno.add(Segment("c", 1, 7, 11, "movement"))
    poses = [_pose_rec(f) for f in range(12)]
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    movement = schema.PHASE_TO_INDEX[schema.PHASE_MOVEMENT]
    # Windows fully inside 0-4 (starts 0,1,2) + fully inside 7-11 (starts 7,8,9).
    # No window straddles the ignored 5-6 region.
    assert len(X) == 6
    assert set(y) == {stroke, movement}


def test_ignore_only_clip_yields_no_windows():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "ignore"))
    poses = [_pose_rec(f) for f in range(10)]
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    assert X == [] and y == []


# --- single-player join (tracker id drift / mismatch) ----------------------

def _pose_rec_pid(frame, player_id, val=0.5):
    return {
        "frame": frame,
        "player_id": player_id,
        "keypoints": [[val, val] for _ in range(NUM_KEYPOINTS)],
    }


def test_single_player_join_ignores_drifting_tracker_id_windowed():
    # Annotation is all player 1; pose ids drift 1 -> 14 -> 138 for the SAME
    # player across consecutive frames (rally47-style). In single-player mode
    # these must form one unbroken run, not three fragments.
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "stroke"))
    poses = []
    for f in range(10):
        pid = {0: 1, 1: 1, 2: 1, 3: 14, 4: 14, 5: 138}.get(f, 1)
        poses.append(_pose_rec_pid(f, pid))
    # single_player=True (default): one stream, 10 consecutive frames.
    X, y = build_windowed_training_dataset(anno, poses, "c", window_size=3)
    assert len(X) == 8  # 10 - 3 + 1, no fragmentation from id changes
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    assert all(v == stroke for v in y)


def test_strict_player_mode_fragments_on_id_change():
    # With single_player=False the id drift fragments the stream, so fewer/zero
    # windows survive -- demonstrating WHY single-player mode exists.
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 9, "stroke"))
    poses = [_pose_rec_pid(f, 1 if f < 2 else 14) for f in range(10)]
    X, y = build_windowed_training_dataset(
        anno, poses, "c", window_size=3, single_player=False)
    # player 1 has only frames 0,1 (run too short for window 3); player 14 has
    # frames 2..9 but annotation only labels player 1 -> those don't match.
    assert len(X) == 0


def test_single_player_join_frame_only_per_frame_builder():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    # Pose ids are 2 and 36 (never matches annotation's player 1), rally05-style.
    poses = [_pose_rec_pid(0, 2), _pose_rec_pid(1, 36), _pose_rec_pid(2, 2)]
    X, y = build_training_dataset(anno, poses, "c", single_player=True)
    stroke = schema.PHASE_TO_INDEX[schema.PHASE_STROKE]
    assert len(X) == 3 and all(v == stroke for v in y)
    # Strict mode: no id matches player 1 -> nothing joins.
    Xs, ys = build_training_dataset(anno, poses, "c", single_player=False)
    assert Xs == []


def test_single_player_frame_labels_rejects_multi_player_clip():
    anno = AnnotationSet()
    anno.add(Segment("c", 1, 0, 2, "stroke"))
    anno.add(Segment("c", 2, 0, 2, "idle"))
    try:
        anno.single_player_frame_labels("c")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for multi-player clip")
