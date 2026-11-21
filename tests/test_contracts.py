"""Tests for the upstream data contracts and stub generators."""

import pytest

from src.temporal import contracts
from src.temporal.contracts import (
    ContractError,
    NUM_KEYPOINTS,
    index_tracks_by_frame,
    load_track_jsonl,
    load_tracks_by_frame,
    pose_record_to_vector,
    stub_pose_stream,
    stub_track_stream,
    validate_pose_record,
    validate_stream,
    validate_track_record,
)


# --- Contract 1: tracked-clip record ---------------------------------------

def test_valid_track_record_passes():
    rec = {"frame": 0, "player_id": 1, "bbox": [10.0, 20.0, 110.0, 220.0], "confidence": 0.9}
    assert validate_track_record(rec) is rec


def test_track_record_tolerates_and_preserves_extra_fields():
    # The shuttle field (Person 1 ships it for the whole team) must be ignored,
    # not rejected, and must survive validation untouched.
    rec = {
        "frame": 5,
        "player_id": 2,
        "bbox": [0, 0, 50, 100],
        "shuttle": [640, 360],
        "anything_else": {"nested": True},
    }
    out = validate_track_record(rec)
    assert out["shuttle"] == [640, 360]
    assert out["anything_else"] == {"nested": True}


@pytest.mark.parametrize("missing", ["frame", "player_id", "bbox"])
def test_track_record_missing_required_field_raises(missing):
    rec = {"frame": 0, "player_id": 1, "bbox": [0, 0, 10, 10]}
    rec.pop(missing)
    with pytest.raises(ContractError):
        validate_track_record(rec)


def test_track_record_degenerate_bbox_raises():
    with pytest.raises(ContractError):
        validate_track_record({"frame": 0, "player_id": 1, "bbox": [10, 10, 10, 5]})


def test_track_record_bool_is_not_accepted_as_int():
    # bool is a subclass of int in Python; the contract must reject it.
    with pytest.raises(ContractError):
        validate_track_record({"frame": True, "player_id": 1, "bbox": [0, 0, 10, 10]})


# --- Contract 2: pose record ------------------------------------------------

def test_valid_pose_record_passes():
    kps = [[0.5, i / NUM_KEYPOINTS] for i in range(NUM_KEYPOINTS)]
    rec = {"frame": 0, "player_id": 1, "keypoints": kps}
    assert validate_pose_record(rec) is rec


def test_pose_record_accepts_optional_scores():
    kps = [[0.5, i / NUM_KEYPOINTS, 0.9] for i in range(NUM_KEYPOINTS)]
    rec = {"frame": 0, "player_id": 1, "keypoints": kps}
    assert validate_pose_record(rec) is rec


def test_pose_record_wrong_keypoint_count_raises():
    kps = [[0.5, 0.5] for _ in range(NUM_KEYPOINTS - 1)]
    with pytest.raises(ContractError):
        validate_pose_record({"frame": 0, "player_id": 1, "keypoints": kps})


def test_pose_record_out_of_range_coord_raises():
    kps = [[0.5, 0.5] for _ in range(NUM_KEYPOINTS)]
    kps[0] = [5.0, 0.5]  # way outside normalized range
    with pytest.raises(ContractError):
        validate_pose_record({"frame": 0, "player_id": 1, "keypoints": kps})


def test_pose_record_to_vector_length_and_order():
    kps = [[0.1 * i, 0.1 * i + 0.5, 0.9] for i in range(NUM_KEYPOINTS)]
    rec = {"frame": 0, "player_id": 1, "keypoints": kps}
    vec = pose_record_to_vector(rec)
    assert len(vec) == 2 * NUM_KEYPOINTS
    # (x, y) interleaved, scores dropped
    assert vec[0] == pytest.approx(0.0)
    assert vec[1] == pytest.approx(0.5)
    assert vec[2] == pytest.approx(0.1)


# --- Stubs ------------------------------------------------------------------

def test_stub_track_stream_conforms_to_contract():
    records = list(stub_track_stream(num_frames=30))
    assert len(records) == 30
    # Every stub record validates, including the shuttle extra field.
    validated = validate_stream(records, "track")
    assert len(validated) == 30
    assert "shuttle" in records[0]


def test_stub_pose_stream_conforms_to_contract():
    records = list(stub_pose_stream(num_frames=30))
    assert len(records) == 30
    validated = validate_stream(records, "pose")
    assert len(validated) == 30
    assert len(records[0]["keypoints"]) == NUM_KEYPOINTS


def test_stub_streams_are_deterministic_under_seed():
    a = list(stub_track_stream(num_frames=10, seed=42))
    b = list(stub_track_stream(num_frames=10, seed=42))
    assert a == b


def test_validate_stream_rejects_unknown_kind():
    with pytest.raises(ValueError):
        validate_stream([], "banana")


# --- Track JSONL loading (Person 1's real output) --------------------------

import json


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_track_jsonl_reads_valid_records(tmp_path):
    rows = list(stub_track_stream(num_frames=5))
    p = tmp_path / "tracking.jsonl"
    _write_jsonl(p, rows)
    loaded = load_track_jsonl(str(p))
    assert len(loaded) == 5
    assert loaded[0]["frame"] == 0


def test_load_track_jsonl_skips_blank_lines(tmp_path):
    rows = list(stub_track_stream(num_frames=3))
    p = tmp_path / "tracking.jsonl"
    content = "\n".join(json.dumps(r) for r in rows)
    p.write_text("\n\n" + content + "\n\n", encoding="utf-8")
    assert len(load_track_jsonl(str(p))) == 3


def test_load_track_jsonl_tolerates_malformed_json(tmp_path):
    good = list(stub_track_stream(num_frames=2))
    p = tmp_path / "tracking.jsonl"
    lines = [json.dumps(good[0]), "{not valid json", json.dumps(good[1])]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Malformed line skipped, two good records survive.
    loaded = load_track_jsonl(str(p))
    assert len(loaded) == 2


def test_load_track_jsonl_tolerates_contract_violations(tmp_path):
    p = tmp_path / "tracking.jsonl"
    lines = [
        json.dumps({"frame": 0, "player_id": 1, "bbox": [0, 0, 10, 20]}),
        json.dumps({"frame": 1, "player_id": 2, "bbox": [5, 5, 5, 1]}),  # degenerate
        json.dumps({"frame": 2, "player_id": 1}),                        # missing bbox
        json.dumps({"frame": 3, "player_id": 1, "bbox": [0, 0, 8, 8]}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_track_jsonl(str(p))
    # Only the two valid records remain.
    assert [r["frame"] for r in loaded] == [0, 3]


def test_load_track_jsonl_preserves_shuttle_extra_field(tmp_path):
    rows = list(stub_track_stream(num_frames=3, include_shuttle=True))
    p = tmp_path / "tracking.jsonl"
    _write_jsonl(p, rows)
    loaded = load_track_jsonl(str(p))
    assert all("shuttle" in r for r in loaded)


def test_index_tracks_by_frame_builds_dict():
    rows = list(stub_track_stream(num_frames=4))
    by_frame = index_tracks_by_frame(rows)
    assert set(by_frame) == {0, 1, 2, 3}
    assert by_frame[2]["frame"] == 2


def test_load_tracks_by_frame_round_trip(tmp_path):
    rows = list(stub_track_stream(num_frames=6))
    p = tmp_path / "tracking.jsonl"
    _write_jsonl(p, rows)
    by_frame = load_tracks_by_frame(str(p))
    assert len(by_frame) == 6
    assert all(f == rec["frame"] for f, rec in by_frame.items())


# --- MediaPipe dict-keypoint format + merged pose JSONL --------------------

from src.temporal.contracts import (
    load_pose_jsonl,
    normalize_keypoints_to_bbox,
    pose_record_to_vector as _p2v,
)


def _mp_keypoints(x=700.0, y=400.0):
    """Synthetic MediaPipe-style dict keypoints in pixel space, using the real
    canonical keypoint names so name-based selection accepts them."""
    from src.temporal.contracts import KEYPOINT_NAMES
    return [
        {"name": name, "x": x + i, "y": y + i,
         "visibility": 0.9, "presence": 0.99}
        for i, name in enumerate(KEYPOINT_NAMES)
    ]


def test_validate_pose_record_accepts_dict_keypoints_when_normalized():
    # dict keypoints already in [0,1] should validate.
    kps = [{"name": f"k{i}", "x": 0.5, "y": i / NUM_KEYPOINTS,
            "visibility": 0.9, "presence": 0.9} for i in range(NUM_KEYPOINTS)]
    rec = {"frame": 0, "player_id": 1, "keypoints": kps}
    assert validate_pose_record(rec) is rec


def test_pose_record_to_vector_handles_dict_keypoints():
    kps = [{"name": f"k{i}", "x": 0.1, "y": 0.2} for i in range(NUM_KEYPOINTS)]
    rec = {"frame": 0, "player_id": 1, "keypoints": kps}
    vec = _p2v(rec)
    assert len(vec) == 2 * NUM_KEYPOINTS
    assert vec[0] == pytest.approx(0.1)
    assert vec[1] == pytest.approx(0.2)


def test_normalize_keypoints_to_bbox_maps_corners():
    bbox = [100.0, 200.0, 300.0, 600.0]  # width 200, height 400
    kps = [{"x": 100.0, "y": 200.0}, {"x": 300.0, "y": 600.0}, {"x": 200.0, "y": 400.0}]
    norm = normalize_keypoints_to_bbox(kps, bbox)
    assert norm[0] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert norm[1] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert norm[2] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_normalize_rejects_degenerate_bbox():
    with pytest.raises(ContractError):
        normalize_keypoints_to_bbox([{"x": 1.0, "y": 1.0}], [10.0, 10.0, 10.0, 20.0])


def test_load_pose_jsonl_reads_merged_record(tmp_path):
    from src.temporal.contracts import KEYPOINT_NAMES
    # bbox 200 wide x 400 tall; keypoints placed INSIDE the box so they
    # normalize into [0, 1]. Use canonical names so selection accepts them.
    bbox = [100.0, 200.0, 300.0, 600.0]
    kps = [
        {"name": name, "x": 100.0 + i * 2.0, "y": 200.0 + i * 4.0,
         "visibility": 0.9, "presence": 0.99}
        for i, name in enumerate(KEYPOINT_NAMES)
    ]
    rec = {"frame": 0, "player_id": 1, "bbox": bbox, "confidence": 0.9, "keypoints": kps}
    p = tmp_path / "pose.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    loaded = load_pose_jsonl(str(p))
    assert len(loaded) == 1
    out = loaded[0]
    assert set(out) == {"frame", "player_id", "keypoints"}
    assert len(out["keypoints"]) == NUM_KEYPOINTS
    # 'nose' is first in KEYPOINT_NAMES; at (x=100, y=200) it sits at the bbox
    # top-left corner -> (0.0, 0.0).
    assert out["keypoints"][0][0] == pytest.approx(0.0)
    assert out["keypoints"][0][1] == pytest.approx(0.0)
    # it validates as an internal pose record
    validate_pose_record(out)


def test_load_pose_jsonl_selects_and_reorders_by_name(tmp_path):
    from src.temporal.contracts import KEYPOINT_NAMES
    # Keypoints supplied in REVERSED order plus an extra landmark that must be
    # dropped. The loader should still output them in canonical order.
    bbox = [0.0, 0.0, 200.0, 400.0]
    reversed_kps = [
        {"name": name, "x": 10.0, "y": 10.0, "visibility": 0.9, "presence": 0.9}
        for name in reversed(KEYPOINT_NAMES)
    ]
    reversed_kps.append(
        {"name": "left_eye", "x": 5.0, "y": 5.0, "visibility": 0.9, "presence": 0.9}
    )
    rec = {"frame": 0, "player_id": 1, "bbox": bbox, "keypoints": reversed_kps}
    p = tmp_path / "pose.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    loaded = load_pose_jsonl(str(p))
    assert len(loaded) == 1
    # Dropped the extra 'left_eye'; kept exactly the canonical count.
    assert len(loaded[0]["keypoints"]) == NUM_KEYPOINTS


def test_load_pose_jsonl_tolerates_empty_keypoints(tmp_path):
    good = {"frame": 0, "player_id": 1, "bbox": [0.0, 0.0, 200.0, 400.0],
            "keypoints": _mp_keypoints(x=10.0, y=10.0)}
    bad = {"frame": 1, "player_id": 1, "bbox": [0.0, 0.0, 200.0, 400.0],
           "keypoints": []}  # pose detection failed this frame
    p = tmp_path / "pose.jsonl"
    p.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    loaded = load_pose_jsonl(str(p))
    assert [r["frame"] for r in loaded] == [0]


def test_load_pose_jsonl_on_real_sample_if_present():
    # The committed sample output, if available, must load and normalize.
    import os
    path = "outputs/pose.jsonl"
    if not os.path.exists(path):
        pytest.skip("no real pose.jsonl sample present")
    loaded = load_pose_jsonl(path)
    assert loaded, "expected at least one valid record"
    for rec in loaded[:20]:
        assert len(rec["keypoints"]) == NUM_KEYPOINTS
        for x, y in rec["keypoints"]:
            assert -0.5 <= x <= 1.5 and -0.5 <= y <= 1.5
