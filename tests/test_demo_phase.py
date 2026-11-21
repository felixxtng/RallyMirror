"""Tests for the phase-recognition demo."""

import json

from src.temporal import schema
from src.temporal.demo_phase import run_phase_demo


def test_demo_writes_jsonl_with_valid_phases(tmp_path):
    out_jsonl = tmp_path / "phase_demo.jsonl"
    run_phase_demo(
        output_jsonl=str(out_jsonl),
        output_video=None,       # skip rendering
        num_frames=60,
        window_size=20,
        smoothing=3,
    )

    assert out_jsonl.exists()
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert rows, "expected at least one phase row"
    for row in rows:
        assert set(row) == {"frame", "player_id", "phase"}
        assert row["phase"] in schema.PHASES
        assert isinstance(row["frame"], int)


def test_demo_summary_counts_match_rows(tmp_path):
    out_jsonl = tmp_path / "phase_demo.jsonl"
    summary = run_phase_demo(
        output_jsonl=str(out_jsonl),
        output_video=None,
        num_frames=50,
        window_size=10,
    )
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert summary["phases_emitted"] == len(rows)
    assert sum(summary["phase_counts"].values()) == len(rows)
    # window_size=10 over 50 frames -> 41 emitted (frames 9..49).
    assert summary["phases_emitted"] == 50 - 10 + 1


def test_demo_runs_without_opencv_video_is_optional(tmp_path):
    # Requesting a video must not crash whether or not OpenCV is present; the
    # JSONL is always produced. summary['video'] is the path or None.
    out_jsonl = tmp_path / "phase_demo.jsonl"
    out_video = tmp_path / "phase_demo.mp4"
    summary = run_phase_demo(
        output_jsonl=str(out_jsonl),
        output_video=str(out_video),
        source_video=None,
        num_frames=40,
        window_size=15,
    )
    assert out_jsonl.exists()
    assert summary["video"] in (str(out_video), None)


def test_demo_renders_video_when_opencv_available(tmp_path):
    # If cv2 is installed (project venv), the overlay video must be produced
    # and be non-empty. Skips cleanly in a minimal env without cv2.
    import pytest
    pytest.importorskip("cv2")

    out_jsonl = tmp_path / "phase_demo.jsonl"
    out_video = tmp_path / "phase_demo.mp4"
    summary = run_phase_demo(
        output_jsonl=str(out_jsonl),
        output_video=str(out_video),
        source_video=None,       # blank canvas, no footage needed
        num_frames=45,
        window_size=15,
    )
    assert summary["video"] == str(out_video)
    assert out_video.exists()
    assert out_video.stat().st_size > 0


def test_demo_is_deterministic(tmp_path):
    a = run_phase_demo(output_jsonl=str(tmp_path / "a.jsonl"), output_video=None,
                       num_frames=40, window_size=12, seed=7)
    b = run_phase_demo(output_jsonl=str(tmp_path / "b.jsonl"), output_video=None,
                       num_frames=40, window_size=12, seed=7)
    assert a["phase_counts"] == b["phase_counts"]
