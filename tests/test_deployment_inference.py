"""Check that disabling intermediate rendering leaves inference data unchanged."""
import json
from unittest.mock import patch
import cv2
import numpy as np
from src.pose import demo
from src.shuttle.offline_demo import run_offline_demo


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320,240))
    for _ in range(3):
        writer.write(np.full((240,320,3),30,dtype=np.uint8))
    writer.release()


def test_pose_headless_same_records_and_models_receive_clean_frames(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    make_video(video)
    seen = []
    class Pose:
        def __init__(self, *a): pass
        def close(self): pass
        def extract(self, frame, target, others):
            # Green rectangles from earlier players would change this channel.
            assert np.ptp(frame[:,:,1]) == 0
            seen.append(frame.tobytes())
            return [{"name":"nose","x":50,"y":50,"visibility":1,"presence":1}]
    class Tracker:
        def __init__(self, **kw): assert kw["device"] == "cpu"
        def track_people(self, frame):
            return [{"bbox":[20,20,100,180],"track_id":1,"confidence":.9},
                    {"bbox":[90,20,160,170],"track_id":2,"confidence":.8}]
    class Racket:
        def __init__(self,*a,**kw): pass
        def annotate(self,frame,players):
            assert np.ptp(frame[:,:,1]) == 0
            seen.append(frame.tobytes())
            return players
    monkeypatch.setattr(demo,"PlayerPose",Pose)
    monkeypatch.setattr(demo,"PlayerTracker",Tracker)
    monkeypatch.setattr(demo,"_selection_with_court",lambda *a:None)
    monkeypatch.setattr(demo,"select_targets",lambda tracks,*a:tracks)
    monkeypatch.setattr(demo,"draw_pose",lambda frame,*a:frame.fill(255))
    monkeypatch.setattr("src.racket.pose_integration.RacketKeypointAnnotator",Racket)
    demo.run_demo(video,None,tmp_path/"preview.mp4",tmp_path/"preview.jsonl",racket_weights="fake", tracker_device="cpu")
    preview_seen=list(seen);seen.clear()
    demo.run_demo(video,None,None,tmp_path/"headless.jsonl",racket_weights="fake", tracker_device="cpu")
    assert seen == preview_seen
    assert (tmp_path/"preview.jsonl").read_bytes() == (tmp_path/"headless.jsonl").read_bytes()
    assert len((tmp_path/"headless.jsonl").read_text().splitlines()) == 6


def test_shuttle_headless_identical_refined_candidates(tmp_path):
    video=tmp_path/"input.mp4"
    make_video(video)
    with patch("src.shuttle.demo.ShuttleDetector") as detector:
        detector.return_value.device="cpu"
        detector.return_value.predict_candidates.side_effect=[[{"x":100+i,"y":100,"score":.9}] for i in range(3)]
        run_offline_demo(video,"fake",tmp_path/"preview")
    with patch("src.shuttle.offline_demo.render_refined_video") as render:
        run_offline_demo(video,"fake",tmp_path/"headless",candidate_cache=tmp_path/"preview/shuttle_tracking.jsonl",render_video=False)
        render.assert_not_called()
    assert (tmp_path/"preview/shuttle_tracking.jsonl").read_bytes() == (tmp_path/"headless/shuttle_tracking.jsonl").read_bytes()
    assert not (tmp_path/"headless/shuttle_demo.mp4").exists()


def test_pipeline_encodes_browser_video_and_report_in_isolated_workspace(tmp_path, monkeypatch):
    import shutil
    import subprocess
    from src.deployment import pipeline
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280,720))
    for _ in range(3):
        writer.write(np.zeros((720,1280,3),dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(pipeline,"check_models",lambda: [{"path":"test-only","sha256":"test"}])
    def fake_run(stem,args,device,*,input_path,workspace,progress):
        assert input_path == video.resolve()
        assert args.window_size == 15 and args.smoothing == 5 and args.overwrite
        assert workspace == (tmp_path/"work").resolve()
        progress("render")
        shutil.copyfile(video, workspace/"overlay.mp4")
        (workspace/"pose.jsonl").write_text("")
        return workspace/"overlay.mp4"
    monkeypatch.setattr("src.combined_demo.run",fake_run)
    report = pipeline.process(video,tmp_path/"work",tmp_path/"result.mp4",device="cpu")
    assert report["pose_records"] == 0 and report["warnings"]
    assert (tmp_path/"report.json").is_file()
    assert not (tmp_path/"result.encoding.mp4").exists()
    info=json.loads(subprocess.run(["ffprobe","-v","error","-show_streams","-of","json",str(tmp_path/"result.mp4")],check=True,capture_output=True,text=True).stdout)
    stream=info["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert stream["nb_frames"] == "3" and stream["width"] == 1280
    # Existing work is never silently reused by the deployment pipeline.
    import pytest
    with pytest.raises(FileExistsError):
        pipeline.process(video,tmp_path/"work",tmp_path/"result.mp4",device="cpu")
