"""Job lifecycle tests use a fake executor; they do not claim model accuracy."""
import json
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from src.deployment.api import create_app
from src.deployment.settings import Settings
from src.deployment.worker import process_next
from src.deployment.storage import key
from src.deployment.pipeline import inspect_video


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(root=tmp_path, access_token="team", worker_secret="worker", max_bytes=16))


def new_job(client):
    response = client.post("/api/jobs", headers={"Authorization": "Bearer team"})
    assert response.status_code == 201
    job = response.json()
    return job, {"Authorization": "Bearer " + job["token"]}


def test_local_upload_queue_result_and_capability(app):
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/").status_code == 200
        assert client.post("/api/jobs").status_code == 401
        job, headers = new_job(client)
        base = "/api/jobs/" + job["id"]
        assert client.get(base).status_code == 404
        assert client.post(base + "/start", headers=headers).status_code == 400
        assert client.put(job["upload_url"], headers=job["upload_headers"], content=b"video").status_code == 200
        for _ in range(2):
            assert client.post(base + "/start", headers=headers).json()["state"] == "queued"
        def executor(settings, input_path, directory, progress):
            assert input_path.read_bytes() == b"video"
            progress("pose_and_racket")
            (directory / "result.mp4").write_bytes(b"output-video")
            (directory / "report.json").write_text('{"warnings": []}')
        assert process_next(app.state.settings, app.state.store, app.state.storage, executor)
        assert not process_next(app.state.settings, app.state.store, app.state.storage, executor)
        assert client.get(base, headers=headers).json()["state"] == "succeeded"
        url = client.get(base + "/result", headers=headers).json()["url"]
        assert client.get(url).content == b"output-video"
        assert client.get(url, headers={"Range": "bytes=0-5"}).status_code == 206
        assert client.get(url + "0").status_code == 403
        assert client.get(base + "/download/result.mp4?expires=0&signature=x").status_code == 403
        assert client.put(job["upload_url"], headers=headers, content=b"again").status_code == 409
        assert client.post(base + "/start", headers=headers).json()["state"] == "succeeded"


def test_oversize_empty_and_failed_execution(app):
    with TestClient(app) as client:
        job, headers = new_job(client)
        assert client.put(job["upload_url"], headers=headers, content=b"x"*17).status_code == 413
        assert client.put(job["upload_url"], headers=headers, content=b"").status_code == 400
        assert not list(app.state.settings.root.rglob("*.part"))
        assert client.put(job["upload_url"], headers=headers, content=b"video").status_code == 200
        base = "/api/jobs/" + job["id"]
        client.post(base + "/start", headers=headers)
        def fail(*args, **kwargs):
            raise RuntimeError("private diagnostic")
        assert process_next(app.state.settings, app.state.store, app.state.storage, fail)
        state = client.get(base, headers=headers).json()
        assert state["state"] == "failed" and "private diagnostic" not in state["error"]
        assert client.get(base + "/result", headers=headers).status_code == 409


def test_claim_exclusive_and_terminal_state_immutable(app):
    with TestClient(app) as client:
        job, headers = new_job(client)
        base = "/internal/jobs/" + job["id"]
        auth = {"Authorization": "Bearer worker"}
        assert client.post(base + "/claim").status_code == 401
        assert not client.post(base + "/claim", headers=auth).json()["claimed"]
        app.state.store.change(job["id"], ["uploading"], state="dispatching")
        assert client.post(base + "/claim", headers=auth).json()["claimed"]
        assert not client.post(base + "/claim", headers=auth).json()["claimed"]
        assert client.post(base + "/events", headers=auth, json={"state":"failed","stage":"failed"}).json()["updated"]
        assert not client.post(base + "/events", headers=auth, json={"state":"running","stage":"pose"}).json()["updated"]
        assert app.state.store.get(job["id"]).state == "failed"


def test_cloud_snapshot_single_dispatch_and_reconcile(app, monkeypatch):
    settings = app.state.settings
    settings.endpoint_id, settings.runpod_key, settings.timeout = "endpoint", "secret", 1800
    storage = app.state.storage
    storage.client = object()
    monkeypatch.setattr(storage, "url", lambda *a, **k: "https://storage.invalid/upload")
    monkeypatch.setattr(storage, "size", lambda *a: 10)
    snapshots = []
    monkeypatch.setattr(storage, "freeze_input", lambda job_id: snapshots.append(job_id))
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST":
            return httpx.Response(200, json={"id":"remote"}, request=httpx.Request(method,url))
        return httpx.Response(200, json={"status":"COMPLETED", "output":{"job_id":job["id"],"state":"succeeded"}}, request=httpx.Request(method,url))
    monkeypatch.setattr(httpx, "request", request)
    with TestClient(app) as client:
        job, headers = new_job(client)
        base = "/api/jobs/" + job["id"]
        assert "Authorization" not in job["upload_headers"]
        assert client.post(base + "/start", headers=headers).status_code == 200
        assert client.post(base + "/start", headers=headers).status_code == 200
        assert len(calls) == 1 and snapshots == [job["id"]]
        assert calls[0][2]["json"]["input"] == {"job_id":job["id"]}
        assert calls[0][2]["json"]["policy"]["executionTimeout"] == 1800000
        assert client.get(base, headers=headers).json()["state"] == "succeeded"
        assert client.get(base + "/result", headers=headers).json()["url"].startswith("https://storage.")


def test_unknown_submission_not_retried(app, monkeypatch):
    storage = app.state.storage
    storage.client = object()
    monkeypatch.setattr(storage, "url", lambda *a, **k: "https://storage.invalid")
    monkeypatch.setattr(storage, "size", lambda *a: 10)
    monkeypatch.setattr(storage, "freeze_input", lambda *a: None)
    def timeout(*a, **k):
        raise httpx.ReadTimeout("submission may have succeeded")
    monkeypatch.setattr(httpx, "request", timeout)
    with TestClient(app) as client:
        job, headers = new_job(client)
        base = "/api/jobs/" + job["id"]
        assert client.post(base + "/start", headers=headers).status_code == 503
        assert client.post(base + "/start", headers=headers).json()["state"] == "dispatch_unknown"


def test_keys_cannot_escape_workspace():
    for job_id, name in [("../x", "input.mp4"),("4dff4ba1-9b7c-4d32-80a2-2249eb18292d", "../secret")]:
        with pytest.raises(ValueError):
            key(job_id, name)


def test_input_validation_reports_source_size_and_still_guards_duration(tmp_path):
    import cv2
    import numpy as np
    path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for _ in range(3):
        writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
    writer.release()
    info = inspect_video(path)
    # inspect_video now reports the SOURCE size (resolution is no longer a gate;
    # normalize_video resizes to 1280x720 downstream) but keeps the fps/duration guard.
    assert info["fps"] == 30
    assert info["source_width"] == 1280 and info["source_height"] == 720
    with pytest.raises(ValueError, match="at most"):
        inspect_video(path, max_seconds=.01)


def test_non_standard_resolution_is_accepted_and_normalized(tmp_path):
    import cv2
    import numpy as np
    from src.deployment.pipeline import normalize_video, TARGET_WIDTH, TARGET_HEIGHT
    src = tmp_path / "small.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(3):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    writer.release()
    # A non-1280x720 clip is NOT rejected by inspect_video ...
    info = inspect_video(src)
    assert (info["source_width"], info["source_height"]) == (640, 480)
    # ... and normalize_video brings it onto the model's canvas.
    out = normalize_video(src, tmp_path / "norm.mp4")
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (TARGET_WIDTH, TARGET_HEIGHT)


def test_gpu_handler_uploads_results_even_when_completion_callback_is_lost(tmp_path, monkeypatch):
    from src.deployment import runpod_worker
    import uuid
    settings=Settings(root=tmp_path, api_url="https://api.invalid", worker_secret="worker")
    monkeypatch.setattr(runpod_worker,"Settings",lambda **kwargs:settings)
    artifacts={}
    class Bucket:
        def __init__(self,*a): pass
        def size(self,*a): return 5
        def download(self,job_id,name,destination): destination.write_bytes(b"video")
        def upload(self,job_id,name,source): artifacts[name]=source.read_bytes()
    monkeypatch.setattr(runpod_worker,"Storage",Bucket)
    def post(url,**kwargs):
        if url.endswith("claim"):
            return httpx.Response(200,json={"claimed":True,"state":"running"},request=httpx.Request("POST",url))
        if kwargs["json"]["state"] == "succeeded":
            raise httpx.ConnectError("callback unavailable")
        return httpx.Response(200,json={"updated":True},request=httpx.Request("POST",url))
    monkeypatch.setattr(httpx,"post",post)
    directories=[]
    def execute(settings,input_path,directory,progress):
        directories.append(directory)
        progress("render")
        (directory/"result.mp4").write_bytes(b"result")
        (directory/"report.json").write_text("{}")
    monkeypatch.setattr(runpod_worker,"execute",execute)
    job_id=str(uuid.uuid4())
    output=runpod_worker.handler({"input":{"job_id":job_id}})
    assert output["state"] == "succeeded" and output["job_id"] == job_id
    assert artifacts == {"report.json":b"{}", "result.mp4":b"result"}
    assert not directories[0].exists()
