"""Mode, queue and static-serving contracts. Executors are explicit test doubles.

These checks never load model weights or claim inference quality.
"""
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.deployment.api import create_app
from src.deployment.settings import ANALYSIS_MODES, Settings
from src.deployment.store import Store, digest
from src.deployment.worker import process_next


@pytest.fixture
def app(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>RallyMirror Studio</title>")
    (dist / "assets" / "app.js").write_text("window.RALLYMIRROR_TEST=true;")
    return create_app(Settings(root=tmp_path / "jobs", frontend_dist=dist,
        access_token="test-team-key", worker_secret="test-worker-key", max_bytes=1024))


def create(client, mode="combined", *, include_body=True):
    kwargs = {"json": {"analysis_mode": mode}} if include_body else {}
    response = client.post("/api/jobs", headers={"Authorization": "Bearer test-team-key"}, **kwargs)
    assert response.status_code == 201
    job = response.json()
    return job, {"Authorization": "Bearer " + job["token"]}


@pytest.mark.parametrize("mode", ANALYSIS_MODES)
def test_each_mode_survives_upload_queue_execution_and_private_result(app, mode):
    seen = []
    with TestClient(app) as client:
        job, auth = create(client, mode)
        assert job["analysis_mode"] == mode
        base = "/api/jobs/" + job["id"]
        assert client.get(base).status_code == 404
        assert client.put(job["upload_url"], headers=job["upload_headers"], content=b"test-video").status_code == 200
        start = client.post(base + "/start", headers=auth)
        assert start.json()["state"] == "queued" and start.json()["analysis_mode"] == mode

        def executor(settings, input_path, directory, progress, analysis_mode="combined"):
            seen.append(analysis_mode)
            assert input_path.read_bytes() == b"test-video"
            progress("render")
            (directory / "result.mp4").write_bytes(b"explicit-mock-result")
            (directory / "report.json").write_text(json.dumps({"analysis_mode": analysis_mode, "warnings": []}))

        assert process_next(app.state.settings, app.state.store, app.state.storage, executor)
        assert seen == [mode]
        assert not process_next(app.state.settings, app.state.store, app.state.storage, executor)
        status = client.get(base, headers=auth).json()
        assert status["state"] == "succeeded" and status["analysis_mode"] == mode
        result = client.get(base + "/result", headers=auth).json()["url"]
        assert job["token"] not in result
        assert client.get(result).content == b"explicit-mock-result"
        assert client.get(result + "tampered").status_code == 403
        # A repeated start reads the persisted result and never queues again.
        assert client.post(base + "/start", headers=auth).json()["state"] == "succeeded"


def test_legacy_creation_defaults_combined_and_invalid_modes_never_create_jobs(app):
    with TestClient(app) as client:
        assert create(client, include_body=False)[0]["analysis_mode"] == "combined"
        assert client.post("/api/jobs", json={"analysis_mode": "court"}).status_code == 401
        response = client.post("/api/jobs", headers={"Authorization": "Bearer test-team-key"},
            json={"analysis_mode": "../../arbitrary-python"})
        assert response.status_code == 422


def test_health_lists_modes_and_built_static_files_do_not_shadow_private_api(app):
    with TestClient(app) as client:
        assert client.get("/healthz").json()["supported_modes"] == list(ANALYSIS_MODES)
        assert "RallyMirror Studio" in client.get("/").text
        assert client.get("/").headers["cache-control"] == "no-cache"
        assert client.get("/assets/app.js").text == "window.RALLYMIRROR_TEST=true;"
        assert client.get("/api/jobs/unknown").status_code == 404
        assert client.post("/internal/jobs/unknown/claim").status_code == 401
        assert client.get("/backend/models/court_final.pt").status_code == 404


def test_missing_build_reports_actionable_unavailability(tmp_path):
    api = create_app(Settings(root=tmp_path / "jobs", frontend_dist=tmp_path / "missing-dist"))
    with TestClient(api) as client:
        assert client.get("/healthz").status_code == 200
        response = client.get("/")
        assert response.status_code == 503
        assert "npm run build" in response.json()["detail"]


def test_authenticated_claim_returns_the_persisted_mode(app):
    with TestClient(app) as client:
        job, auth = create(client, "court")
        base = "/api/jobs/" + job["id"]
        client.put(job["upload_url"], headers=job["upload_headers"], content=b"test")
        client.post(base + "/start", headers=auth)
        claim = client.post("/internal/jobs/" + job["id"] + "/claim",
            headers={"Authorization": "Bearer test-worker-key"}, json={"analysis_mode": "shuttle"})
        assert claim.json() == {"claimed": True, "state": "running", "analysis_mode": "court"}
        assert not client.post("/internal/jobs/" + job["id"] + "/claim",
            headers={"Authorization": "Bearer test-worker-key"}).json()["claimed"]


def test_existing_sqlite_jobs_are_migrated_to_combined_without_losing_tokens(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    job_id = str(uuid.uuid4())
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE video_jobs (id VARCHAR(36) PRIMARY KEY, token_hash VARCHAR(64) NOT NULL, state VARCHAR(24) NOT NULL, stage VARCHAR(64), created FLOAT, updated FLOAT, remote_id VARCHAR(256), error TEXT)")
        connection.execute("INSERT INTO video_jobs (id,token_hash,state,stage,created,updated) VALUES (?,?,?,?,?,?)",
            (job_id, digest("legacy-token"), "queued", "queued", 1., 2.))
        connection.commit()
    url = "sqlite:///" + path.as_posix()
    store = Store(url)
    old = store.get(job_id)
    assert old.analysis_mode == "combined" and old.token_hash == digest("legacy-token")
    assert old.state == "queued"
    assert store.create("new-token", "shuttle").analysis_mode == "shuttle"
    # A second API/worker startup must not try adding the column again.
    reopened = Store(url)
    assert reopened.get(job_id).analysis_mode == "combined"
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        store.create("invalid", "invalid")
    reopened.engine.dispose()
    store.engine.dispose()


def test_worker_subprocess_receives_persisted_mode(tmp_path, monkeypatch):
    from src.deployment import worker
    seen = []

    class Child:
        returncode = 0
        def poll(self): return 0

    def popen(command, **kwargs):
        seen.append((command, kwargs))
        return Child()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    settings = Settings(root=tmp_path / "jobs")
    worker.execute(settings, tmp_path / "input.mp4", tmp_path, analysis_mode="racket")
    command = seen[0][0]
    assert command[command.index("--analysis-mode") + 1] == "racket"


def test_runpod_uses_authenticated_claim_mode_not_payload(tmp_path, monkeypatch):
    from src.deployment import runpod_worker
    settings = Settings(root=tmp_path / "jobs", api_url="https://api.example", worker_secret="test-worker")
    monkeypatch.setattr(runpod_worker, "Settings", lambda **kwargs: settings)
    artifacts, modes = {}, []

    class Bucket:
        def __init__(self, *args): pass
        def size(self, *args): return 5
        def download(self, job_id, name, destination): destination.write_bytes(b"video")
        def upload(self, job_id, name, source): artifacts[name] = source.read_bytes()

    monkeypatch.setattr(runpod_worker, "Storage", Bucket)

    def post(url, **kwargs):
        body = {"claimed": True, "state": "running", "analysis_mode": "court"} if url.endswith("/claim") else {"updated": True}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)

    def executor(settings, input_path, directory, progress, analysis_mode):
        modes.append(analysis_mode)
        progress("render")
        (directory / "result.mp4").write_bytes(b"result")
        (directory / "report.json").write_text(json.dumps({"analysis_mode": analysis_mode}))

    monkeypatch.setattr(runpod_worker, "execute", executor)
    result = runpod_worker.handler({"input": {"job_id": str(uuid.uuid4()), "analysis_mode": "shuttle"}})
    assert result["state"] == "succeeded" and result["analysis_mode"] == "court"
    assert modes == ["court"] and set(artifacts) == {"result.mp4", "report.json"}
