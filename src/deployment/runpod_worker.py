"""Runpod queue handler. Credentials are worker environment secrets only."""
import json
import os
import tempfile
import uuid
from pathlib import Path
import httpx
from .settings import Settings
from .storage import Storage, key
from .worker import execute


def handler(job):
    job_id = str(uuid.UUID(job["input"]["job_id"]))
    # The GPU worker doesn't need database or Runpod submission credentials.
    settings = Settings(backend="local")
    settings.backend = "runpod"
    if not settings.api_url.startswith("https://") or not settings.worker_secret:
        raise ValueError("Worker needs HTTPS API_BASE_URL and WORKER_SECRET")
    storage = Storage(settings)
    headers = {"Authorization": "Bearer " + settings.worker_secret}
    base = settings.api_url + "/internal/jobs/" + job_id
    def callback(event):
        response = httpx.post(base + "/events", headers=headers, json=event, timeout=20)
        response.raise_for_status()
    response = httpx.post(base + "/claim", headers=headers, timeout=20)
    response.raise_for_status()
    claim = response.json()
    if not claim["claimed"]:
        return {"job_id": job_id, "state": claim["state"], "duplicate": True}
    with tempfile.TemporaryDirectory(prefix="rallymirror-") as temporary:
        directory = Path(temporary)
        try:
            if not 0 < storage.size(job_id, "input.mp4") <= settings.max_bytes:
                raise ValueError("Invalid input size")
            storage.download(job_id, "input.mp4", directory / "input.mp4")
            def progress(stage):
                try:
                    callback({"state": "running", "stage": stage})
                except httpx.HTTPError:
                    pass
            execute(settings, directory / "input.mp4", directory, progress=progress)
            storage.upload(job_id, "report.json", directory / "report.json")
            storage.upload(job_id, "result.mp4", directory / "result.mp4")
        except Exception as exc:
            # Diagnostics stay in worker logs, not the browser or Runpod output.
            if (directory / "worker.log").exists():
                print((directory / "worker.log").read_text(encoding="utf-8", errors="replace")[-16000:], flush=True)
            print(f"Job failed: {type(exc).__name__}", flush=True)
            try:
                callback({"state": "failed", "stage": "failed", "error": "Processing failed; inspect GPU worker logs"})
            except httpx.HTTPError:
                pass
            return {"job_id": job_id, "state": "failed"}
    # Returning completion even if callback fails lets API reconcile /status.
    try:
        callback({"state": "succeeded", "stage": "complete"})
    except httpx.HTTPError:
        pass
    return {"job_id": job_id, "state": "succeeded", "result_key": key(job_id, "result.mp4")}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
