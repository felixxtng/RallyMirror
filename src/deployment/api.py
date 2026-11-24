"""Small CPU-only API: uploads, durable job state and private results."""
import hmac
import secrets
from pathlib import Path
from typing import Literal
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from .settings import Settings
from .storage import Storage
from .store import Store, digest, TERMINAL

class Event(BaseModel):
    stage: str
    state: Literal["running", "succeeded", "failed"] = "running"
    error: str | None = None


def create_app(settings=None):
    settings = settings or Settings()
    store, storage = Store(settings.database_url), Storage(settings)
    app = FastAPI(title="RallyMirror video jobs", docs_url=None, redoc_url=None)
    app.state.settings, app.state.store, app.state.storage = settings, store, storage

    def authorized(job_id, authorization):
        job = store.get(job_id)
        token = (authorization or "").removeprefix("Bearer ")
        if job is None or not hmac.compare_digest(job.token_hash, digest(token)):
            raise HTTPException(404, "Job not found")
        return job

    def worker_auth(authorization):
        if not settings.worker_secret or not hmac.compare_digest(
                authorization or "", "Bearer " + settings.worker_secret):
            raise HTTPException(401, "Worker authentication required")

    def remote_request(method, suffix, payload=None):
        response = httpx.request(method, f"https://api.runpod.ai/v2/{settings.endpoint_id}/{suffix}",
            headers={"Authorization": "Bearer " + settings.runpod_key}, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    @app.get("/healthz")
    def health():
        # Exercises DB as well as HTTP readiness, without importing Torch.
        store.get("healthcheck")
        return {"status": "ok", "backend": settings.backend}

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    @app.post("/api/jobs", status_code=201)
    def create_job(authorization: str | None = Header(None)):
        if settings.access_token and not hmac.compare_digest(
                authorization or "", "Bearer " + settings.access_token):
            raise HTTPException(401, "Enter the team access key")
        token = secrets.token_urlsafe(32)
        job = store.create(token)
        return {"id": job.id, "token": token, "max_bytes": settings.max_bytes,
            "upload_url": storage.url(job.id, "upload.mp4", upload=True) if storage.client else f"/api/jobs/{job.id}/input",
            "upload_headers": {"Content-Type": "video/mp4"} if storage.client else
                {"Content-Type": "video/mp4", "Authorization": "Bearer " + token}}

    @app.put("/api/jobs/{job_id}/input")
    async def upload(job_id: str, request: Request, authorization: str | None = Header(None)):
        job = authorized(job_id, authorization)
        if storage.client:
            raise HTTPException(400, "Use the direct upload URL")
        if not store.change(job_id, ["uploading"], state="receiving", stage="upload"):
            raise HTTPException(409, "Upload already received or in progress")
        destination = storage.path(job_id, "input.mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".part")
        total = 0
        try:
            with partial.open("xb") as file:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > settings.max_bytes:
                        raise HTTPException(413, "Video exceeds upload limit")
                    file.write(chunk)
            if not total:
                raise HTTPException(400, "Empty upload")
            partial.replace(destination)
            store.change(job_id, ["receiving"], state="uploaded", stage="uploaded")
        except BaseException:
            partial.unlink(missing_ok=True)
            store.change(job_id, ["receiving"], state="uploading", stage="upload")
            raise
        return {"bytes": total}

    @app.post("/api/jobs/{job_id}/start")
    def start(job_id: str, authorization: str | None = Header(None)):
        job = authorized(job_id, authorization)
        if job.state not in {"uploading", "uploaded"}:
            return {"id": job_id, "state": job.state}
        try:
            size = storage.size(job_id, "upload.mp4" if storage.client else "input.mp4")
        except Exception:
            raise HTTPException(400, "Upload the video first")
        if not 0 < size <= settings.max_bytes:
            raise HTTPException(413, "Input is empty or exceeds upload limit")
        target = "dispatching" if storage.client else "queued"
        if not store.change(job_id, ["uploading", "uploaded"], state=target, stage=target):
            return {"id": job_id, "state": store.get(job_id).state}
        if storage.client:
            try:
                storage.freeze_input(job_id)
                if not 0 < storage.size(job_id, "input.mp4") <= settings.max_bytes:
                    raise ValueError("Invalid frozen input size")
            except Exception:
                store.change(job_id, ["dispatching"], state="failed", stage="failed", error="Cannot snapshot uploaded video")
                raise HTTPException(503, "Cannot snapshot uploaded video; submit again")
            try:
                remote = remote_request("POST", "run", {"input": {"job_id": job_id},
                    "policy": {"executionTimeout": settings.timeout * 1000, "ttl": 7200000}})
                remote_id = remote["id"]
                # A fast worker may already have claimed or even finished this job.
                store.change(job_id, ["dispatching"], state="queued", stage="queued")
                store.change(job_id, ["queued", "running", "succeeded", "failed"], remote_id=remote_id)
            except Exception:
                # Do NOT blindly retry /run: the remote may have accepted it.
                store.change(job_id, ["dispatching"], state="dispatch_unknown", stage="dispatch_unknown",
                    error="Remote submission uncertain; inspect Runpod jobs before resubmitting to avoid duplicate charges")
                raise HTTPException(503, "Submission uncertain; check job status and Runpod console")
        return {"id": job_id, "state": store.get(job_id).state}

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str, authorization: str | None = Header(None)):
        job = authorized(job_id, authorization)
        # Runpod status is also a recovery channel when the final callback failed.
        if storage.client and job.remote_id and job.state not in TERMINAL:
            try:
                remote = remote_request("GET", "status/" + job.remote_id)
                if remote["status"] == "COMPLETED":
                    output = remote.get("output") or {}
                    if output.get("job_id") == job_id and output.get("state") == "succeeded":
                        storage.size(job_id, "result.mp4")
                        storage.size(job_id, "report.json")
                        store.change(job_id, ["queued", "running", "dispatch_unknown"], state="succeeded", stage="complete")
                    elif output.get("state") == "failed":
                        store.change(job_id, ["queued", "running"], state="failed", stage="failed", error="GPU worker failed; inspect worker logs")
                elif remote["status"] in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                    store.change(job_id, ["queued", "running"], state="failed", stage="failed", error="GPU job " + remote["status"])
            except Exception:
                # Runpod retains async results only briefly. Private artifacts
                # remain the recovery record after provider status has expired.
                try:
                    storage.size(job_id, "report.json")
                    storage.size(job_id, "result.mp4")
                    store.change(job_id, ["queued", "running", "dispatch_unknown"], state="succeeded", stage="complete")
                except Exception:
                    pass  # Transient outage must not overwrite durable state.
            job = store.get(job_id)
        return {"id": job.id, "state": job.state, "stage": job.stage, "error": job.error,
                "created": job.created, "updated": job.updated}

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str, artifact: Literal["video", "report"] = "video",
               authorization: str | None = Header(None)):
        job = authorized(job_id, authorization)
        if job.state != "succeeded":
            raise HTTPException(409, "Result is not ready")
        name = "result.mp4" if artifact == "video" else "report.json"
        if storage.client:
            return {"url": storage.url(job_id, name)}
        # Local results also use a short-lived download capability so <video> can
        # issue Range requests without putting the persistent job token in URLs.
        import time, hashlib
        expires = int(time.time()) + 900
        signature = hmac.new(job.token_hash.encode(), f"{name}:{expires}".encode(), hashlib.sha256).hexdigest()
        return {"url": f"/api/jobs/{job_id}/download/{name}?expires={expires}&signature={signature}"}

    @app.get("/api/jobs/{job_id}/download/{name}")
    def download(job_id: str, name: str, expires: int, signature: str):
        import time, hashlib
        job = store.get(job_id)
        if storage.client or name not in {"result.mp4", "report.json"} or job is None or job.state != "succeeded":
            raise HTTPException(404)
        expected = hmac.new(job.token_hash.encode(), f"{name}:{expires}".encode(), hashlib.sha256).hexdigest()
        if expires < time.time() or expires > time.time()+901 or not hmac.compare_digest(signature, expected):
            raise HTTPException(403)
        return FileResponse(storage.path(job_id, name), media_type="video/mp4" if name.endswith("mp4") else "application/json")

    @app.post("/internal/jobs/{job_id}/claim")
    def claim(job_id: str, authorization: str | None = Header(None)):
        worker_auth(authorization)
        if store.get(job_id) is None:
            raise HTTPException(404)
        claimed = store.change(job_id, ["dispatching", "dispatch_unknown", "queued"], state="running", stage="starting", error=None)
        return {"claimed": claimed, "state": store.get(job_id).state}

    @app.post("/internal/jobs/{job_id}/events")
    def event(job_id: str, event: Event, authorization: str | None = Header(None)):
        worker_auth(authorization)
        if event.state == "succeeded":
            storage.size(job_id, "result.mp4")
            storage.size(job_id, "report.json")
        changed = store.change(job_id, ["running"], state=event.state,
            stage=event.stage[:64], error=(event.error or "")[:1000] or None)
        return {"updated": changed}

    return app
