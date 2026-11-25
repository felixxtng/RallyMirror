"""Environment configuration shared by API and workers."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_MODES = ("combined", "court", "tracking", "pose", "racket", "shuttle", "phases")
AnalysisMode = Literal["combined", "court", "tracking", "pose", "racket", "shuttle", "phases"]

@dataclass
class Settings:
    backend: str = field(default_factory=lambda: os.getenv("JOB_BACKEND", "local"))
    root: Path = field(default_factory=lambda: Path(os.getenv("JOB_ROOT", str(ROOT / "outputs/deployment"))).resolve())
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    access_token: str = field(default_factory=lambda: os.getenv("APP_ACCESS_TOKEN", ""))
    worker_secret: str = field(default_factory=lambda: os.getenv("WORKER_SECRET", ""))
    api_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "" ).rstrip("/"))
    endpoint_id: str = field(default_factory=lambda: os.getenv("RUNPOD_ENDPOINT_ID", ""))
    runpod_key: str = field(default_factory=lambda: os.getenv("RUNPOD_API_KEY", ""))
    bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_endpoint: str = field(default_factory=lambda: os.getenv("S3_ENDPOINT_URL", ""))
    max_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", "209715200")))
    max_seconds: int = field(default_factory=lambda: int(os.getenv("MAX_VIDEO_SECONDS", "60")))
    timeout: int = field(default_factory=lambda: int(os.getenv("JOB_TIMEOUT_SECONDS", "1800")))
    device: str = field(default_factory=lambda: os.getenv("INFERENCE_DEVICE", "auto"))
    frontend_dist: Path = field(default_factory=lambda: Path(os.getenv(
        "FRONTEND_DIST", str(ROOT.parent / "dist"))).resolve())

    def __post_init__(self):
        if self.backend not in {"local", "runpod"}:
            raise ValueError("JOB_BACKEND must be local or runpod")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.database_url:
            self.database_url = "sqlite:///" + (self.root / "jobs.sqlite3").as_posix()
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        elif self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        if self.backend == "runpod" and not all((self.access_token, self.worker_secret,
                self.api_url.startswith("https://"), self.endpoint_id, self.runpod_key,
                self.bucket, self.s3_endpoint, not self.database_url.startswith("sqlite:"))):
            raise ValueError("Runpod API needs HTTPS API_BASE_URL, PostgreSQL, APP_ACCESS_TOKEN, "
                             "WORKER_SECRET, RUNPOD_ENDPOINT_ID/API_KEY, S3_BUCKET/ENDPOINT_URL")
