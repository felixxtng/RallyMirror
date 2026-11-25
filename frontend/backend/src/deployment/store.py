"""Durable queue. Conditional updates arbitrate competing workers/submissions."""
import hashlib
import time
import uuid
from sqlalchemy import create_engine, Column, String, Float, Text, update, select, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker
from .settings import ANALYSIS_MODES

Base = declarative_base()
TERMINAL = {"succeeded", "failed"}

def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()

class Job(Base):
    __tablename__ = "video_jobs"
    id = Column(String(36), primary_key=True)
    token_hash = Column(String(64), nullable=False)
    state = Column(String(24), nullable=False, default="uploading")
    analysis_mode = Column(String(24), nullable=False, default="combined", server_default="combined")
    stage = Column(String(64), default="upload")
    created = Column(Float, default=time.time)
    updated = Column(Float, default=time.time)
    remote_id = Column(String(256), nullable=True)
    error = Column(Text, nullable=True)

class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30} if url.startswith("sqlite:") else {})
        Base.metadata.create_all(self.engine)
        self._migrate_analysis_mode()
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def _migrate_analysis_mode(self):
        """Existing SQLite/PostgreSQL jobs retain their original combined mode.

        create_all does not add columns to an existing table. Use a constant,
        additive statement and tolerate another API/worker winning the startup
        race; an unrelated database failure must still fail startup visibly.
        """
        if "analysis_mode" in {column["name"] for column in inspect(self.engine).get_columns("video_jobs")}:
            return
        try:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE video_jobs ADD COLUMN analysis_mode VARCHAR(24) NOT NULL DEFAULT 'combined'"))
        except SQLAlchemyError:
            if "analysis_mode" not in {column["name"] for column in inspect(self.engine).get_columns("video_jobs")}:
                raise

    def create(self, token, analysis_mode="combined"):
        if analysis_mode not in ANALYSIS_MODES:
            raise ValueError("Unsupported analysis mode")
        with self.session.begin() as db:
            row = Job(id=str(uuid.uuid4()), token_hash=digest(token), analysis_mode=analysis_mode)
            db.add(row)
        return row

    def get(self, job_id):
        with self.session() as db:
            return db.get(Job, job_id)

    def change(self, job_id, allowed, **values):
        with self.session.begin() as db:
            result = db.execute(update(Job).where(Job.id == job_id, Job.state.in_(allowed))
                .values(updated=time.time(), **values))
            return result.rowcount == 1

    def claim_next(self):
        with self.session() as db:
            ids = list(db.scalars(select(Job.id).where(Job.state == "queued").order_by(Job.created).limit(10)))
        for job_id in ids:
            if self.change(job_id, ["queued"], state="running", stage="starting"):
                return job_id
        return None

    def fail_interrupted(self):
        with self.session.begin() as db:
            return db.execute(update(Job).where(Job.state == "running").values(
                state="failed", stage="failed", error="Worker interrupted; submit video again",
                updated=time.time())).rowcount
