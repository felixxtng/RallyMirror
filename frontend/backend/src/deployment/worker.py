"""Sequential local queue worker and isolated inference subprocess launcher."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from .settings import Settings, ROOT, ANALYSIS_MODES
from .storage import Storage
from .store import Store


def execute(settings, input_path, job_dir, progress=None, analysis_mode="combined"):
    if analysis_mode not in ANALYSIS_MODES:
        raise ValueError("Unsupported analysis mode")
    log = job_dir / "worker.log"
    command = [sys.executable, "-u", "-m", "src.deployment.pipeline", str(input_path),
        str(job_dir / "work"), str(job_dir / "result.mp4"), "--device", settings.device,
        "--max-seconds", str(settings.max_seconds), "--analysis-mode", analysis_mode]
    with log.open("w", encoding="utf-8") as output, log.open("r", encoding="utf-8", errors="replace") as reader:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + settings.timeout
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, settings.timeout)
                for line in reader.readlines():
                    if progress and line.startswith("STAGE "):
                        progress(line.strip().split(" ", 1)[1])
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def process_next(settings, store, storage, executor=execute):
    job_id = store.claim_next()
    if job_id is None:
        return False
    directory = storage.path(job_id, "input.mp4").parent
    try:
        job = store.get(job_id)
        executor(settings, storage.path(job_id, "input.mp4"), directory,
            progress=lambda stage: store.change(job_id, ["running"], stage=stage),
            analysis_mode=job.analysis_mode)
        storage.size(job_id, "result.mp4")
        storage.size(job_id, "report.json")
        store.change(job_id, ["running"], state="succeeded", stage="complete")
    except Exception:
        store.change(job_id, ["running"], state="failed", stage="failed",
            error="Processing failed. Inspect this job's worker.log (model preflight, input format, memory or timeout).")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--recover", action="store_true", help="Mark interrupted running jobs failed; STOP all previous workers first")
    args = parser.parse_args()
    settings = Settings()
    if settings.backend != "local":
        raise ValueError("Use runpod_worker for JOB_BACKEND=runpod")
    store, storage = Store(settings.database_url), Storage(settings)
    if args.recover:
        print("Marked interrupted jobs failed:", store.fail_interrupted())
    while True:
        worked = process_next(settings, store, storage)
        if args.once:
            break
        if not worked:
            time.sleep(1)

if __name__ == "__main__":
    main()
