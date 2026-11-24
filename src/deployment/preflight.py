"""Verify selected assets without importing ML libraries or downloading weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from .settings import ROOT

MANIFEST = ROOT / "configs/deployment_models.json"

def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def check_models():
    rows = json.loads(MANIFEST.read_text())
    errors = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            errors.append(f"Missing selected model: {row['path']}")
        elif not row["sha256"]:
            errors.append(f"Unrecorded checksum: {row['path']}; run preflight --record after confirming final weights")
        elif sha256(path) != row["sha256"]:
            errors.append(f"Checksum differs: {row['path']}; confirm intended model before recording a replacement")
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            errors.append(f"Missing executable on PATH: {binary}")
    if errors:
        raise ValueError("\n".join(errors))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record checksums of ALL present selected assets")
    args = parser.parse_args()
    if args.record:
        rows = json.loads(MANIFEST.read_text())
        for row in rows:
            row["sha256"] = sha256(ROOT / row["path"])
        MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(check_models(), indent=2))

if __name__ == "__main__":
    main()
