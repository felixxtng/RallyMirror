#!/usr/bin/env python3
"""Fetch the selected RallyMirror models, or verify them without networking.

Run from frontend: python scripts/fetch-models.py [--check]
Uses only the Python standard library. Does not install packages, change the
model selection, or run inference. Existing files with a different hash stop
the operation instead of being overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

SOURCE_COMMIT = "017a366aed9ec4091881b678412e14e2eca518f8"
SOURCE_REPOSITORY = "felixxtng/RallyMirror"
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
MODELS = (
    ("models/yolo11n.pt", "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    ("models/court_final.pt", "09e968a6ecc6dd3673101789da4ac5661c1f5db908a47ebd41e2b877c34d0413"),
    ("models/phase_tcn.pt", "1403c241b7535fd71519211c4074eed9b1a93320e0d88d7f288bf009e77a02e5"),
    ("models/pose_landmarker_lite.task", "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    ("models/racket_center_v006.pt", "313da2f2f2958744d81318d1dbdef9b0b84fd2eb6ca95d7250ec0555f07283d5"),
    ("models/shuttle_full_b8/best.pt", "273e34974c29d0c066de99080b265a4b9ab4d2e7e8058654bf2185e3bfc96b91"),
)
BLOCK_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(BLOCK_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def validate_manifest(backend: Path = BACKEND_ROOT) -> None:
    """A changed local manifest must not silently change this pinned selection."""
    rows = json.loads((backend / "configs/deployment_models.json").read_text(encoding="utf-8-sig"))
    actual = {row["path"]: row["sha256"] for row in rows}
    if len(rows) != len(MODELS) or actual != dict(MODELS):
        raise ValueError("Model manifest differs from the pinned selection; review the source commit and hashes first")


def selected_model(path: str, expected: str, backend: Path = BACKEND_ROOT, *, check: bool = False) -> str:
    relative = PurePosixPath(path)
    if ("\\" in path or relative.is_absolute() or ".." in relative.parts
            or len(relative.parts) < 2 or relative.parts[0] != "models"):
        raise ValueError("Model path must stay inside backend/models")
    destination = backend.joinpath(*relative.parts)
    if destination.is_file():
        if sha256(destination) != expected:
            raise ValueError(f"Checksum differs: {path}. Existing model was left unchanged")
        return "verified"
    if destination.exists():
        raise ValueError(f"Model destination is not a file: {path}")
    if check:
        raise FileNotFoundError(f"Missing model: {path}. Run this script without --check to fetch it")

    url = (f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}/"
           + urllib.parse.quote(path, safe="/"))
    request = urllib.request.Request(url, headers={
        "User-Agent": "RallyMirror-model-fetch/1.0", "Accept": "application/octet-stream",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix="." + destination.name + ".", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            value = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=60) as response:
                for block in iter(lambda: response.read(BLOCK_BYTES), b""):
                    size += len(block)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"Unexpectedly large model download: {path}")
                    value.update(block)
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if value.hexdigest() != expected:
            raise ValueError(f"Downloaded checksum differs: {path}. No model was installed")
        # A concurrent downloader or user may have populated this path while the
        # transfer ran. Preserve that file unless it is already the same model.
        if destination.exists():
            if not destination.is_file() or sha256(destination) != expected:
                raise ValueError(f"Model destination changed while downloading: {path}; left unchanged")
            return "verified"
        os.replace(temporary, destination)
        temporary = None
        return "downloaded"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all files without network access")
    args = parser.parse_args(argv)
    try:
        validate_manifest()
        print(f"Selected source: {SOURCE_REPOSITORY}@{SOURCE_COMMIT}")
        for path, expected in MODELS:
            result = selected_model(path, expected, check=args.check)
            print(f"{result}: {path}")
        print("All six selected models match their recorded SHA-256 checksums.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as error:
        print(f"Model preparation stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
