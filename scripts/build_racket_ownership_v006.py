"""Prepare the completed 1000-frame annotation pass for one Runpod run."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.racket.sampled_owner_package import build, finalize


if __name__ == '__main__':
    bundle = build(REPO, REPO/'data/racket_ownership_sampled_v006', REPO/'outputs/racket_ownership_v006')
    finalize(REPO, bundle, REPO/'Runpod/racket_ownership_v006.zip', REPO/'docs/racket_ownership_v006.md')
