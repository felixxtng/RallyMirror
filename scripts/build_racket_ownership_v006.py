"""Prepare the completed 1000-frame annotation pass for one Runpod run."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.racket.sampled_owner_package import build, finalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'data/racket_ownership_sampled_v006')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/racket_ownership_v006')
    parser.add_argument('--package', type=Path, default=REPO/'Runpod/racket_ownership_v006.zip')
    args = parser.parse_args()
    if args.output.exists() or args.package.exists():
        parser.error('Preserve existing artifacts: choose fresh --output and --package paths.')
    bundle = build(REPO, args.root, args.output)
    finalize(REPO, bundle, args.package, REPO/'docs/racket_ownership_v006.md')
