"""One-off diagnostic: pose-record survival rate per annotated clip.

Flags clips whose pose JSONL mostly fails Contract 2 validation (e.g. the
tracker locked onto a spectator), so they can be excluded from training.
"""

import contextlib
import glob
import io
import os

from src.temporal.contracts import load_pose_jsonl

anno_ids = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob("data/annotations/*.jsonl")
)

print(f'{"clip":12} {"raw":>6} {"valid":>6} {"rate":>6}')
print("-" * 34)
rows = []
for cid in anno_ids:
    path = f"data/shuttle/clips/{cid}/pose.jsonl"
    if not os.path.exists(path):
        print(f'{cid:12} {"MISSING":>6}')
        continue
    raw = sum(1 for ln in open(path, encoding="utf-8") if ln.strip())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # swallow per-line skip warnings
        valid = len(load_pose_jsonl(path))
    rate = (valid / raw * 100) if raw else 0.0
    rows.append((cid, raw, valid, rate))
    print(f"{cid:12} {raw:>6} {valid:>6} {rate:>5.0f}%")

bad = [cid for cid, _, _, rate in rows if rate < 50]
print("-" * 34)
print(f"clips below 50% survival (candidate excludes): {bad or 'none'}")
