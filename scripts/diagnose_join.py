"""Diagnose why a clip yields zero training windows: compare the frame ranges
present in the pose JSONL vs the annotation, and their overlap.

Usage:
    python -m scripts.diagnose_join rally47
"""

import contextlib
import io
import sys

from src.temporal.annotation import AnnotationSet
from src.temporal import schema
from src.temporal.contracts import load_pose_jsonl

clip_id = sys.argv[1] if len(sys.argv) > 1 else "rally47"
anno_path = f"data/annotations/{clip_id}.jsonl"
pose_path = f"data/clips/{clip_id}/pose.jsonl"

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    poses = load_pose_jsonl(pose_path)
pose_frames = sorted(r["frame"] for r in poses)
pose_players = sorted({r["player_id"] for r in poses})

anno = AnnotationSet.load(anno_path)
anno_players = sorted({s.player_id for s in anno})

# All annotated frames, and the subset that is a trainable phase (not ignore).
anno_frames = set()
phase_frames = set()
for s in anno:
    anno_frames.update(s.frames())
    if not schema.is_ignore(s.label):
        phase_frames.update(s.frames())

print(f"clip: {clip_id}")
print(f"pose: {len(poses)} valid records, "
      f"frames {pose_frames[0] if pose_frames else '-'}.."
      f"{pose_frames[-1] if pose_frames else '-'} (player ids {pose_players})")
print(f"anno: frames {min(anno_frames)}..{max(anno_frames)} "
      f"(player ids {anno_players}); "
      f"{len(phase_frames)} phase frames, "
      f"{len(anno_frames) - len(phase_frames)} ignore frames")

overlap = sorted(set(pose_frames) & phase_frames)
print(f"frames with BOTH a pose record and a trainable phase: {len(overlap)}")
if overlap:
    print(f"  overlap range: {overlap[0]}..{overlap[-1]}")

# Longest run of consecutive overlapping frames (what windowing needs).
longest = cur = 0
prev = None
for f in overlap:
    cur = cur + 1 if prev is not None and f == prev + 1 else 1
    longest = max(longest, cur)
    prev = f
print(f"  longest consecutive overlapping run: {longest}")

# Player-id mismatch is a common cause: annotations use one id, pose another.
if pose_players and anno_players and not set(pose_players) & set(anno_players):
    print("  !! player_id MISMATCH: pose and annotation share no player ids "
          "-> join yields nothing (likely an upstream tracking bug, not an "
          "annotation gap).")
