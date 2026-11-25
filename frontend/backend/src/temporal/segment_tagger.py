"""Interactive segment tagger for phase/action labels.

A keyboard-driven OpenCV review tool to label time-segments of a clip. It is a
thin shell over the data layer in ``annotation.py`` -- all persistence and
validation live there, so this file stays small and does not need unit tests
(it is interactive). It imports cleanly even when OpenCV is absent; the
interactive loop only requires cv2 when actually run.

Workflow
--------
Open a clip, scrub to the start of an action, press ``[`` to mark the segment
start, scrub to the end, press ``]`` to mark the end, then press a label key to
assign a label and save the segment. Repeat. Press ``w`` to write the
annotation file, ``q`` to quit (auto-saves).

Run::

    .\\.venv\\Scripts\\python.exe -m src.temporal.segment_tagger \\
        --video data/videos/rally01.mp4 \\
        --out data/annotations/rally01.jsonl \\
        --clip-id rally01 --player-id 1

Labels are read from ``schema.ANNOTATION_LABELS`` and bound to keys 1..9, 0,
and a few letters, shown in the on-screen legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from src.temporal import schema
from src.temporal.annotation import AnnotationSet, Segment

WINDOW_NAME = "RallyMirror segment tagger"

# Max on-screen display size. The window scales the view to fit within this box
# (preserving aspect ratio); the underlying frame stays at native resolution.
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720


def _fit_within(w: int, h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Scale (w, h) down to fit within (max_w, max_h), preserving aspect ratio.
    Never upscales (clips smaller than the box keep their size).
    """
    if w <= 0 or h <= 0:
        return max_w, max_h
    scale = min(max_w / w, max_h / h, 1.0)
    return max(1, round(w * scale)), max(1, round(h * scale))


def _build_keymap() -> Dict[str, str]:
    """Map single keys to labels. Digits 1-9 then 0, then letters a, s, d..."""
    keys = list("1234567890") + list("zxcvbnm")
    labels = schema.ANNOTATION_LABELS
    if len(labels) > len(keys):
        raise RuntimeError("more labels than available keys; extend the keymap")
    return {keys[i]: labels[i] for i in range(len(labels))}


def _legend_lines(keymap: Dict[str, str]) -> List[str]:
    return [f"{k}: {label}" for k, label in keymap.items()]


def run_tagger(
    video: str,
    out: str,
    clip_id: str,
    player_id: int = 1,
) -> None:  # pragma: no cover (interactive)
    """Launch the interactive tagger. Requires OpenCV."""
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "OpenCV is required for the interactive tagger. Install the project "
            "requirements (opencv-python) and run inside the project venv."
        )

    keymap = _build_keymap()

    # Resume from an existing file if present, so tagging is incremental.
    annotations = AnnotationSet.load(out) if Path(out).exists() else AnnotationSet()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    # A resizable window lets OpenCV scale the *display* to fit the screen while
    # the underlying frame stays at native resolution (so HUD/overlay pixel
    # coordinates remain correct). We size it once to fit within a max display
    # box, preserving aspect ratio -- this is what keeps large clips (e.g.
    # 1080p) from spilling off-screen without resizing the frame data itself.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    disp_w, disp_h = _fit_within(src_w, src_h, MAX_DISPLAY_W, MAX_DISPLAY_H)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)

    # Scale HUD text/line thickness by frame height so it looks consistent
    # across resolutions (tuned against 720p).
    hud_scale = src_h / 720.0

    frame_idx = 0
    pending_start: Optional[int] = None
    status = "mark start with [  then end with ]  then a label key"

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return frame if ok else None

    legend = _legend_lines(keymap)

    while True:
        frame = read_frame(frame_idx)
        if frame is None:
            frame_idx = max(0, frame_idx - 1)
            continue

        hud = frame.copy()
        thick = max(1, round(2 * hud_scale))
        thin = max(1, round(hud_scale))
        y = round(24 * hud_scale)
        cv2.putText(hud, f"frame {frame_idx}/{total-1}  player {player_id}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * hud_scale,
                    (0, 255, 255), thick)
        y += round(26 * hud_scale)
        sel = f"start={pending_start}" if pending_start is not None else "start=-"
        cv2.putText(hud, f"{sel}   {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * hud_scale, (0, 255, 0), thick)
        y += round(30 * hud_scale)
        for line in legend:
            cv2.putText(hud, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * hud_scale, (200, 200, 200), thin)
            y += round(18 * hud_scale)
        cv2.putText(hud, "arrows/./, step  [ ] mark  w write  q quit",
                    (10, hud.shape[0] - round(12 * hud_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * hud_scale, (255, 255, 255), thin)

        cv2.imshow(WINDOW_NAME, hud)
        key = cv2.waitKey(0) & 0xFF
        ch = chr(key) if 32 <= key < 127 else ""

        if ch == "q":
            break
        elif ch == "w":
            annotations.save(out)
            status = f"saved {len(annotations)} segments -> {out}"
        elif ch in (".", "d") or key == 83:   # step forward / right-arrow
            frame_idx = min(total - 1, frame_idx + 1)
        elif ch in (",", "a") or key == 81:   # step back / left-arrow
            frame_idx = max(0, frame_idx - 1)
        elif ch == "m":                         # jump forward 10
            frame_idx = min(total - 1, frame_idx + 10)
        elif ch == "n":                         # jump back 10
            frame_idx = max(0, frame_idx - 10)
        elif ch == "[":
            pending_start = frame_idx
            status = "start marked; scrub to end and press ]"
        elif ch == "]":
            if pending_start is None:
                status = "mark a start first with ["
            else:
                status = "end marked; press a label key to assign"
        elif ch in keymap:
            if pending_start is None:
                status = "mark start [ and end ] before labeling"
            else:
                start = min(pending_start, frame_idx)
                end = max(pending_start, frame_idx)
                try:
                    seg = Segment(clip_id, player_id, start, end, keymap[ch])
                    annotations.add(seg)
                    annotations.save(out)
                    status = f"added {keymap[ch]} [{start}-{end}] ({len(annotations)} total)"
                    pending_start = None
                except ValueError as exc:
                    status = f"rejected: {exc}"

    annotations.save(out)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[tagger] saved {len(annotations)} segments to {out}")


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="RallyMirror segment tagger")
    parser.add_argument("--video", required=True, help="path to the clip")
    parser.add_argument("--out", required=True, help="annotation JSONL to write")
    parser.add_argument("--clip-id", required=True, help="clip identifier")
    parser.add_argument("--player-id", type=int, default=1)
    args = parser.parse_args(argv)
    run_tagger(args.video, args.out, args.clip_id, args.player_id)


if __name__ == "__main__":  # pragma: no cover
    main()
