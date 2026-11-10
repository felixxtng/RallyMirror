"""Segment-level annotation data model for phase/action labels.

This is the data layer behind the segment tagger. It is deliberately separate
from any UI so it can be unit-tested and reused by the training pipeline.

Core idea: humans label **segments** ("frames 1200-1250 of player 1 = smash"),
not individual frames. A segment is::

    {"clip_id": str, "player_id": int, "start_frame": int, "end_frame": int,
     "label": str}

``label`` must be in ``schema.ANNOTATION_LABELS`` (the fine-grained vocabulary).
Segments are stored as JSONL, one per line.

For training we need **per-frame** labels aligned to the pose stream. The
segment -> frame expansion lives here (``to_frame_labels``), as does the join
with a pose record stream (``build_training_dataset``) which yields
``(feature_vector, phase_index)`` pairs ready for a classifier. Projection from
the fine-grained label to the coarse POC phase is applied at this step (never at
annotation time), using ``schema.project_label``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.temporal import schema
from src.temporal.contracts import pose_record_to_vector


@dataclass(frozen=True)
class Segment:
    clip_id: str
    player_id: int
    start_frame: int
    end_frame: int  # inclusive
    label: str

    def __post_init__(self):
        if self.label not in schema.ANNOTATION_LABELS:
            raise ValueError(
                f"unknown label {self.label!r}; must be one of "
                f"{schema.ANNOTATION_LABELS}"
            )
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) < start_frame ({self.start_frame})"
            )

    def frames(self) -> range:
        return range(self.start_frame, self.end_frame + 1)

    def to_dict(self) -> Dict:
        return asdict(self)


class AnnotationSet:
    """A collection of segments for one or more clips, with validation and IO."""

    def __init__(self, segments: Optional[Iterable[Segment]] = None):
        self._segments: List[Segment] = list(segments or [])

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self):
        return iter(self._segments)

    @property
    def segments(self) -> List[Segment]:
        return list(self._segments)

    def add(self, segment: Segment) -> None:
        """Add a segment, rejecting overlaps with an existing segment for the
        same (clip_id, player_id). Overlap means ambiguous labels for a frame.
        """
        for existing in self._segments:
            if (
                existing.clip_id == segment.clip_id
                and existing.player_id == segment.player_id
                and segment.start_frame <= existing.end_frame
                and existing.start_frame <= segment.end_frame
            ):
                raise ValueError(
                    f"segment {segment.to_dict()} overlaps existing "
                    f"{existing.to_dict()} (same clip/player)"
                )
        self._segments.append(segment)

    # --- IO ---------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for seg in self._segments:
                fh.write(json.dumps(seg.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str) -> "AnnotationSet":
        segments: List[Segment] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                try:
                    segments.append(Segment(**data))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid segment at {path}:{line_no}: {exc}")
        anno = cls()
        # Use add() to enforce the overlap check on load, too.
        for seg in segments:
            anno.add(seg)
        return anno

    # --- Expansion to per-frame labels -----------------------------------

    def to_frame_labels(
        self, clip_id: str, player_id: int
    ) -> Dict[int, str]:
        """Return ``{frame -> fine-grained label}`` for one clip+player.

        Frames not covered by any segment are simply absent (unlabeled).
        """
        frame_labels: Dict[int, str] = {}
        for seg in self._segments:
            if seg.clip_id != clip_id or seg.player_id != player_id:
                continue
            for f in seg.frames():
                frame_labels[f] = seg.label
        return frame_labels

    def players_for_clip(self, clip_id: str) -> List[int]:
        """Distinct annotated player ids for a clip."""
        return sorted({s.player_id for s in self._segments if s.clip_id == clip_id})

    def single_player_frame_labels(self, clip_id: str) -> Dict[int, str]:
        """Return ``{frame -> label}`` for a clip's SINGLE annotated player,
        ignoring player id. For the singles POC where each clip labels exactly
        one player and pose records carry an arbitrary/unstable tracker id.

        Raises
        ------
        ValueError
            If the clip has more than one annotated player (ambiguous in
            single-player mode -- use ``to_frame_labels`` per player instead).
        """
        players = self.players_for_clip(clip_id)
        if len(players) > 1:
            raise ValueError(
                f"clip {clip_id!r} has multiple annotated players {players}; "
                "single-player mode is ambiguous. Use doubles/multi-player join."
            )
        if not players:
            return {}
        return self.to_frame_labels(clip_id, players[0])

    def label_counts(self) -> Dict[str, int]:
        """Total labeled frames per fine-grained label (across all clips)."""
        counts: Dict[str, int] = {}
        for seg in self._segments:
            n = seg.end_frame - seg.start_frame + 1
            counts[seg.label] = counts.get(seg.label, 0) + n
        return counts


def _resolve_frame_labels(
    annotations: "AnnotationSet",
    clip_id: str,
    pose_player_id: int,
    single_player: bool,
    cache: Dict[int, Dict[int, str]],
):
    """Return the frame->label map to use for a pose record.

    In ``single_player`` mode the pose record's ``player_id`` is ignored and the
    clip's single annotated player's labels are used (matches by frame only).
    This handles trackers that assign arbitrary / drifting ids to the one
    player being annotated (e.g. the same player relabeled 1 -> 14 -> 138).
    """
    key = -1 if single_player else pose_player_id
    if key not in cache:
        if single_player:
            cache[key] = annotations.single_player_frame_labels(clip_id)
        else:
            cache[key] = annotations.to_frame_labels(clip_id, pose_player_id)
    return cache[key]


def build_training_dataset(
    annotations: AnnotationSet,
    pose_records: Iterable[Dict],
    clip_id: str,
    projection: Optional[Dict[str, str]] = None,
    single_player: bool = True,
) -> Tuple[List[List[float]], List[int]]:
    """Join annotations with a pose stream into ``(features, phase_indices)``.

    Parameters
    ----------
    annotations:
        The labeled segments.
    pose_records:
        Internal-format pose records (e.g. from ``contracts.load_pose_jsonl``):
        ``{"frame": int, "player_id": int, "keypoints": [[x, y], ...]}``.
    clip_id:
        Which clip these pose records belong to (matches ``Segment.clip_id``).
    projection:
        Annotation-label -> phase mapping. Defaults to
        ``schema.IDENTITY_PROJECTION`` (phases are annotated directly). A future
        fine-grained vocabulary can pass a different projection here.
    single_player:
        Singles POC default (True): match pose records to the annotation by
        FRAME ONLY, ignoring the pose record's ``player_id``, and attribute every
        record to the clip's single annotated player. This assumes one tracked
        body per frame (true of the current tracker/pose pipeline). Set False for
        doubles / multi-player, where records join on their own ``player_id``.

    Returns
    -------
    (X, y) where X is a list of per-frame feature vectors and y is a list of
    phase indices (see ``schema.PHASE_TO_INDEX``). Only frames that have BOTH a
    pose record and a (non-ignore) label are included.
    """
    projection = projection if projection is not None else schema.IDENTITY_PROJECTION

    cache: Dict[int, Dict[int, str]] = {}
    X: List[List[float]] = []
    y: List[int] = []
    for rec in pose_records:
        frame_labels = _resolve_frame_labels(
            annotations, clip_id, rec["player_id"], single_player, cache
        )
        label = frame_labels.get(rec["frame"])
        if label is None or schema.is_ignore(label):
            continue  # frame not annotated, or deliberately marked ignore
        phase = schema.project_label(label, projection)
        X.append(pose_record_to_vector(rec))
        y.append(schema.PHASE_TO_INDEX[phase])
    return X, y


def build_windowed_training_dataset(
    annotations: AnnotationSet,
    pose_records: Iterable[Dict],
    clip_id: str,
    window_size: int,
    stride: int = 1,
    label_from: str = "last",
    projection: Optional[Dict[str, str]] = None,
    single_player: bool = True,
) -> Tuple[List[List[List[float]]], List[int]]:
    """Join annotations with a pose stream into windowed training samples.

    Each sample ``X[i]`` is a ``window_size x FEATURE_LEN`` sequence of
    consecutive per-frame feature vectors -- byte-for-byte the shape the online
    ``TrackWindow``/``PhasePipeline`` produce at inference, so train and infer
    see identical inputs.

    Windowing rules (mirroring ``windower.TrackWindow``):
    - windows are per ``player_id`` and never span a frame gap (a break in
      consecutive frame numbers resets the window);
    - a window is only emitted if EVERY frame in it is annotated with a phase
      (frames that are unlabeled OR marked ``schema.IGNORE_LABEL`` invalidate
      any window containing them);
    - the window label is taken from one frame, selected by ``label_from``.

    Parameters
    ----------
    window_size:
        Number of consecutive frames per sample (must be >= 1).
    stride:
        Step between successive window start positions (default 1 = maximal
        overlap, as at inference). Use a larger stride to reduce correlation /
        dataset size.
    label_from:
        Which frame supplies the window's label: ``"last"`` (default, matches
        the online classifier predicting the current frame from the trailing
        window), ``"center"``, or ``"majority"`` (most common label in-window).
    projection:
        Annotation-label -> phase mapping. Defaults to
        ``schema.IDENTITY_PROJECTION``.
    single_player:
        Singles POC default (True): ignore the pose records' ``player_id`` and
        treat them as ONE stream for the clip's single annotated player, joining
        by frame only. This keeps an unstable tracker id (same player relabeled
        1 -> 14 -> 138) from fragmenting the frame runs. Set False for
        doubles/multi-player, where records are grouped and labeled per id.

    Returns
    -------
    (X, y): X is a list of ``window_size x FEATURE_LEN`` sequences, y a list of
    phase indices.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if label_from not in ("last", "center", "majority"):
        raise ValueError("label_from must be 'last', 'center', or 'majority'")
    projection = projection if projection is not None else schema.IDENTITY_PROJECTION

    # Group pose records into streams. In single-player mode every record is
    # one stream (id ignored); otherwise group by the record's player_id.
    streams: Dict[int, List[Dict]] = {}
    for rec in pose_records:
        key = -1 if single_player else rec["player_id"]
        streams.setdefault(key, []).append(rec)
    for recs in streams.values():
        recs.sort(key=lambda r: r["frame"])

    X: List[List[List[float]]] = []
    y: List[int] = []

    for stream_key, recs in streams.items():
        if single_player:
            frame_labels = annotations.single_player_frame_labels(clip_id)
        else:
            frame_labels = annotations.to_frame_labels(clip_id, stream_key)

        # Split into runs of consecutive frames (no gaps) so no window straddles
        # a discontinuity -- same guarantee as the online windower.
        runs: List[List[Dict]] = []
        current: List[Dict] = []
        prev_frame: Optional[int] = None
        for rec in recs:
            if prev_frame is not None and rec["frame"] != prev_frame + 1:
                runs.append(current)
                current = []
            current.append(rec)
            prev_frame = rec["frame"]
        if current:
            runs.append(current)

        for run in runs:
            for start in range(0, len(run) - window_size + 1, stride):
                window = run[start:start + window_size]
                labels = [frame_labels.get(r["frame"]) for r in window]
                # Reject the window if any frame is unlabeled OR marked ignore.
                # An ignored frame (camera cut, close-up, bad track) breaks
                # windows exactly like a gap -- a window must never stitch
                # across an excluded region.
                if any(lbl is None or schema.is_ignore(lbl) for lbl in labels):
                    continue

                if label_from == "last":
                    chosen = labels[-1]
                elif label_from == "center":
                    chosen = labels[window_size // 2]
                else:  # majority
                    chosen = max(set(labels), key=labels.count)

                phase = schema.project_label(chosen, projection)
                X.append([pose_record_to_vector(r) for r in window])
                y.append(schema.PHASE_TO_INDEX[phase])

    return X, y
