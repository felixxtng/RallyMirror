"""Temporal phase-recognition pipeline (the public interface).

This is the single entry point the demo (and, later, real callers) use. It wires
together the three pieces Person 4 owns:

    pose record (Contract 2)  ->  windower  ->  classifier  ->  phase

Usage (streaming, one frame at a time):

    pipe = PhasePipeline(window_size=30)
    for pose_rec in pose_stream:            # conforms to Contract 2
        result = pipe.process(pose_rec)
        if result is not None:
            frame, player_id, phase = result
            # ... overlay / log the phase

Until a track has accumulated ``window_size`` consecutive frames, ``process``
returns ``None`` for that frame (not enough temporal context yet). The optional
``smoothing`` majority filter reduces single-frame flicker in the emitted phase.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from src.temporal.classifier import PhaseClassifier, RuleBasedClassifier
from src.temporal.contracts import pose_record_to_vector, validate_pose_record
from src.temporal.windower import MultiTrackWindower


class PhasePipeline:
    def __init__(
        self,
        window_size: int = 30,
        classifier: Optional[PhaseClassifier] = None,
        smoothing: int = 1,
    ):
        if smoothing < 1:
            raise ValueError("smoothing must be >= 1")
        self.windower = MultiTrackWindower(window_size)
        self.classifier: PhaseClassifier = classifier or RuleBasedClassifier()
        self.smoothing = smoothing
        self._recent: Dict[int, Deque[str]] = defaultdict(
            lambda: deque(maxlen=smoothing)
        )

    def process(self, pose_record: dict) -> Optional[Tuple[int, int, str]]:
        """Feed one validated pose record; return (frame, player_id, phase) once
        enough temporal context exists for that track, else ``None``.
        """
        validate_pose_record(pose_record)
        player_id = pose_record["player_id"]
        frame = pose_record["frame"]
        features = pose_record_to_vector(pose_record)

        window = self.windower.push(player_id, frame, features)
        if window is None:
            return None

        # ``window`` is coordinate-only. Classifiers apply their own feature
        # transform internally (the rule-based one uses coordinates; the TCN
        # wrapper adds velocity if it was trained with it), so the pipeline
        # hands over the raw coordinate window unchanged -- a single convention
        # that keeps train and inference in lockstep.
        raw_phase = self.classifier.predict(window)
        phase = self._smooth(player_id, raw_phase)
        return frame, player_id, phase

    def _smooth(self, player_id: int, phase: str) -> str:
        if self.smoothing == 1:
            return phase
        recent = self._recent[player_id]
        recent.append(phase)
        # Majority vote over the recent window; ties resolved by most recent.
        counts: Dict[str, int] = {}
        for p in recent:
            counts[p] = counts.get(p, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == phase))
        return best[0]

    def reset(self) -> None:
        self.windower.reset_all()
        self._recent.clear()
