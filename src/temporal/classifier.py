"""Phase classifiers for the temporal interface.

Per the Person 4 brief ("build the interface first; do not start by training a
huge model"), this module defines a *swappable* classifier interface and ships a
rule-based implementation that works today. A trainable implementation is a
documented placeholder: it slots into the same interface later without changing
any calling code (the windower, demo overlay, etc.).

Interface contract (:class:`PhaseClassifier`):
    predict(window: list[list[float]]) -> str
        Given a temporal window of per-frame pose feature vectors
        (oldest -> newest, each of length ``2 * NUM_KEYPOINTS``), return one of
        ``schema.PHASES``.

The rule-based classifier derives simple motion cues from the window:
- vertical/horizontal displacement of the body centre over the window
  (movement vs. idle),
- wrist motion magnitude (COCO keypoints 9 and 10) as a proxy for a swing
  (a stroke).
These are deliberately crude -- the point is a working, inspectable interface
with real temporal reasoning, not accuracy.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from src.temporal import schema
from src.temporal.contracts import LEFT_WRIST, NUM_KEYPOINTS, RIGHT_WRIST

FEATURE_LEN = 2 * NUM_KEYPOINTS


@runtime_checkable
class PhaseClassifier(Protocol):
    """A classifier mapping a temporal pose window to a POC phase string."""

    def predict(self, window: List[List[float]]) -> str:
        ...


def _validate_window(window: List[List[float]]) -> None:
    if not window:
        raise ValueError("window must be non-empty")
    for i, frame in enumerate(window):
        if len(frame) != FEATURE_LEN:
            raise ValueError(
                f"frame {i} has length {len(frame)}, expected {FEATURE_LEN} "
                f"(2 * {NUM_KEYPOINTS} keypoints). The rule-based classifier "
                "operates on coordinate-only windows (no velocity)."
            )


def _keypoint_xy(frame: List[float], kp_index: int) -> tuple[float, float]:
    return frame[2 * kp_index], frame[2 * kp_index + 1]


def _body_centre(frame: List[float]) -> tuple[float, float]:
    xs = frame[0::2]
    ys = frame[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class RuleBasedClassifier:
    """Heuristic phase classifier over a pose-feature window.

    Thresholds are expressed in bbox-normalized units (keypoints are in [0, 1]),
    so they are resolution-independent. They are tunable and intentionally
    simple; the trainable classifier will replace the heuristics later.
    """

    def __init__(
        self,
        swing_threshold: float = 0.08,
        movement_threshold: float = 0.04,
    ):
        self.swing_threshold = swing_threshold
        self.movement_threshold = movement_threshold

    def predict(self, window: List[List[float]]) -> str:
        _validate_window(window)
        first, last = window[0], window[-1]

        # --- swing cue: max wrist displacement across the window ---
        swing = 0.0
        for wrist in (LEFT_WRIST, RIGHT_WRIST):
            x0, y0 = _keypoint_xy(first, wrist)
            for frame in window[1:]:
                x, y = _keypoint_xy(frame, wrist)
                disp = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
                swing = max(swing, disp)

        # --- body translation cue over the window ---
        cx0, cy0 = _body_centre(first)
        cx1, cy1 = _body_centre(last)
        body_shift = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5

        # --- decision ---
        # The heuristic only separates three motion-distinguishable phases from
        # pose alone: a swing -> stroke, translation -> movement, else the
        # player is settled -> ready. It cannot tell ready from recovery, nor
        # active-play idle from dead-time idle, without context (direction
        # relative to the last hit, rally state). That is expected: the learned
        # TrainableClassifier is what resolves those. The rule-based stand-in
        # returns a valid-but-coarse subset of schema.PHASES.
        if swing >= self.swing_threshold:
            return schema.PHASE_STROKE
        if body_shift >= self.movement_threshold:
            return schema.PHASE_MOVEMENT
        return schema.PHASE_READY


class TrainableClassifier:
    """Learned temporal model (TCN over pose sequences) behind the
    :class:`PhaseClassifier` interface.

    Wraps the torch ``TCN`` model so the rest of the system (windower, pipeline,
    demo) treats it identically to the rule-based baseline: ``predict(window) ->
    phase``. torch is imported lazily (only when this class is constructed), so
    importing ``classifier.py`` stays torch-free.

    Typical use::

        clf = TrainableClassifier.from_new(window_size=30)  # fresh model
        # ... train via src/temporal/train.py, which saves a checkpoint ...
        clf = TrainableClassifier.load("outputs/phase_tcn.pt")  # trained model
        pipeline = PhasePipeline(window_size=30, classifier=clf)
    """

    def __init__(self, model, use_velocity: bool, device: str = "cpu"):
        # ``model`` is a torch ``TCN``. Use the factory methods below rather than
        # constructing directly in most cases. ``use_velocity`` must match how
        # the model was built/trained -- it governs the feature transform applied
        # to incoming coordinate windows.
        import torch  # lazy

        self._torch = torch
        self.device = device
        self.use_velocity = use_velocity
        self.model = model.to(device)
        self.model.eval()

    # --- factories --------------------------------------------------------

    @classmethod
    def from_new(
        cls,
        num_classes: int | None = None,
        channels: List[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_velocity: bool = False,
        device: str = "cpu",
    ) -> "TrainableClassifier":
        """Create a wrapper around a freshly-initialised TCN.

        ``use_velocity`` selects the feature set: when True the model consumes
        coordinate+velocity frames (``features.feature_len(True)`` channels) and
        velocity is computed from each incoming window; when False, coordinates
        only.
        """
        from src.temporal.tcn import TCN
        from src.temporal.features import feature_len

        num_classes = num_classes if num_classes is not None else len(schema.PHASES)
        model = TCN(
            feature_len=feature_len(use_velocity),
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        return cls(model, use_velocity=use_velocity, device=device)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TrainableClassifier":
        """Reconstruct a trained model from a checkpoint saved by ``save``."""
        import torch
        from src.temporal.tcn import TCN

        ckpt = torch.load(path, map_location=device)
        model = TCN(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        # Default False for older checkpoints that predate the velocity feature.
        use_velocity = ckpt.get("use_velocity", False)
        return cls(model, use_velocity=use_velocity, device=device)

    def save(self, path: str) -> None:
        """Persist architecture hyperparameters + weights (and the feature flag)
        so ``load`` reconstructs an identically-configured model."""
        self._torch.save(
            {
                "hparams": self.model.hparams,
                "state_dict": self.model.state_dict(),
                "use_velocity": self.use_velocity,
            },
            path,
        )

    # --- inference --------------------------------------------------------

    def _window_to_tensor(self, window: List[List[float]]):
        # ``window`` arrives as coordinate-only frames (the same shape the
        # rule-based classifier and the pipeline use). Apply the configured
        # feature transform HERE so train and inference stay in lockstep.
        from src.temporal.features import transform_window

        if not window:
            raise ValueError("window must be non-empty")
        for i, frame in enumerate(window):
            if len(frame) != FEATURE_LEN:
                raise ValueError(
                    f"frame {i} has length {len(frame)}, expected coordinate "
                    f"length {FEATURE_LEN} (velocity is added internally if the "
                    "model was trained with it)"
                )
        feats = transform_window(window, self.use_velocity)
        return self._torch.tensor([feats], dtype=self._torch.float32, device=self.device)

    def predict(self, window: List[List[float]]) -> str:
        with self._torch.no_grad():
            logits = self.model(self._window_to_tensor(window))
            idx = int(logits.argmax(dim=1).item())
        return schema.INDEX_TO_PHASE[idx]

    def predict_proba(self, window: List[List[float]]) -> dict:
        """Return a ``{phase: probability}`` dict for inspection/debugging."""
        with self._torch.no_grad():
            logits = self.model(self._window_to_tensor(window))
            probs = self._torch.softmax(logits, dim=1)[0].tolist()
        return {schema.INDEX_TO_PHASE[i]: p for i, p in enumerate(probs)}
