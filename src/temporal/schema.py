"""Label schema for RallyMirror phase/action recognition.

The first training target is a **phase** classifier: every frame of a player is
assigned exactly one of five mutually-exclusive, duration-based phases. Phases
are *states a player occupies over many frames*, deliberately distinct from
*instantaneous events* like the exact contact frame of a hit. Keeping the
vocabulary purely phase-based (no stroke types mixed in) means no class is
starved of frames -- the problem that arises when a brief "smash" moment competes
with the long "ready" stance that immediately follows it.

The five phases (``PHASES``):

1. ``ready``     -- neutral/waiting stance, poised for the opponent's hit.
2. ``movement``  -- travelling toward the shuttle to play it.
3. ``stroke``    -- the hitting action as a WINDOW: loading + swing +
                    follow-through around contact (Option A). May later be split
                    into finer stroke-type sub-labels; for now it is one phase.
4. ``recovery``  -- regaining balance / landing / repositioning toward base
                    after a stroke, before settling back into ``ready``.
5. ``idle``      -- dead time, no active play (between points, after a score).

The natural rally cycle is ready -> movement -> stroke -> recovery -> ready,
with idle as the out-of-rally state.

Annotators label these five phases directly (see ``src/temporal/segment_tagger``).

Future layers (NOT part of the current phase vocabulary; added only once the
phase model performs well):
- **Stroke type** (smash / clear / drop / ... , following BFMD syntax): a
  separate label set predicted by a second model, conditioned on a detected
  ``stroke`` phase.
- **Rally state** (between_points / serve_initiation / rally_active / point_end):
  a separate parallel track describing match structure.
These are intentionally kept out of ``PHASES`` so the first classifier stays
simple. ``FUTURE_STROKE_TYPES`` is recorded below for reference only.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Phase vocabulary (the labels we annotate and train on, first-class)
# ---------------------------------------------------------------------------

PHASE_READY = "ready"
PHASE_MOVEMENT = "movement"
PHASE_STROKE = "stroke"
PHASE_RECOVERY = "recovery"
PHASE_IDLE = "idle"

PHASES: List[str] = [
    PHASE_READY,
    PHASE_MOVEMENT,
    PHASE_STROKE,
    PHASE_RECOVERY,
    PHASE_IDLE,
]

# Sentinel label for frames the annotator deliberately marks UNUSABLE (camera
# close-ups, replays, cuts, bad tracking). It is NOT a phase: the model must
# never predict it, and windows containing an ignored frame are dropped from
# training (treated like a gap). It lives in the annotation vocabulary so the
# tagger offers it and the annotation file is explicit ("I saw this, skip it"),
# but it is deliberately excluded from PHASES / PHASE_TO_INDEX.
IGNORE_LABEL = "ignore"

# What annotators assign, one per segment: the five phases plus the ignore
# sentinel. For the phase model, PHASES (below) is the trainable vocabulary;
# IGNORE_LABEL is an exclusion marker, not a class.
ANNOTATION_LABELS: List[str] = list(PHASES) + [IGNORE_LABEL]

# Stable integer indices (order fixed by ``PHASES``) for training.
PHASE_TO_INDEX: Dict[str, int] = {phase: i for i, phase in enumerate(PHASES)}
INDEX_TO_PHASE: Dict[int, str] = {i: phase for phase, i in PHASE_TO_INDEX.items()}


# ---------------------------------------------------------------------------
# Future layers (reference only -- not annotated or trained yet)
# ---------------------------------------------------------------------------

# Fine-grained stroke types, added as a SEPARATE classifier later (BFMD-aligned).
FUTURE_STROKE_TYPES: List[str] = [
    "serve", "clear", "drop", "smash", "drive",
    "net_shot", "net_kill", "lift", "push", "block",
]

# Rally-state track, added as a SEPARATE parallel annotation later.
FUTURE_RALLY_STATES: List[str] = [
    "between_points", "serve_initiation", "rally_active", "point_end",
]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
#
# With the phase labels now first-class, the "projection" is the identity: each
# annotated label is already a target phase. The hook is kept so a future
# fine-grained vocabulary (e.g. stroke types) can project down to these phases
# via config without code changes -- the same mechanism, a different source
# vocabulary.

# The projection covers the trainable PHASES only. IGNORE_LABEL is intentionally
# NOT a key here: it maps to no phase and is handled as an exclusion by callers
# (see ``is_ignore`` and the dataset builders), not projected.
IDENTITY_PROJECTION: Dict[str, str] = {phase: phase for phase in PHASES}


def is_ignore(label: str) -> bool:
    """True if ``label`` is the deliberate-exclusion sentinel."""
    return label == IGNORE_LABEL


def project_label(label: str, projection: Dict[str, str] | None = None) -> str:
    """Map an annotation label to a phase. Defaults to the identity projection
    (phase labels are annotated directly).

    Raises
    ------
    ValueError
        If called on ``IGNORE_LABEL`` -- ignored frames are not a phase and must
        be filtered out *before* projection, not projected. Catching it here
        surfaces a caller bug loudly.
    KeyError
        If ``label`` is otherwise not covered by the projection.
    """
    if is_ignore(label):
        raise ValueError(
            f"{IGNORE_LABEL!r} is an exclusion marker, not a phase; filter it "
            "out before projecting (see schema.is_ignore)."
        )
    projection = projection if projection is not None else IDENTITY_PROJECTION
    if label not in projection:
        raise KeyError(
            f"Annotation label {label!r} is not covered by the projection. "
            f"Known labels: {sorted(projection)}"
        )
    return projection[label]


def validate_projection(projection: Dict[str, str] | None = None) -> None:
    """Assert the projection is total over the trainable phase labels and that
    every target is a known phase. ``IGNORE_LABEL`` is exempt (it maps to no
    phase by design).

    Raises
    ------
    ValueError
        If any trainable annotation label is unmapped, or maps to an unknown
        phase.
    """
    projection = projection if projection is not None else IDENTITY_PROJECTION

    trainable = [lbl for lbl in ANNOTATION_LABELS if not is_ignore(lbl)]
    missing = [lbl for lbl in trainable if lbl not in projection]
    if missing:
        raise ValueError(f"Projection is missing annotation labels: {missing}")

    bad_targets = {
        lbl: phase for lbl, phase in projection.items() if phase not in PHASE_TO_INDEX
    }
    if bad_targets:
        raise ValueError(
            f"Projection maps labels to unknown phases: {bad_targets}. "
            f"Valid phases: {PHASES}"
        )
