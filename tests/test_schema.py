"""Tests for the phase label schema and config projection."""

from src.temporal import schema
from src.temporal.config import active_projection, load_config


def test_phase_vocabulary_is_the_five_phases():
    assert schema.PHASES == [
        schema.PHASE_READY,
        schema.PHASE_MOVEMENT,
        schema.PHASE_STROKE,
        schema.PHASE_RECOVERY,
        schema.PHASE_IDLE,
    ]
    assert len(schema.PHASES) == len(set(schema.PHASES)) == 5


def test_annotation_labels_are_phases_plus_ignore_sentinel():
    # Annotators assign the five phases directly, plus the ignore sentinel.
    assert schema.ANNOTATION_LABELS == schema.PHASES + [schema.IGNORE_LABEL]
    # The sentinel is NOT a trainable phase.
    assert schema.IGNORE_LABEL not in schema.PHASES
    assert schema.IGNORE_LABEL not in schema.PHASE_TO_INDEX


def test_ignore_label_is_recognised_and_not_projectable():
    assert schema.is_ignore(schema.IGNORE_LABEL)
    assert not schema.is_ignore(schema.PHASE_STROKE)
    # Projecting the sentinel is a caller error -> loud failure.
    try:
        schema.project_label(schema.IGNORE_LABEL)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError projecting the ignore sentinel")


def test_phase_indices_are_stable_and_bijective():
    assert schema.PHASES == list(schema.PHASE_TO_INDEX.keys())
    for phase, idx in schema.PHASE_TO_INDEX.items():
        assert schema.INDEX_TO_PHASE[idx] == phase


def test_identity_projection_is_total_over_trainable_labels():
    schema.validate_projection()  # default identity projection, should not raise
    # Projection is total over PHASES; the ignore sentinel is exempt.
    for label in schema.PHASES:
        phase = schema.project_label(label)
        assert phase == label  # identity
        assert phase in schema.PHASES


def test_project_label_raises_on_unknown_label():
    try:
        schema.project_label("not_a_real_label")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown label")


def test_validate_projection_detects_missing_label():
    incomplete = dict(schema.IDENTITY_PROJECTION)
    incomplete.pop(schema.ANNOTATION_LABELS[0])
    try:
        schema.validate_projection(incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for incomplete projection")


def test_validate_projection_detects_bad_target_phase():
    broken = dict(schema.IDENTITY_PROJECTION)
    broken[schema.ANNOTATION_LABELS[0]] = "not_a_phase"
    try:
        schema.validate_projection(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad target phase")


def test_future_layers_are_recorded_but_separate():
    # Stroke types and rally states exist for reference but are NOT part of the
    # phase vocabulary the current model trains on.
    assert "smash" in schema.FUTURE_STROKE_TYPES
    assert "rally_active" in schema.FUTURE_RALLY_STATES
    for label in schema.FUTURE_STROKE_TYPES + schema.FUTURE_RALLY_STATES:
        assert label not in schema.PHASES


def test_config_loads_and_matches_schema():
    cfg = load_config()
    assert cfg["project"]["discipline"] == "singles"
    assert set(cfg["annotation"]["labels"]) == set(schema.ANNOTATION_LABELS)


def test_config_projection_matches_code_projection():
    cfg = load_config()
    proj = active_projection(cfg)
    assert proj == schema.IDENTITY_PROJECTION
    schema.validate_projection(proj)
