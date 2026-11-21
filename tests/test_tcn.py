"""Tests for the TCN model and the TrainableClassifier wrapper.

These require torch; the whole module is skipped cleanly where torch is absent,
preserving the torch-free guarantee of classifier.py's import path.
"""

import pytest

torch = pytest.importorskip("torch")

from src.temporal import schema
from src.temporal.classifier import FEATURE_LEN, PhaseClassifier, TrainableClassifier
from src.temporal.tcn import TCN

WINDOW = 12


def _window(val=0.5):
    return [[val] * FEATURE_LEN for _ in range(WINDOW)]


def test_tcn_forward_shape():
    model = TCN(feature_len=FEATURE_LEN, num_classes=len(schema.PHASES))
    x = torch.zeros(4, WINDOW, FEATURE_LEN)  # batch 4
    out = model(x)
    assert out.shape == (4, len(schema.PHASES))


def test_tcn_handles_various_window_lengths():
    # Causal/dilated convs with left padding must accept different lengths.
    model = TCN(feature_len=FEATURE_LEN, num_classes=len(schema.PHASES))
    for length in (1, 5, 30):
        out = model(torch.zeros(2, length, FEATURE_LEN))
        assert out.shape == (2, len(schema.PHASES))


def test_trainable_classifier_satisfies_protocol():
    clf = TrainableClassifier.from_new()
    assert isinstance(clf, PhaseClassifier)


def test_trainable_classifier_predict_returns_valid_phase():
    clf = TrainableClassifier.from_new()
    phase = clf.predict(_window())
    assert phase in schema.PHASES


def test_trainable_classifier_rejects_bad_window():
    clf = TrainableClassifier.from_new()
    with pytest.raises(ValueError):
        clf.predict([[0.0, 0.0, 0.0]])  # wrong feature length
    with pytest.raises(ValueError):
        clf.predict([])  # empty


def test_predict_proba_is_distribution():
    clf = TrainableClassifier.from_new()
    probs = clf.predict_proba(_window())
    assert set(probs) == set(schema.PHASES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_save_load_round_trip_preserves_predictions(tmp_path):
    clf = TrainableClassifier.from_new(channels=[16, 16])
    w = _window(0.3)
    before = clf.predict_proba(w)

    path = tmp_path / "tcn.pt"
    clf.save(str(path))
    reloaded = TrainableClassifier.load(str(path))
    after = reloaded.predict_proba(w)

    for phase in schema.PHASES:
        assert abs(before[phase] - after[phase]) < 1e-6


def test_model_can_overfit_a_tiny_batch():
    # Sanity check that the architecture learns: two clearly different windows
    # should be separable after a few steps.
    torch.manual_seed(0)
    model = TCN(feature_len=FEATURE_LEN, num_classes=len(schema.PHASES),
                channels=[16, 16])
    # Window A: all low; Window B: all high.
    xa = torch.full((1, WINDOW, FEATURE_LEN), 0.1)
    xb = torch.full((1, WINDOW, FEATURE_LEN), 0.9)
    x = torch.cat([xa, xb], dim=0)
    y = torch.tensor([0, 1])

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    first_loss = None
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss  # learning happened
    model.eval()
    assert model(x).argmax(dim=1).tolist() == [0, 1]  # overfit the two samples


# --- velocity feature mode --------------------------------------------------

def test_velocity_model_consumes_coordinate_windows():
    # A velocity model is built at 2x channels but still RECEIVES coordinate
    # windows; the wrapper adds velocity internally.
    from src.temporal.features import feature_len
    clf = TrainableClassifier.from_new(use_velocity=True)
    assert clf.model.hparams["feature_len"] == feature_len(True)
    phase = clf.predict(_window())      # coordinate-only window
    assert phase in schema.PHASES


def test_velocity_flag_survives_save_load(tmp_path):
    clf = TrainableClassifier.from_new(use_velocity=True, channels=[16, 16])
    w = _window(0.3)
    before = clf.predict_proba(w)
    path = tmp_path / "vel.pt"
    clf.save(str(path))
    reloaded = TrainableClassifier.load(str(path))
    assert reloaded.use_velocity is True
    after = reloaded.predict_proba(w)
    for phase in schema.PHASES:
        assert abs(before[phase] - after[phase]) < 1e-6


def test_coordinate_and_velocity_models_have_different_input_sizes():
    coord = TrainableClassifier.from_new(use_velocity=False)
    vel = TrainableClassifier.from_new(use_velocity=True)
    assert vel.model.hparams["feature_len"] == 2 * coord.model.hparams["feature_len"]
