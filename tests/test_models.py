"""Smoke tests: every model builds, has the right outputs and survives one training step.
Run: python -m pytest tests -q   (no ImageNet download: the transfer model uses weights=None)"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import NUM_COMBINED, NUM_FRUITS  # noqa: E402
from src.models import build_model, compile_model, label_mode_of, set_backbone_trainable  # noqa: E402

BATCH = 4


def fake_batch(label_mode):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (BATCH, 224, 224, 3)).astype("float32")
    if label_mode == "combined":
        return x, rng.integers(0, NUM_COMBINED, BATCH)
    return x, {"fruit": rng.integers(0, NUM_FRUITS, BATCH), "freshness": rng.integers(0, 2, BATCH).astype("float32")}


@pytest.mark.parametrize("name", ["simple_cnn", "multitask_cnn", "transfer"])
def test_outputs_and_one_training_step(name):
    model = build_model(name, **({"weights": None} if name == "transfer" else {}))
    mode = label_mode_of(name)
    compile_model(model, mode, 1e-3)
    x, y = fake_batch(mode)
    out = model.predict(x, verbose=0)
    if mode == "combined":
        assert out.shape == (BATCH, NUM_COMBINED)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-4)
    else:
        assert out["fruit"].shape == (BATCH, NUM_FRUITS)
        assert out["freshness"].shape == (BATCH, 1)
        assert np.all((out["freshness"] >= 0) & (out["freshness"] <= 1))
    logs = model.train_on_batch(x, y, return_dict=True)
    assert np.isfinite(logs["loss"])


def test_transfer_model_freezing():
    model = build_model("transfer", weights=None)
    frozen = set_backbone_trainable(model, None)
    assert frozen["unfrozen_layers"] == 0
    head_params = frozen["trainable_params"]

    tuned = set_backbone_trainable(model, "block_13_expand")
    assert tuned["unfrozen_layers"] > 0 and tuned["trainable_params"] > head_params
    bn_trainable = [l.name for l in model.layers if l.__class__.__name__ == "BatchNormalization" and l.trainable]
    assert bn_trainable == []  # batch norm stays in inference mode while fine-tuning

    full = set_backbone_trainable(model, "all")
    assert full["trainable_params"] > tuned["trainable_params"]
