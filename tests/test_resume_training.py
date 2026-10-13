from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import pandas as pd

from pulsecls import PulseCLS
from pulsecls.engine.trainer import BaseTrainer


def pipeline(folder, epochs, **overrides):
    return {
        "task": "classify", "output_dir": str(folder),
        "data_cfg": {"num_classes": 2, "sequence_length": 32, "features": ["CF", "PW", "TOA"]},
        "model_cfg": {"type": "cnn1d", "in_channels": 3, "hidden_channels": [8, 16],
                      "stream_embedding_dim": 8, "contrast_dim": 4, "dropout": .2},
        "train": {"epochs": epochs, "device": "cpu", "seed": 2026, "batch_size": 4,
                  "lr": .001, "use_amp": False, "early_stopping_patience": 100, **overrides},
        "augment": {"enabled": False}, "loss": {"ce": {"weight": 1.}},
    }


def train(cfg):
    data = np.random.default_rng(7).normal(size=(8, 32, 3)).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    with patch("pulsecls.engine.trainer.load_split_arrays", return_value=(data, labels, ["sample"] * 8)):
        return PulseCLS().train(pipeline=cfg)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_resume_additional_epochs_and_optimizer_rng_equivalence(tmp_path):
    continuous, initial, resumed = (tmp_path / p for p in ("continuous", "initial", "resumed"))
    train(pipeline(continuous, 3))
    train(pipeline(initial, 1))
    source = initial / "last_model.pt"
    before = load(source)
    # A different configured LR must not replace the restored optimizer LR.
    train(pipeline(resumed, 2, resume=True, resume_checkpoint=str(source), lr=.8))
    actual, expected = load(resumed / "last_model.pt"), load(continuous / "last_model.pt")
    assert actual["epoch"] == 3
    assert actual["optimizer_state_dict"]["param_groups"][0]["lr"] != .8
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
    for key, value in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key], value, rtol=0, atol=0)
    for key, state in expected["optimizer_state_dict"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(actual["optimizer_state_dict"]["state"][key][field], value)
    assert pd.read_csv(resumed / "training_history.csv")["epoch"].tolist() == [1, 2, 3]
    assert before["epoch"] == load(source)["epoch"] == 1
    assert (resumed / "best_model.pt").exists()


def test_resume_disabled_ignores_invalid_path(tmp_path):
    cfg = pipeline(tmp_path, 1, resume=False, resume_checkpoint="missing.pt")
    train(cfg)
    assert load(tmp_path / "last_model.pt")["epoch"] == 1


def test_resume_requires_path(tmp_path):
    with pytest.raises(ValueError, match="requires resume_checkpoint"):
        BaseTrainer(pipeline(tmp_path, 1, resume=True)).fit()


def test_old_weight_only_checkpoint_continues_from_saved_epoch(tmp_path):
    source_dir, destination = tmp_path / "old", tmp_path / "new"
    train(pipeline(source_dir, 1))
    payload = load(source_dir / "best_model.pt")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state", "history"):
        payload.pop(key, None)
    source = source_dir / "legacy.pt"
    torch.save(payload, source)
    with pytest.warns(UserWarning, match="Legacy weight-only"):
        train(pipeline(destination, 2, resume=True, resume_checkpoint=str(source)))
    assert load(destination / "last_model.pt")["epoch"] == 3
    assert pd.read_csv(destination / "training_history.csv")["epoch"].tolist() == [1, 2, 3]


def test_incompatible_features_rejected_before_training(tmp_path):
    train(pipeline(tmp_path / "source", 1))
    cfg = pipeline(tmp_path / "bad", 1, resume=True, resume_checkpoint=str(tmp_path / "source/last_model.pt"))
    cfg["data_cfg"]["features"] = ["PW", "CF", "TOA"]
    with pytest.raises(ValueError, match="features"):
        BaseTrainer(cfg).fit()


def test_best_checkpoint_and_normalization_preserved_without_improvement(tmp_path):
    train(pipeline(tmp_path / "source", 1))
    source = tmp_path / "source/last_model.pt"
    saved = load(source)
    # Force a known previous best so no new validation can supersede it.
    saved["best_checkpoint"]["val_metrics"]["macro_f1"] = 2.
    saved["best_checkpoint"]["epoch"] = 1
    torch.save(saved, source)
    train(pipeline(tmp_path / "new", 1, resume=True, resume_checkpoint=str(source)))
    result = load(tmp_path / "new/best_model.pt")
    assert result["epoch"] == 1
    np.testing.assert_array_equal(result["mean"], saved["mean"])
    for key, value in saved["best_checkpoint"]["model_state_dict"].items():
        torch.testing.assert_close(result["model_state_dict"][key], value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_state_restores_and_training_continues(tmp_path):
    train(pipeline(tmp_path / "first", 1, device="cuda:0", use_amp=True))
    source = tmp_path / "first/last_model.pt"
    saved = load(source)
    assert saved["scaler_state_dict"]
    train(pipeline(tmp_path / "second", 1, device="cuda:0", use_amp=True,
                   resume=True, resume_checkpoint=str(source)))
    continued = load(tmp_path / "second/last_model.pt")
    assert continued["epoch"] == 2
    assert continued["scaler_state_dict"]
    assert all(torch.isfinite(v).all() for v in continued["model_state_dict"].values() if v.is_floating_point())
