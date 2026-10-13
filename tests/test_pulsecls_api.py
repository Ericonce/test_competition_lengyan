from __future__ import annotations

from pathlib import Path

from pulsecls import PulseCLS
from pulsecls.cfg import DEFAULT_CFG_DICT, get_cfg
from pulsecls.models.classify import ClassificationPredictor, ClassificationTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_default_cfg_has_canonical_task_and_mode() -> None:
    assert DEFAULT_CFG_DICT["task"] == "classify"
    assert DEFAULT_CFG_DICT["mode"] == "train"
    assert "pipeline" in DEFAULT_CFG_DICT


def test_pulsecls_task_map() -> None:
    model = PulseCLS(ROOT / "configs/pipeline/modular_multigrain_iq_synthetic.yaml")
    assert model.task == "classify"
    assert model.task_map["classify"]["trainer"] is ClassificationTrainer
    assert model.task_map["classify"]["predictor"] is ClassificationPredictor
    assert model.pipeline["data_cfg"]["num_classes"] == get_cfg(ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml")["data_cfg"]["num_classes"]


def test_get_cfg_applies_train_overrides() -> None:
    pipeline = get_cfg(
        ROOT / "configs/pipeline/modular_multigrain_iq_synthetic.yaml",
        overrides={"epochs": 3, "device": "cpu", "batch_size": 8},
    )
    assert pipeline["train"]["epochs"] == 3
    assert pipeline["train"]["device"] == "cpu"
    assert pipeline["train"]["batch_size"] == 8
