from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pulsecls import PulseCLS
from pulsecls.cfg import REPO_ROOT, get_cfg
from pulsecls.data.dataset import load_split_arrays
from pulsecls.data.synthetic import SyntheticRadarGenerator
from pulsecls.engine.trainer import BaseTrainer


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "raw"
    SyntheticRadarGenerator(num_emitters=3, sequences_per_emitter=4,
        overlap={"enabled": False}, observation={"enabled": False}, noise={"enabled": False},
        sequence={"random_length": False, "length": 16}).generate(root)
    return get_cfg(REPO_ROOT / "configs/pipeline/cnn_softmax_synthetic.yaml", overrides={
        "data": root / "data.yaml", "output_dir": tmp_path / "output", "device": "cpu",
        "sequence_length": 16, "dataset": {"sample_ratio": 1., "train_ratio": .75, "val_ratio": .25},
        "train": {"epochs": 1, "batch_size": 4, "use_amp": False},
        "model_cfg": {"hidden_channels": [8, 16], "stream_embedding_dim": 8, "contrast_dim": 4},
        "inference": {"progress": False}, "data_cfg": {"progress": False},
    })


def test_training_auto_manual_and_code_class_count_priority(cfg):
    assert BaseTrainer(cfg).num_classes == 3
    manual = get_cfg(cfg, overrides={"train": {"num_classes": 5}})
    assert manual["data_cfg"]["num_classes"] == BaseTrainer(manual).num_classes == 5
    assert BaseTrainer(manual, num_classes=6).num_classes == 6
    assert BaseTrainer(manual, **{"train.num_classes": 7}).num_classes == 7
    assert BaseTrainer(manual, data_cfg={"num_classes": 8}).num_classes == 8
    assert BaseTrainer(manual, train={"num_classes": None}, data_cfg={"num_classes": 3}).num_classes == 3


@pytest.mark.parametrize("section", ["train", "inference"])
@pytest.mark.parametrize("value", [0, -1, True, 2.5, "10"])
def test_invalid_class_count_is_rejected_before_training_or_inference(cfg, section, value):
    with pytest.raises(ValueError, match=f"{section}.num_classes"):
        get_cfg(cfg, overrides={section: {"num_classes": value}})


def test_training_checkpoint_stores_configured_classifier_count(cfg, tmp_path):
    model = PulseCLS()
    model.train(pipeline=cfg, num_classes=5)
    checkpoint_path = Path(cfg["output_dir"]) / "best_model.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["data_cfg"]["num_classes"] == 5
    assert saved["model_state_dict"]["classifier.weight"].shape[0] == 5
    # A flat inference argument overrides the conflicting YAML value.
    configured = get_cfg(cfg, overrides={"inference": {"num_classes": 6}})
    result = model.infer(pipeline=configured, checkpoint=checkpoint_path,
        source=Path(cfg["data_cfg"]["root"]) / "0", num_classes=5,
        save_dir=tmp_path / "inference")
    assert result["num_sequences"] == 4 and model.predictor.num_classes == 5
    expanded = model.infer(pipeline=configured, checkpoint=checkpoint_path,
                           source=Path(cfg["data_cfg"]["root"]) / "0",
                           save_dir=tmp_path / "expanded_inference")
    assert expanded["num_classes"] == model.predictor.num_classes == 6
    assert model.predictor.checkpoint_num_classes == 5
    torch.testing.assert_close(model.predictor.model.classifier.weight,
                               saved["model_state_dict"]["classifier.weight"])


def test_reduced_class_count_rejects_invalid_labels_before_csv_loading(cfg):
    reduced = get_cfg(cfg, overrides={"train": {"num_classes": 2}})
    with patch("pulsecls.data.dataset.load_sample", side_effect=AssertionError("Must check labels first")):
        with pytest.raises(ValueError, match="outside num_classes=2"):
            load_split_arrays(reduced["data_cfg"], "train")


def test_invalid_cached_labels_are_checked_before_decompressing_feature_arrays(tmp_path):
    cache = tmp_path / ".pulsecls_cache"
    cache.mkdir()
    np.savez(cache / "val_seq2.npz", labels=[0, 2], data=np.zeros((2, 2, 2)), paths=["a", "b"])
    class LabelOnlyCache:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key):
            assert key == "labels", "Invalid cached labels must fail before reading large feature arrays"
            return np.array([0, 2])
    with patch("pulsecls.data.dataset.np.load", return_value=LabelOnlyCache()):
        with pytest.raises(ValueError, match="outside num_classes=2"):
            load_split_arrays({"root": str(tmp_path), "val": "val", "num_classes": 2,
                               "sequence_length": 2, "progress": False}, "val")
