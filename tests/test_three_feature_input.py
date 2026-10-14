import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from pulsecls.cfg import get_cfg
from pulsecls.data.augment import PulseAugment
from pulsecls.data.dataset import load_sample, load_split_arrays, PulseArrayDataset, feature_stats
from pulsecls.nn.tasks import build_model
from pulsecls.signals.generator import HypothesisSignalGenerator

ROOT = Path(__file__).resolve().parents[1]


def test_column_order_and_large_toa_precision(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("label,脉冲到达时间(ms),脉宽(us),载频(MHz)\n0,1000000000.125,2,3\n0,1000000000.375,4,5\n", encoding="utf-8-sig")
    a = load_sample(p, 2, ("载频(MHz)", "脉宽(us)", "脉冲到达时间(ms)"))
    np.testing.assert_array_equal(a, [[3, 2, 0], [5, 4, .25]])
    mean, std = feature_stats(a[None])
    dataset = PulseArrayDataset(a[None], np.array([0]), mean, std)
    assert dataset[0][0].shape == (3, 2)


def test_missing_actual_pri_fails(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("载频(MHz),脉宽(us)\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing feature"):
        load_sample(p, 2)


def test_augment_treats_pri_as_a_value_channel():
    x = torch.ones(2, 3, 16)
    augment = PulseAugment(noise_snr_db=20, scale_std=0, pulse_mask_prob=0,
                           feature_mask_prob=0)
    y = augment(x)
    assert not torch.allclose(y[:, 2], x[:, 2])


def test_pri_affects_hypothesis_and_has_gradient():
    conditions = torch.tensor([[0., 0., -.5], [0., 0., .5]], requires_grad=True)
    generator = HypothesisSignalGenerator(modulation_type="lfm")
    signal = generator(conditions)
    assert signal.shape == (2, 2, 64)  # I/Q remains two components.
    assert not torch.allclose(signal[0], signal[1])
    signal[:, 0].sum().backward()
    assert conditions.grad[:, 2].abs().sum() > 0


@pytest.mark.parametrize("name", [
    "cnn_softmax", "multigrain_infonce", "pulse_hypothesis_fused", "modular_multigrain_iq",
    "atenet", "invconv1d", "mambasl", "modern_tcn", "mptsnet1d", "resnet_softmax",
])
def test_all_methods_accept_three_features_and_backpropagate(name):
    cfg = get_cfg(ROOT / f"configs/pipeline/{name}_synthetic.yaml")
    assert len(cfg["data_cfg"]["features"]) == 3
    assert cfg["model_cfg"]["in_channels"] == 3
    model = build_model(cfg["model_cfg"], cfg["data_cfg"])
    model.train()
    x = torch.randn(2, 3, cfg["data_cfg"]["sequence_length"])
    out = model(x)
    assert out["logits"].shape == (2, cfg["data_cfg"]["num_classes"])
    loss = torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if "pulse_pair_contrast" in out:
        assert out["pulse_pair_contrast"][0].shape == out["pulse_pair_contrast"][1].shape


def test_cache_feature_signature_prevents_two_channel_reuse(tmp_path):
    root = tmp_path / "data"
    sample = root / "train/0/sample_000001.csv"
    sample.parent.mkdir(parents=True)
    sample.write_text("载频(MHz),脉宽(us),脉冲到达时间\n1,2,3\n4,5,8\n", encoding="utf-8")
    cache = root / ".pulsecls_cache"
    cache.mkdir()
    np.savez(cache / "train_seq2.npz", data=np.zeros((1, 2, 2)), labels=[0], paths=["old"])
    cfg = {"root": str(root), "train": "train", "sequence_length": 2,
           "features": ["载频(MHz)", "脉宽(us)", "脉冲到达时间"]}
    data, _, _ = load_split_arrays(cfg, "train")
    assert data.shape == (1, 2, 3)
    np.testing.assert_array_equal(data[0, :, 2], [0, 5])
    assert list(cache.glob("train_seq2_c3_*.npz"))
