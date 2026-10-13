from __future__ import annotations

from pathlib import Path

import torch

from pulsecls.data.augment import PulseAugment
from pulsecls.utils.config import load_pipeline
from pulsecls.cfg import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_time_flip_reverses_selected_sequences() -> None:
    torch.manual_seed(0)
    augment = PulseAugment(
        enabled=True,
        noise_std=0.0,
        scale_std=0.0,
        pulse_mask_prob=0.0,
        feature_mask_prob=0.0,
        time_flip_prob=1.0,
    )
    inputs = torch.arange(24, dtype=torch.float32).view(2, 2, 6)
    flipped = augment(inputs)
    assert torch.equal(flipped, inputs.flip(-1))


def test_time_shift_keeps_shape() -> None:
    torch.manual_seed(1)
    augment = PulseAugment(
        enabled=True,
        noise_std=0.0,
        scale_std=0.0,
        pulse_mask_prob=0.0,
        feature_mask_prob=0.0,
        time_shift_ratio=0.5,
    )
    inputs = torch.randn(4, 2, 32)
    shifted = augment(inputs)
    assert shifted.shape == inputs.shape


def test_mixup_returns_partner_index() -> None:
    torch.manual_seed(2)
    augment = PulseAugment(
        enabled=True,
        noise_std=0.0,
        scale_std=0.0,
        pulse_mask_prob=0.0,
        feature_mask_prob=0.0,
        mixup_prob=1.0,
        mixup_alpha=0.2,
    )
    inputs = torch.randn(5, 2, 16)
    labels = torch.arange(5)
    mixed, state = augment.apply(inputs, labels)
    assert mixed.shape == inputs.shape
    assert state is not None
    assert state.partner_index.numel() == 5
    assert 0.0 < state.lam < 1.0


def test_mambasl_pipeline_resolves_new_augment_keys() -> None:
    pipeline = load_pipeline(ROOT / "configs/pipeline/mambasl_synthetic.yaml")
    preset = load_yaml(ROOT / "configs/augment/default.yaml")
    assert pipeline["model_cfg"]["type"] == "mambasl"
    for name in ("time_shift", "mixup", "jittering"):
        assert pipeline["augment"]["methods"][name] == preset["methods"][name]
    assert pipeline["loss"]["stream_infonce"]["weight"] == 0.0
