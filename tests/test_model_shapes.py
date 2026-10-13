from __future__ import annotations

import torch

from pulsecls.models.backbones.resnet1d import ResNet1DBackbone
from pulsecls.models.build import build_model


def test_standard_resnet_token_shape() -> None:
    backbone = ResNet1DBackbone(in_channels=2, base_channels=64)
    outputs = backbone(torch.randn(2, 2, 256))
    assert outputs.shape == (2, 512, 8)


def test_modular_classifier_forward_contract() -> None:
    model_cfg = {
        "type": "modular_resnet1d_hypothesis",
        "in_channels": 2,
        "base_channels": 8,
        "stream_embedding_dim": 32,
        "contrast_dim": 16,
        "dropout": 0.0,
        "hypothesis_for_classification": True,
        "hypothesis_signal_length": 32,
        "hypothesis_pulses_per_sequence": 4,
        "hypothesis_hidden_dim": 16,
        "hypothesis_modulation_type": "lfm",
        "hypothesis_signal_mode": "full",
        "classification_hypothesis_randomize": False,
        "contrast_hypothesis_randomize": False,
    }
    data_cfg = {"num_classes": 5}
    model = build_model(model_cfg, data_cfg)
    model.train()
    outputs = model(torch.randn(3, 2, 256))

    assert outputs["logits"].shape == (3, 5)
    assert outputs["stream_embedding"].shape == (3, 32)
    assert outputs["stream_contrast"].shape == (3, 16)
    assert outputs["pulse_contrast"].shape == (3, 8, 16)
    assert "pulse_pair_contrast" in outputs
    raw, iq = outputs["pulse_pair_contrast"]
    assert raw.shape == (12, 16)
    assert iq.shape == (12, 16)


def test_legacy_model_name_uses_modular_implementation() -> None:
    model = build_model(
        {
            "type": "resnet1d_hypothesis",
            "base_channels": 4,
            "stream_embedding_dim": 16,
            "contrast_dim": 8,
            "hypothesis_for_classification": False,
            "hypothesis_pulses_per_sequence": 0,
        },
        {"num_classes": 3},
    )
    outputs = model(torch.randn(2, 2, 128))
    assert outputs["logits"].shape == (2, 3)


def test_mambasl_forward_contract() -> None:
    model = build_model(
        {
            "type": "mambasl",
            "in_channels": 2,
            "d_model": 32,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "stream_embedding_dim": 16,
            "contrast_dim": 8,
            "dropout": 0.0,
            "mamba_backend": "pure",
        },
        {"num_classes": 5},
    )
    model.eval()
    outputs = model(torch.randn(3, 2, 64))
    assert outputs["logits"].shape == (3, 5)
    assert outputs["stream_embedding"].shape == (3, 16)
    assert outputs["stream_contrast"].shape == (3, 8)
    assert outputs["pulse_contrast"].shape == (3, 64, 8)


def _assert_classifier_contract(model, batch: int, classes: int, contrast_dim: int) -> None:
    model.eval()
    outputs = model(torch.randn(batch, 2, 64))
    assert outputs["logits"].shape == (batch, classes)
    assert outputs["stream_embedding"].ndim == 2
    assert outputs["stream_embedding"].shape[0] == batch
    assert outputs["stream_contrast"].shape == (batch, contrast_dim)
    assert outputs["pulse_contrast"].shape[0] == batch
    assert outputs["pulse_contrast"].shape[-1] == contrast_dim
    assert outputs["pulse_contrast"].ndim == 3


def test_atenet_forward_contract() -> None:
    model = build_model(
        {
            "type": "atenet",
            "in_channels": 2,
            "hidden": 16,
            "embed_time": 8,
            "num_heads": 2,
            "dropout": 0.0,
            "contrast_dim": 8,
        },
        {"num_classes": 5},
    )
    _assert_classifier_contract(model, batch=3, classes=5, contrast_dim=8)


def test_modern_tcn_forward_contract() -> None:
    model = build_model(
        {
            "type": "modern_tcn",
            "in_channels": 2,
            "d_model": 8,
            "depth": 1,
            "large_kernel": 7,
            "small_kernel": 3,
            "patch_size": 4,
            "patch_stride": 2,
            "dropout": 0.0,
            "contrast_dim": 8,
        },
        {"num_classes": 5},
    )
    _assert_classifier_contract(model, batch=3, classes=5, contrast_dim=8)


def test_invconv_forward_contract() -> None:
    model = build_model(
        {
            "type": "invconv1d",
            "in_channels": 2,
            "d_model": 12,
            "kernel_size": 7,
            "dropout": 0.0,
            "contrast_dim": 8,
        },
        {"num_classes": 5, "sequence_length": 64},
    )
    _assert_classifier_contract(model, batch=3, classes=5, contrast_dim=8)


def test_mptsnet_forward_contract() -> None:
    model = build_model(
        {
            "type": "mptsnet1d",
            "in_channels": 2,
            "d_model": 16,
            "depth": 1,
            "kernel_sizes": (3, 5),
            "periods": (2, 4),
            "num_heads": 2,
            "token_count": 8,
            "dropout": 0.0,
            "contrast_dim": 8,
        },
        {"num_classes": 5},
    )
    _assert_classifier_contract(model, batch=3, classes=5, contrast_dim=8)
