from unittest.mock import patch

import pytest
import torch

from pulsecls.data.augment import PulseAugment, default_augment_config
from pulsecls.data.augmentation_config import METHOD_DEFAULTS, LABEL_METHODS


def inputs(device="cpu"):
    x = torch.randn(4, 3, 32, device=device)
    x[:, 2] = torch.randn(4, 32, device=device) * .1 + 1.0
    labels = torch.tensor([0, 0, 1, 1], device=device)
    return x, labels


@pytest.mark.parametrize("name", list(METHOD_DEFAULTS))
def test_each_method_shape_finite_no_mutation_for_pri(name):
    torch.manual_seed(2026)
    x, labels = inputs()
    original = x.clone()
    augment = PulseAugment(methods={name: {"enabled": True}}, order=[name])
    result, state = augment.apply(x, labels)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    if name == "mixup":
        assert state is not None


def test_composition_is_sequential_and_order_is_effective():
    x, _ = inputs()
    first = PulseAugment(methods={"jittering": {"enabled": True, "snr_db": 20}}, order=["jittering"])
    second = PulseAugment(methods={"scaling": {"enabled": True, "std": .5}}, order=["scaling"])
    combined = PulseAugment(methods={"jittering": {"enabled": True, "snr_db": 20},
                                    "scaling": {"enabled": True, "std": .5}}, order=["jittering", "scaling"])
    torch.manual_seed(77)
    expected = second(first(x))
    torch.manual_seed(77)
    actual = combined(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reversed_order = PulseAugment(methods={"jittering": {"enabled": True, "snr_db": 20},
                                         "scaling": {"enabled": True, "std": .5}}, order=["scaling", "jittering"])
    torch.manual_seed(77)
    assert not torch.allclose(reversed_order(x), actual)


def test_master_switch_method_switch_and_zero_probability():
    x, _ = inputs()
    for kwargs in ({"enabled": False, "methods": {"jittering": {"enabled": True}}},
                   {"methods": {"jittering": {"enabled": False}}},
                   {"methods": {"jittering": {"enabled": True, "prob": 0}}}):
        torch.testing.assert_close(PulseAugment(**kwargs)(x), x, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["jittering", "scaling", "magnitude_warping", "white_noise_windows"])
def test_pri_channel_participates_in_value_augmentation(name):
    torch.manual_seed(11)
    x, labels = inputs()
    result, _ = PulseAugment(methods={name: {"enabled": True}}, order=[name]).apply(x, labels)
    assert not torch.allclose(result[:, 2], x[:, 2])


def test_flip_reverses_pri_without_arrival_time_reflection():
    x, _ = inputs()
    result = PulseAugment(methods={"time_flip": {"enabled": True}}, order=["time_flip"])(x)
    torch.testing.assert_close(result, x.flip(-1))


@pytest.mark.parametrize("name", ["aaft", "iaaft"])
def test_surrogates_preserve_amplitude_distribution(name):
    x, _ = inputs()
    y = PulseAugment(methods={name: {"enabled": True}}, order=[name])(x)
    torch.testing.assert_close(y[:, :2].sort(-1).values, x[:, :2].sort(-1).values)


def test_same_class_partner_never_crosses_class_boundaries():
    x, labels = inputs()
    augment = PulseAugment(methods={"interpolation": {"enabled": True, "blend": .5}})
    partner = augment.runner.partner(x, labels)
    assert torch.equal(labels[partner], labels)
    assert (partner != torch.arange(len(labels))).all()
    singleton = torch.arange(len(labels))
    torch.testing.assert_close(augment.apply(x, singleton)[0], x)


def test_label_methods_require_labels():
    x, _ = inputs()
    with pytest.raises(ValueError, match="labels"):
        PulseAugment(methods={"interpolation": {"enabled": True}})(x)


@pytest.mark.parametrize("kwargs", [
    {"methods": {"typo": {"enabled": True}}},
    {"methods": {"jittering": {"enabled": True, "snr_db": float("nan")}}},
    {"methods": {"jittering": {"enabled": True, "prob": 2}}},
    {"methods": {"jittering": {"enabled": True}}, "order": []},
    {"methods": {}, "order": ["scaling", "scaling"]},
    {"methods": {"mixup": {"enabled": True}, "interpolation": {"enabled": True}}, "order": ["mixup", "interpolation"]},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PulseAugment(**kwargs)


def test_shared_defaults_and_contrastive_mixup_handling():
    config = default_augment_config()
    assert set(config["methods"]) == set(METHOD_DEFAULTS)
    x, labels = inputs()
    augment = PulseAugment(methods={"mixup": {"enabled": True}})
    with pytest.warns(UserWarning, match="contrastive"):
        y, state = augment.apply(x, labels, allow_mixup=False)
    assert state is None
    torch.testing.assert_close(y, x)


def test_shared_master_switch_is_not_overridden_by_pipeline(tmp_path, monkeypatch):
    from pulsecls import cfg
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    original = cfg.load_yaml

    def disabled_shared_preset(path):
        loaded = original(path)
        if Path(path).name == "default.yaml" and Path(path).parent.name == "augment":
            loaded["enabled"] = False
        return loaded

    with patch("pulsecls.cfg.load_yaml", side_effect=disabled_shared_preset):
        pipeline = cfg.get_cfg(root / "configs/pipeline/modular_multigrain_iq_synthetic.yaml")
    assert pipeline["augment"]["enabled"] is False


def test_all_methods_can_be_composed_in_one_pipeline():
    config = default_augment_config()
    for spec in config["methods"].values():
        spec["enabled"] = True
    x, labels = inputs()
    y, state = PulseAugment(**config).apply(x, labels)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert state is not None


@pytest.mark.parametrize("contrastive", [False, True])
def test_fusion_pipeline_is_used_by_trainer_with_correct_labels(contrastive):
    from pulsecls.engine.trainer import BaseTrainer
    from pulsecls.nn.tasks import build_model
    from pulsecls.losses.multigrain import MultiGranularityObjective
    model_cfg = {"type": "multigrain_cnn", "in_channels": 3,
                 "hidden_channels": [8, 16], "stream_embedding_dim": 8, "contrast_dim": 4}
    data_cfg = {"num_classes": 2, "features": ["CF", "PW", "PRI"]}
    loss_cfg = {"ce": {"weight": 1.}, "stream_infonce": {"weight": .1 if contrastive else 0, "mode": "hybrid"},
                "pulse_infonce": {"weight": 0}, "adversarial": {"weight": 0}}
    pipeline = {"model_cfg": model_cfg, "data_cfg": data_cfg, "output_dir": ".",
                "train": {"device": "cpu"}, "loss": loss_cfg}
    trainer = BaseTrainer(pipeline)
    criterion = torch.nn.CrossEntropyLoss()
    trainer.objective = MultiGranularityObjective(loss_cfg, criterion)
    model = build_model(model_cfg, data_cfg).train()
    augment = PulseAugment(methods={"interpolation": {"enabled": True}, "time_warping": {"enabled": True}})
    x, labels = inputs()
    loss, logits, _ = trainer._loss_batch(model, x, labels, criterion, augment, training=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("name", list(METHOD_DEFAULTS))
def test_all_methods_cuda(name):
    x, labels = inputs("cuda")
    y, _ = PulseAugment(methods={name: {"enabled": True}}, order=[name]).apply(x, labels)
    assert y.device == x.device and y.dtype == x.dtype and y.shape == x.shape
    assert torch.isfinite(y).all()
