import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pulsecls import PulseCLS
from pulsecls.cfg import get_cfg
from pulsecls.data.dataset import PulseArrayDataset, feature_stats
from pulsecls.data.inference import InferenceCSVDataset
from pulsecls.utils.normalization import FeatureNormalizer, METHODS, normalization_config


def values():
    return np.array([[10., 2., 100.], [20., 4., 200.], [30., 6., 300.], [40., 8., 400.]], np.float32)[None]


@pytest.mark.parametrize("method", METHODS)
def test_formulas_and_numpy_torch_agreement(method):
    raw = values()
    normalizer = FeatureNormalizer.fit(raw, method)
    flat = raw[0].astype(np.float64)
    if method == "zscore":
        expected = (flat - flat.mean(axis=0)) / flat.std(axis=0)
    elif method == "minmax":
        expected = (flat - flat.min(axis=0)) / np.ptp(flat, axis=0)
    elif method == "maxabs":
        expected = flat / np.abs(flat).max(axis=0)
    elif method == "robust":
        expected = (flat - np.median(flat, axis=0)) / (np.percentile(flat, 75, axis=0) - np.percentile(flat, 25, axis=0))
    elif method == "log_zscore":
        logged = np.log1p(flat)
        expected = (logged - logged.mean(axis=0)) / logged.std(axis=0)
    else:
        expected = flat
    actual = normalizer.transform_numpy(raw)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=2e-6)
    tensor = torch.from_numpy(raw.transpose(0, 2, 1).copy()).requires_grad_()
    transformed = normalizer(tensor)
    np.testing.assert_allclose(transformed.detach().numpy().transpose(0, 2, 1), actual, rtol=1e-5, atol=2e-6)
    transformed.square().mean().backward()
    assert torch.isfinite(tensor.grad).all()
    restored = FeatureNormalizer(json.loads(json.dumps(normalizer.export_state())))
    np.testing.assert_array_equal(restored.transform_numpy(raw), actual)


@pytest.mark.parametrize("method", METHODS)
def test_zero_padding_constant_features_and_training_only_fit(method):
    raw = np.array([[[10, 2, 5], [20, 4, 5], [30, 6, 5], [0, 0, 0]]], np.float32)
    normalizer = FeatureNormalizer.fit(raw, method, ignore_zero_padding=True)
    np.testing.assert_array_equal(normalizer.mean, [20, 4, 5])
    actual = normalizer.transform_numpy(raw, zero_padding=True)
    assert np.isfinite(actual).all() and (actual[:, -1] == 0).all()
    tensor = normalizer(torch.from_numpy(raw.transpose(0, 2, 1)), zero_padding=True)
    np.testing.assert_allclose(tensor.numpy().transpose(0, 2, 1), actual, rtol=1e-5, atol=2e-6)
    state = normalizer.export_state()
    normalizer.transform_numpy(raw + 100000)
    assert normalizer.export_state() == state


def test_minmax_signed_range_and_explicit_clipping():
    raw = values()
    normalizer = FeatureNormalizer.fit(raw, {"method": "minmax", "feature_range": [-1, 1]})
    actual = normalizer.transform_numpy(raw)
    np.testing.assert_array_equal(actual[0, 0], [-1, -1, -1])
    np.testing.assert_array_equal(actual[0, -1], [1, 1, 1])
    assert (normalizer.transform_numpy(raw * 3) > 1).any()
    clipped = FeatureNormalizer.fit(raw, {"method": "minmax", "feature_range": [-1, 1], "clip": True})
    for future in (raw * 3, -raw):
        result = clipped.transform_numpy(future)
        assert result.min() >= -1 and result.max() <= 1
        np.testing.assert_allclose(clipped(torch.from_numpy(future.transpose(0, 2, 1))).numpy().transpose(0, 2, 1), result)


def test_signed_log_accepts_negative_values_and_none_preserves_input():
    raw = values() - [25, 5, 250]
    normalizer = FeatureNormalizer.fit(raw, "log_zscore")
    assert np.isfinite(normalizer.transform_numpy(raw)).all()
    assert np.isfinite(normalizer(torch.from_numpy(raw.transpose(0, 2, 1).copy())).numpy()).all()
    identity = FeatureNormalizer.fit(raw, {"method": "none", "eps": 10})
    np.testing.assert_array_equal(identity.transform_numpy(raw), raw.astype(np.float32))


def test_legacy_checkpoint_and_default_statistics_are_compatible():
    raw = values()
    mean, std = feature_stats(raw)
    normalizer = FeatureNormalizer.from_checkpoint({"mean": mean, "std": std}, 3)
    np.testing.assert_array_equal(normalizer.transform_numpy(raw), (raw - mean) / std)
    fitted = FeatureNormalizer.fit(raw)
    np.testing.assert_array_equal(fitted.mean, mean)
    np.testing.assert_array_equal(fitted.std, std)


@pytest.mark.parametrize("options", ["typo", {"method": "minmax", "feature_range": [1, 0]},
    {"method": "robust", "quantile_range": [-1, 75]}, {"eps": 0}, {"eps": float("nan")},
    {"unexpected": 1}, {"clip": True}, {"clip": "yes"}])
def test_invalid_options_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        normalization_config(options)


def test_config_override_paths_and_eager_lazy_preprocessing(tmp_path):
    cfg = {"task": "classify", "output_dir": str(tmp_path), "train": {"device": "cpu", "normalization": "robust"},
           "model_cfg": {"type": "cnn1d"}, "data_cfg": {"num_classes": 2, "features": ["CF_MHz", "PW_us", "PRI_us"]}}
    assert get_cfg(cfg)["train"]["normalization"]["method"] == "robust"
    assert get_cfg(cfg, {"normalization": "minmax"})["train"]["normalization"]["method"] == "minmax"
    assert get_cfg(cfg, {"train.normalization.method": "maxabs"})["train"]["normalization"]["method"] == "maxabs"
    sample = tmp_path / "sample.csv"
    sample.write_text("CF_MHz,PW_us,PRI_us\n10,2,100\n20,4,200\n30,6,300\n40,8,400\n")
    raw = values()
    data_cfg = {"features": ["CF_MHz", "PW_us", "PRI_us"], "sequence_length": 4}
    for method in METHODS:
        normalizer = FeatureNormalizer.fit(raw, method)
        eager = PulseArrayDataset(raw, np.array([0]), normalizer.mean, normalizer.std, normalizer=normalizer)
        lazy = InferenceCSVDataset([sample], data_cfg, normalizer.mean, normalizer.std, normalizer=normalizer)
        torch.testing.assert_close(eager[0][0], lazy[0][0], rtol=0, atol=0)


def test_global_normalization_defaults_are_inherited(monkeypatch, tmp_path):
    from pulsecls.cfg import TRAIN_DEFAULTS
    monkeypatch.setitem(TRAIN_DEFAULTS, "normalization",
                        {"method": "minmax", "feature_range": [-1, 1], "eps": 1e-4})
    cfg = {"task": "classify", "output_dir": str(tmp_path), "train": {"device": "cpu"},
           "model_cfg": {"type": "cnn1d"}, "data_cfg": {"num_classes": 2}}
    actual = get_cfg(cfg)["train"]["normalization"]
    assert actual["method"] == "minmax" and actual["feature_range"] == [-1, 1]
    explicit = get_cfg(cfg, {"normalization": "maxabs"})["train"]["normalization"]
    assert explicit["method"] == "maxabs" and explicit["eps"] == 1e-4


def test_explicit_normalization_overrides_invalid_base_before_validation(tmp_path):
    cfg = {"task": "classify", "output_dir": str(tmp_path),
           "train": {"device": "cpu", "normalization": "typo"},
           "model_cfg": {"type": "cnn1d"}, "data_cfg": {"num_classes": 2}}
    assert get_cfg(cfg, {"normalization": "minmax"})["train"]["normalization"]["method"] == "minmax"
    assert get_cfg(cfg, {"train.normalization.method": "maxabs"})["train"]["normalization"]["method"] == "maxabs"


@pytest.mark.parametrize("method", ["zscore", "minmax", "log_zscore"])
def test_training_inference_and_resume_restore_fitted_normalization(tmp_path, method):
    raw = np.random.default_rng(12).uniform(1, 10, (8, 32, 3)).astype(np.float32)
    labels = np.array([0, 1] * 4)
    cfg = {"task": "classify", "output_dir": str(tmp_path / "first"),
           "data_cfg": {"features": ["CF_MHz", "PW_us", "PRI_us"], "sequence_length": 32, "num_classes": 2, "progress": False},
           "model_cfg": {"type": "cnn1d", "in_channels": 3, "hidden_channels": [4, 8], "stream_embedding_dim": 8,
                         "contrast_dim": 4, "dropout": 0.},
           "train": {"device": "cpu", "epochs": 1, "batch_size": 4, "use_amp": False,
                     "visualize": False, "normalization": method},
           "augment": {"enabled": False}, "loss": {"ce": {"weight": 1.}}}

    def samples(data_cfg, split, **kwargs):
        return (raw if split == "train" else raw + 1000), labels, ["sample"] * len(raw)

    with patch("pulsecls.engine.trainer.load_split_arrays", side_effect=samples):
        api = PulseCLS()
        api.train(pipeline=cfg)
        path = tmp_path / "first/last_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = FeatureNormalizer.fit(raw, method).export_state()
        assert checkpoint["normalization"] == expected
        assert json.loads((tmp_path / "first/normalization.json").read_text()) == expected
        if method == "zscore":
            checkpoint.pop("normalization")
            torch.save(checkpoint, path)  # Reproduce the format of existing legacy weights.
        with patch("pulsecls.engine.predictor.load_split_arrays", return_value=(raw, labels, ["sample"] * len(raw))):
            api.predict(pipeline=cfg, checkpoint=path, split="val", inference={"progress": False})
            assert api.predictor.normalizer.export_state() == expected
        # A different requested method cannot alter the meaning of saved weights.
        api.train(pipeline=cfg, output_dir=tmp_path / "resumed", resume=True,
                  resume_checkpoint=path, normalization="none")
        continued = torch.load(tmp_path / "resumed/last_model.pt", map_location="cpu", weights_only=False)
        assert continued["epoch"] == 2 and continued["normalization"] == expected
