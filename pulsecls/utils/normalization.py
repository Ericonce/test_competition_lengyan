"""Feature-wise preprocessing fitted on training pulses and saved in checkpoints."""
from copy import deepcopy
from numbers import Real

import numpy as np
import torch
from torch import nn


METHODS = ("zscore", "minmax", "maxabs", "robust", "log_zscore", "none")
DEFAULTS = {"method": "zscore", "eps": 1e-6, "feature_range": [0., 1.],
            "quantile_range": [25., 75.], "clip": False}


def normalization_config(options=None, *, defaults=None):
    if isinstance(options, str):
        options = {"method": options}
    if options is not None and not isinstance(options, dict):
        raise TypeError("normalization must be a method name or mapping")
    base = deepcopy(DEFAULTS) if defaults is None else normalization_config(defaults)
    cfg = {**base, **(options or {})}
    if set(cfg) - set(DEFAULTS):
        raise ValueError(f"Unknown normalization options: {sorted(set(cfg) - set(DEFAULTS))}")
    if not isinstance(cfg["method"], str) or cfg["method"] not in METHODS:
        raise ValueError(f"normalization.method must be one of {METHODS}")
    eps = cfg["eps"]
    if isinstance(eps, bool) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("normalization.eps must be finite and positive")
    cfg["eps"] = float(eps)
    for key in ("feature_range", "quantile_range"):
        pair = cfg[key]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in pair
        ) or pair[0] >= pair[1]:
            raise ValueError(f"normalization.{key} must contain two increasing finite numbers")
        cfg[key] = [float(v) for v in pair]
    if not 0 <= cfg["quantile_range"][0] < cfg["quantile_range"][1] <= 100:
        raise ValueError("normalization.quantile_range must lie within [0, 100]")
    if not isinstance(cfg["clip"], bool) or (cfg["clip"] and cfg["method"] != "minmax"):
        raise ValueError("normalization.clip is a boolean supported only for minmax")
    return cfg


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


class FeatureNormalizer(nn.Module):
    """Equivalent NumPy/CPU/NPU transforms; tensors use [batch, channels, length]."""
    def __init__(self, state, channels=None):
        super().__init__()
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported checkpoint normalization version")
        if not {"config", "offset", "scale", "mean", "std"} <= state.keys():
            raise ValueError("Checkpoint normalization is missing required statistics")
        self.config = normalization_config(state.get("config"))
        arrays = {name: np.asarray(state[name], dtype=np.float32)
                  for name in ("offset", "scale", "mean", "std")}
        shape = arrays["offset"].shape
        if len(shape) != 1 or not shape[0] or (channels is not None and shape != (channels,)) or any(
            value.shape != shape or not np.isfinite(value).all() for value in arrays.values()
        ) or (arrays["scale"] <= 0).any() or (arrays["std"] <= 0).any():
            raise ValueError("Checkpoint normalization must match features and have finite positive scales")
        self._state = {"version": 1, "config": deepcopy(self.config),
                       **{name: value.tolist() for name, value in arrays.items()}}
        self.mean, self.std = arrays["mean"].copy(), arrays["std"].copy()
        self.register_buffer("offset", torch.from_numpy(arrays["offset"].copy()).view(1, -1, 1))
        self.register_buffer("scale", torch.from_numpy(arrays["scale"].copy()).view(1, -1, 1))

    @classmethod
    def fit(cls, train_data, options=None, *, ignore_zero_padding=False):
        cfg = normalization_config(options)
        raw = np.asarray(train_data, dtype=np.float32)
        if raw.ndim != 3 or not raw.shape[-1] or not np.isfinite(raw).all():
            raise ValueError("Normalization requires finite [sequences, length, channels] training data")
        valid = raw[~np.all(raw == 0, axis=-1)] if ignore_zero_padding else raw.reshape(-1, raw.shape[-1])
        if not len(valid):
            raise ValueError("No valid pulses for normalization")
        mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(valid.std(axis=0, dtype=np.float64).astype(np.float32), cfg["eps"])
        method = cfg["method"]
        if method == "zscore":
            offset, scale = mean, std
        elif method == "log_zscore":
            transformed = _signed_log(valid)
            offset = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
        elif method == "minmax":
            offset = valid.min(axis=0)
            scale = valid.max(axis=0) - offset
        elif method == "maxabs":
            offset, scale = np.zeros(raw.shape[-1], np.float32), np.abs(valid).max(axis=0)
        elif method == "robust":
            lower, upper = np.percentile(valid, cfg["quantile_range"], axis=0)
            offset, scale = np.median(valid, axis=0), upper - lower
        else:
            offset, scale = np.zeros(raw.shape[-1], np.float32), np.ones(raw.shape[-1], np.float32)
        return cls({"version": 1, "config": cfg, "offset": np.asarray(offset).tolist(),
                    "scale": (scale if method == "none" else np.maximum(scale, cfg["eps"])).tolist(),
                    "mean": mean.tolist(), "std": std.tolist()})

    @classmethod
    def from_checkpoint(cls, checkpoint, channels):
        if "normalization" in checkpoint:
            return cls(checkpoint["normalization"], channels)
        # Checkpoints created before selectable normalization always use z-score.
        mean, std = checkpoint["mean"], checkpoint["std"]
        return cls({"version": 1, "config": DEFAULTS, "offset": mean, "scale": std,
                    "mean": mean, "std": std}, channels)

    def export_state(self):
        return deepcopy(self._state)

    def transform_numpy(self, values, *, zero_padding=False):
        raw = np.asarray(values, dtype=np.float32)
        if raw.shape[-1] != len(self.mean):
            raise ValueError("Normalization feature count mismatch")
        result = _signed_log(raw) if self.config["method"] == "log_zscore" else raw
        result = (result - np.asarray(self._state["offset"], np.float32)) / np.asarray(self._state["scale"], np.float32)
        if self.config["method"] == "minmax":
            low, high = self.config["feature_range"]
            result = low + (high - low) * result
            if self.config["clip"]:
                result = np.clip(result, low, high)
        if zero_padding:
            result = np.where(np.all(raw == 0, axis=-1, keepdims=True), 0., result)
        return result.astype(np.float32)

    def forward(self, values, *, zero_padding=False):
        if values.ndim != 3 or values.shape[1] != len(self.mean):
            raise ValueError("Normalization expects [batch, channels, length]")
        raw = values.float()
        result = torch.sign(raw) * torch.log1p(torch.abs(raw)) if self.config["method"] == "log_zscore" else raw
        result = (result - self.offset) / self.scale
        if self.config["method"] == "minmax":
            low, high = self.config["feature_range"]
            result = low + (high - low) * result
            if self.config["clip"]:
                result = result.clamp(low, high)
        if zero_padding:
            result = result.masked_fill(raw.eq(0).all(dim=1, keepdim=True), 0)
        return result
