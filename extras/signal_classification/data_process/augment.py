"""Lightweight augmentations for radar pulse-description sequences."""

from __future__ import annotations

import random

import torch


class RadarAugment:
    """Apply label-preserving perturbations to ``[channels, pulses]`` tensors.

    Available methods are ``shift``, ``gaussian_noise``, ``scale``,
    ``feature_mask``, ``pulse_mask`` and ``reverse``. All operations preserve
    sequence shape and are applied independently with probability ``prob``.
    """

    def __init__(
        self,
        methods: tuple[str, ...] | list[str] = ("shift", "gaussian_noise"),
        prob: float = 0.5,
        shift_max: int = 8,
        noise_std: float = 0.02,
        scale_range: tuple[float, float] = (0.9, 1.1),
        feature_mask_prob: float = 0.1,
        pulse_mask_prob: float = 0.05,
    ) -> None:
        self.methods = tuple(str(method).strip().lower() for method in methods if str(method).strip())
        valid = {"shift", "gaussian_noise", "noise", "scale", "feature_mask", "pulse_mask", "reverse"}
        unknown = set(self.methods) - valid
        if unknown:
            raise ValueError(f"Unknown augmentation methods: {sorted(unknown)}")
        if not 0 <= prob <= 1:
            raise ValueError("prob must be in [0, 1]")
        if shift_max < 0 or noise_std < 0:
            raise ValueError("shift_max and noise_std must be non-negative")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must contain positive ascending bounds")
        self.prob = float(prob)
        self.shift_max = int(shift_max)
        self.noise_std = float(noise_std)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.feature_mask_prob = float(feature_mask_prob)
        self.pulse_mask_prob = float(pulse_mask_prob)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        result = sample
        for method in self.methods:
            if random.random() > self.prob:
                continue
            if method == "shift":
                if self.shift_max:
                    shift = random.randint(-self.shift_max, self.shift_max)
                    result = torch.roll(result, shifts=shift, dims=-1)
            elif method in {"gaussian_noise", "noise"}:
                result = result + torch.randn_like(result) * self.noise_std
            elif method == "scale":
                factor = random.uniform(*self.scale_range)
                result = result * factor
            elif method == "feature_mask":
                mask = torch.rand(result.shape[0], device=result.device) < self.feature_mask_prob
                result = result.masked_fill(mask[:, None], 0.0)
            elif method == "pulse_mask":
                mask = torch.rand(result.shape[-1], device=result.device) < self.pulse_mask_prob
                result = result.masked_fill(mask[None, :], 0.0)
            elif method == "reverse":
                result = result.flip(-1)
        return result


def parse_augment_methods(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())