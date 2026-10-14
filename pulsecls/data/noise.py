"""Gaussian perturbations with per-sequence, per-channel feature SNR."""
from __future__ import annotations

import math
from numbers import Real

import torch


def validate_snr_db(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("snr_db must be a finite number in dB")
    try:
        ratio = 10.0 ** (-float(value) / 10.0)
    except OverflowError as exc:
        raise ValueError("snr_db is outside the supported numeric range") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("snr_db is outside the supported numeric range")
    return float(value)


def gaussian_noise(signal, snr_db, *, mask=None, dimensions=(-1,)):
    """Match measured perturbation energy to signal energy over the selected region.

    Leading dimensions remain independent (normally batch and channel). Complex
    input uses circular complex Gaussian draws. Zero-energy regions get no noise.
    """
    snr_db = validate_snr_db(snr_db)
    values = signal.to(torch.complex64 if signal.is_complex() else torch.float32)
    if values.is_complex():
        noise = torch.complex(torch.randn_like(values.real), torch.randn_like(values.real))
    else:
        noise = torch.randn_like(values)
    if mask is not None:
        values = values.masked_fill(~mask, 0)
        noise = noise.masked_fill(~mask, 0)
    signal_energy = values.abs().square().sum(dim=dimensions, keepdim=True)
    noise_energy = noise.abs().square().sum(dim=dimensions, keepdim=True)
    gain = (signal_energy / noise_energy.clamp_min(torch.finfo(torch.float32).tiny)).sqrt()
    gain = gain * (10.0 ** (-snr_db / 20.0))
    return (noise * gain).to(signal.dtype)
