from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters, validate_conditions


def map_conditions_to_parameters(
    conditions: torch.Tensor,
    *,
    randomize: bool = False,
    frequency_jitter: float = 0.08,
    width_jitter: float = 0.04,
    phase_jitter: bool = True,
) -> ConditionParameters:
    """Map normalized CF/PW/TOA values to differentiable signal parameters.

    The input is the training-normalized triplet ``[CF, PW, TOA]``. This is a
    hypothesis generator, not a physical IQ reconstruction. The mapping is
    deliberately explicit so that an experiment can audit exactly how each
    PDW controls the generated view.
    """

    validate_conditions(conditions)
    cf = conditions[:, 0:1].clamp(-6.0, 6.0)
    pw = conditions[:, 1:2].clamp(-6.0, 6.0)

    cycles = 2.0 + 14.0 * torch.sigmoid(cf)
    width = 0.25 + 0.74 * torch.sigmoid(pw)
    chirp = 2.0 + 10.0 * torch.sigmoid(0.5 * cf + 0.25 * pw)
    nonlinear = 0.5 + 3.0 * torch.sigmoid(0.25 * cf - 0.5 * pw)

    if randomize:
        cycles = cycles * (
            1.0 + torch.randn_like(cycles) * float(frequency_jitter)
        )
        width = width + torch.randn_like(width) * float(width_jitter)
        chirp = chirp * (
            1.0 + torch.randn_like(chirp) * float(frequency_jitter)
        )
        nonlinear = nonlinear * (
            1.0 + torch.randn_like(nonlinear) * float(frequency_jitter)
        )

    width = width.clamp(0.08, 0.99)
    # A bounded TOA-conditioned phase for the hypothesis view, not physical IQ recovery.
    phase0 = (2.0 * math.pi * torch.sigmoid(conditions[:, 2:3].clamp(-6.0, 6.0))
              if conditions.size(1) == 3 else torch.zeros_like(cycles))
    if randomize and phase_jitter:
        phase0 = phase0 + torch.rand_like(cycles) * (2.0 * math.pi)

    return ConditionParameters(
        cycles=cycles,
        width=width,
        chirp=chirp,
        nonlinear=nonlinear,
        phase0=phase0,
    )
