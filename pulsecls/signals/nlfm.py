from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters


def nlfm_phase(time: torch.Tensor, params: ConditionParameters) -> torch.Tensor:
    """Nonlinear-FM phase using a sinusoidal phase perturbation."""

    modulation = params.nonlinear * torch.sin(
        2.0 * math.pi * (1.0 + params.width) * time
    )
    return (
        2.0 * math.pi * (params.cycles * time + modulation)
        + params.phase0
    )
