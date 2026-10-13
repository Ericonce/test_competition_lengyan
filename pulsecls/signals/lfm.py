from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters


def lfm_phase(time: torch.Tensor, params: ConditionParameters) -> torch.Tensor:
    """Linear-FM phase: 2*pi*(N_c*t + 0.5*K*t^2) + phi_0."""

    return (
        2.0
        * math.pi
        * (params.cycles * time + 0.5 * params.chirp * time.square())
        + params.phase0
    )
