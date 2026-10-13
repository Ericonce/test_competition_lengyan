from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters


def bpsk_phase(time: torch.Tensor, params: ConditionParameters) -> torch.Tensor:
    """Binary phase-shift-keying hypothesis phase."""

    symbol_index = torch.floor(time * 8.0)
    phase_code = symbol_index.remainder(2.0) * math.pi
    return 2.0 * math.pi * (params.cycles * time) + phase_code + params.phase0
