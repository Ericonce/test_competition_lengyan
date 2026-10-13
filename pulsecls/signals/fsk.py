from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters


def fsk_phase(time: torch.Tensor, params: ConditionParameters) -> torch.Tensor:
    """Four-state frequency-shift-keying hypothesis phase."""

    symbol_index = torch.floor(time * 8.0)
    frequency_offset = ((symbol_index.remainder(4.0)) - 1.5) * 0.35
    return (
        2.0
        * math.pi
        * ((params.cycles + frequency_offset) * time)
        + params.phase0
    )
