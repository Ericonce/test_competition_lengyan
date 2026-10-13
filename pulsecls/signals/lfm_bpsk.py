from __future__ import annotations

import math

import torch

from pulsecls.signals.base import ConditionParameters
from pulsecls.signals.lfm import lfm_phase


def lfm_bpsk_phase(time: torch.Tensor, params: ConditionParameters) -> torch.Tensor:
    """Combined LFM and binary phase-code hypothesis phase."""

    symbol_index = torch.floor(time * 8.0)
    phase_code = symbol_index.remainder(2.0) * math.pi
    return lfm_phase(time, params) + phase_code
