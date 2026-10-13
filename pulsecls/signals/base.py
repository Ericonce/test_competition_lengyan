from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ConditionParameters:
    """Continuous parameters derived from one normalized CF/PW/TOA triplet."""

    cycles: torch.Tensor
    width: torch.Tensor
    chirp: torch.Tensor
    nonlinear: torch.Tensor
    phase0: torch.Tensor


PhaseBuilder = Callable[[torch.Tensor, ConditionParameters], torch.Tensor]


def validate_conditions(conditions: torch.Tensor) -> None:
    if conditions.ndim != 2 or conditions.size(-1) not in (2, 3):
        raise ValueError(
            "conditions must have shape [N, 3] with [CF, PW, TOA] (legacy [N, 2] accepted), "
            f"got {tuple(conditions.shape)}"
        )
