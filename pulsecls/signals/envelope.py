from __future__ import annotations

import torch


def soft_rectangular_envelope(
    time: torch.Tensor,
    width: torch.Tensor,
    *,
    steepness: float = 48.0,
) -> torch.Tensor:
    """Differentiable approximation of a rectangular pulse envelope."""

    return torch.sigmoid((width - time) * float(steepness))
