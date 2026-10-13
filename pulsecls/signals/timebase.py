from __future__ import annotations

import torch


def normalized_time(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(length) < 2:
        raise ValueError(f"Signal length must be >= 2, got {length}")
    return torch.linspace(
        0.0,
        1.0,
        int(length),
        device=device,
        dtype=dtype,
    ).reshape(1, -1)
