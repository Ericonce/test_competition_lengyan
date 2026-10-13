from __future__ import annotations

from collections.abc import Callable

from torch import nn

from pulsecls.losses.multigrain import MultiGranularityObjective


LOSS_REGISTRY: dict[str, Callable[..., object]] = {
    "multigrain": MultiGranularityObjective,
}


def build_objective(
    name: str,
    loss_cfg: dict,
    criterion: nn.Module,
) -> object:
    key = str(name).lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown objective {name!r}; available: {', '.join(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[key](loss_cfg, criterion)
