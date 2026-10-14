from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pulsecls.cfg import validate_label_smoothing
from pulsecls.data.dataset import class_weights


class PriorAdjustedCrossEntropyLoss(nn.Module):
    """Cross entropy with optional log-count adjustment."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        *,
        alpha: float = 0.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "log_priors",
            torch.log(class_counts.float().clamp_min(1.0)),
        )
        self.alpha = float(alpha)
        self.weight = weight
        self.label_smoothing = validate_label_smoothing(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = logits + self.alpha * self.log_priors.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        weight = (
            self.weight.to(logits.device, logits.dtype)
            if self.weight is not None
            else None
        )
        return F.cross_entropy(
            adjusted,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


def build_classification_loss(
    loss_cfg: dict,
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    class_weight_mode = loss_cfg.get("class_weights")
    weights = None
    if class_weight_mode in {"balanced", "sqrt_balanced", "log_balanced"}:
        weights = class_weights(
            labels,
            int(num_classes),
            mode=class_weight_mode,
        ).to(device)

    label_smoothing = validate_label_smoothing(loss_cfg.get("label_smoothing", 0.0))
    alpha = float(loss_cfg.get("prior_alpha", 0.0))
    if alpha == 0.0:
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
        )
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    return PriorAdjustedCrossEntropyLoss(
        torch.from_numpy(np.maximum(counts, 1.0)).to(device),
        alpha=alpha,
        weight=weights,
        label_smoothing=label_smoothing,
    )
