from __future__ import annotations

import torch
from torch import nn

from .outputs import ContrastHeads
from ..modules.atenet import ATENet1D


class ATENetClassifier(nn.Module):
    """ATENet uses the real TOA channel; synthetic time is for legacy two-channel inputs only."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        hidden = int(cfg.get("hidden", 64))
        dropout = float(cfg.get("dropout", 0.1))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        self.time_channel = int(cfg.get("time_channel", 2))
        core_channels = max(in_channels, self.time_channel + 1)
        self.core = ATENet1D(
            in_channels=core_channels,
            num_classes=int(num_classes),
            hidden=hidden,
            embed_time=int(cfg.get("embed_time", 32)),
            num_heads=int(cfg.get("num_heads", 4)),
            dropout=dropout,
            time_channel=self.time_channel,
        )
        self.heads = ContrastHeads(hidden, hidden, contrast_dim, dropout)

    def _with_time(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.time_channel:
            return inputs
        batch, _, length = inputs.shape
        time = torch.linspace(0, 1, length, device=inputs.device, dtype=inputs.dtype)
        time = time.view(1, 1, length).expand(batch, 1, length)
        return torch.cat([inputs, time], dim=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, embedding, tokens = self.core.encode(self._with_time(inputs))
        return self.heads.pack(logits, embedding, tokens)
