from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLPProjection(nn.Module):
    """Projection head shared by stream and pulse-level objectives."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        *,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(int(in_dim), int(hidden_dim)),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm1d(int(hidden_dim)))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(out_dim)),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(inputs), dim=-1)


class StreamProjectionHead(MLPProjection):
    pass


class TokenProjectionHead(nn.Module):
    """MLP projection for tensors shaped ``[B, T, C]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(inputs), dim=-1)
