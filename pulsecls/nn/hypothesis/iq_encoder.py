from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pulsecls.nn.heads.projection import MLPProjection


class IQEncoder(nn.Module):
    """Encode one generated I/Q pulse into hidden and contrastive features."""

    def __init__(
        self,
        hidden_dim: int = 64,
        contrast_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        half_dim = max(hidden_dim // 2, 1)
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(2, half_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(half_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(half_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.projection = MLPProjection(
            hidden_dim,
            hidden_dim,
            int(contrast_dim),
            dropout,
        )

    def encode_features(self, signals: torch.Tensor) -> torch.Tensor:
        return self.encoder(signals)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.project(self.encode_features(signals))
