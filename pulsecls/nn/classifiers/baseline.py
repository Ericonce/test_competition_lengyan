from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.backbones.resnet1d import ResNet1DBackbone
from pulsecls.nn.heads.projection import (
    StreamProjectionHead,
    TokenProjectionHead,
)


class ResNet1DClassifier(nn.Module):
    """ResNet-18 baseline with the same output contract as the full model."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        base_channels = int(cfg.get("base_channels", 64))
        embedding_dim = int(cfg.get("stream_embedding_dim", 256))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        dropout = float(cfg.get("dropout", 0.1))

        self.backbone = ResNet1DBackbone(in_channels, base_channels)
        self.stream_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, int(num_classes))
        self.stream_head = StreamProjectionHead(
            embedding_dim,
            embedding_dim,
            contrast_dim,
            dropout,
        )
        self.pulse_head = TokenProjectionHead(
            self.backbone.out_channels,
            self.backbone.out_channels,
            contrast_dim,
            dropout,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        pulse_features = self.backbone(inputs)
        stream_embedding = self.stream_proj(pulse_features)
        pulse_tokens = pulse_features.transpose(1, 2).contiguous()
        return {
            "logits": self.classifier(stream_embedding),
            "stream_embedding": stream_embedding,
            "stream_contrast": self.stream_head(stream_embedding),
            "pulse_contrast": self.pulse_head(pulse_tokens),
        }
