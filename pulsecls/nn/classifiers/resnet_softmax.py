"""Pure ResNet1D-18 classifier for CF/PW/PRI pulse sequences."""
from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.backbones.resnet1d import ResNet1DBackbone


class ResNetSoftmaxClassifier(nn.Module):
    """ResNet encoder -> global average pooling -> dropout -> linear logits.

    CrossEntropyLoss consumes raw logits during training. The shared predictor
    applies softmax to these logits to produce class probabilities at inference.
    """

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 3))
        base_channels = int(cfg.get("base_channels", 64))
        if min(in_channels, base_channels, int(num_classes)) < 1:
            raise ValueError("in_channels, base_channels and num_classes must be positive")
        self.backbone = ResNet1DBackbone(in_channels, base_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.1)))
        self.classifier = nn.Linear(self.backbone.out_channels, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.pool(self.backbone(inputs)).flatten(1)
        logits = self.classifier(self.dropout(embedding))
        return {"logits": logits, "stream_embedding": embedding}
