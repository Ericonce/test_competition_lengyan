from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _token_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """MLP for B x T x C tokens; normalization is intentionally omitted."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ResNet1DBackbone(nn.Module):
    """A fixed ResNet-18-style 1D backbone for fair ablations."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        channels = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(
                int(in_channels),
                channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(channels, channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(channels, channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(channels * 2, channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(channels * 4, channels * 8, blocks=2, stride=2)
        self.out_channels = channels * 8

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        return self.layer4(outputs)


class ResNet1DClassifier(nn.Module):
    """ResNet-18 baseline with optional stream and pulse contrast heads."""

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        base_channels = int(cfg.get("base_channels", 64))
        embedding_dim = int(cfg.get("stream_embedding_dim", 256))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        dropout = float(cfg.get("dropout", 0.1))

        self.backbone = ResNet1DBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
        )
        self.stream_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.stream_head = _mlp(
            embedding_dim,
            embedding_dim,
            contrast_dim,
            dropout,
        )
        self.pulse_head = _token_mlp(
            self.backbone.out_channels,
            self.backbone.out_channels,
            contrast_dim,
            dropout,
        )

    def encode_raw(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone(inputs)
        stream_embedding = self.stream_proj(tokens)
        return tokens, stream_embedding

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        pulse_features, stream_embedding = self.encode_raw(inputs)
        logits = self.classifier(stream_embedding)
        stream_contrast = self.stream_head(stream_embedding)
        pulse_tokens = pulse_features.transpose(1, 2).contiguous()
        pulse_contrast = self.pulse_head(pulse_tokens)
        return {
            "logits": logits,
            "stream_embedding": stream_embedding,
            "stream_contrast": F.normalize(stream_contrast, dim=-1),
            "pulse_contrast": F.normalize(pulse_contrast, dim=-1),
        }
