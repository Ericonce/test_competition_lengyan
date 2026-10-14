"""Deeper 1D ResNet classifiers adapted from the torchvision ResNet design."""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv1d(in_channels, channels, 1, stride=stride, bias=False), nn.BatchNorm1d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv1d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        # As in torchvision ResNet v1.5, spatial stride is placed in the 3-tap convolution.
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)


class DeepResNet1D(nn.Module):
    def __init__(self, cfg: dict, num_classes: int, *, bottleneck: bool) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 3))
        width = int(cfg.get("base_channels", 32))
        block = Bottleneck1D if bottleneck else BasicBlock1D
        counts = (3, 4, 6, 3)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1),
        )
        current = width
        stages = []
        for index, (channels, count) in enumerate(zip((width, width * 2, width * 4, width * 8), counts)):
            blocks = [block(current, channels, 1 if index == 0 else 2)]
            current = channels * block.expansion
            blocks.extend(block(current, channels) for _ in range(count - 1))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.1)))
        self.classifier = nn.Linear(current, int(num_classes))
        self.out_channels = current

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stages(self.stem(inputs))
        embedding = self.pool(x).flatten(1)
        return {"logits": self.classifier(self.dropout(embedding)), "stream_embedding": embedding}


class ResNet34Classifier(DeepResNet1D):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__(cfg, num_classes, bottleneck=False)


class ResNet50Classifier(DeepResNet1D):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__(cfg, num_classes, bottleneck=True)
