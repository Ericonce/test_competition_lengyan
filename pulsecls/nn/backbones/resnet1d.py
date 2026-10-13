from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
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


class ResNet1DBackbone(nn.Module):
    """ResNet-18-style 1D encoder.

    For the standard TSRD setting ``[B, 3, 256]`` and ``base_channels=64``,
    the output is ``[B, 512, 8]``. The last dimension is the pulse-token
    axis; it is not the feature dimension.
    """

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
        self.layer1 = self._make_layer(channels, channels, 2, 1)
        self.layer2 = self._make_layer(channels, channels * 2, 2, 2)
        self.layer3 = self._make_layer(channels * 2, channels * 4, 2, 2)
        self.layer4 = self._make_layer(channels * 4, channels * 8, 2, 2)
        self.out_channels = channels * 8

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, int(blocks)):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        return self.layer4(outputs)
