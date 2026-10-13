"""InvConvNet classification model adapted from the ICML 2025 reference code.

The three branches are standard, local-offset invariant, and local-linear-trend
invariant convolutions. Inputs are [batch, channels, sequence_length].
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InvConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: int,
        padding: int,
        init_type: str = "gaussian",
    ):
        super().__init__()
        if len(out_channels) != 3 or sum(out_channels) < 1:
            raise ValueError("out_channels must contain three non-negative sizes")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.out_channels = tuple(int(value) for value in out_channels)
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(
            self._initialize_weight(in_channels, self.out_channels, kernel_size, init_type)
        )
        self.bias = nn.Parameter(torch.randn(sum(self.out_channels)))

    @staticmethod
    def _initialize_weight(in_channels, out_channels, kernel_size, init_type):
        if init_type == "gaussian":
            weight = torch.randn(sum(out_channels), in_channels, kernel_size)
        elif init_type == "cosine":
            time = torch.linspace(0, torch.pi, kernel_size).view(1, 1, -1)
            pieces = []
            for branch_size in out_channels:
                frequencies = torch.arange(1, branch_size + 1).view(-1, 1, 1)
                pieces.append(torch.cos(time * frequencies).expand(-1, in_channels, -1))
            weight = torch.cat(pieces, dim=0)
            weight = weight + 0.05 * torch.randn_like(weight)
        else:
            raise ValueError(f"Unknown InvConv initialization: {init_type}")
        return weight

    def _local_stats(self, signal: Tensor):
        ones = signal.new_ones((signal.shape[1], 1, self.kernel_size))
        mean = F.conv1d(signal, ones / self.kernel_size, padding=self.padding, groups=signal.shape[1])
        second = F.conv1d(signal.square(), ones / self.kernel_size,
                          padding=self.padding, groups=signal.shape[1])
        std = torch.sqrt(torch.clamp(second - mean.square(), min=0.0))
        length = float(self.kernel_size)
        norm = std * length**0.5

        positions = torch.arange(self.kernel_size, device=signal.device, dtype=signal.dtype)
        positions = positions.view(1, 1, -1)
        linear_mean = (self.kernel_size - 1) / 2.0
        linear_std = ((self.kernel_size**2 - 1) / 12.0) ** 0.5
        linear_kernel = positions.expand(signal.shape[1], -1, -1)
        weighted = F.conv1d(signal, linear_kernel, padding=self.padding, groups=signal.shape[1])
        coef = (weighted / length - linear_mean * mean) / (linear_std**2)
        offset = mean - coef * linear_mean
        linear_norm = torch.sqrt(torch.clamp(length * (std.square() - coef.square() * linear_std**2), min=0.0))
        return mean, norm, coef, offset, linear_norm

    def forward(self, signal: Tensor) -> Tensor:
        if signal.shape[1] != self.weight.shape[1]:
            raise ValueError("Input channels do not match InvConvNet")
        # Keep the input-channel axis, matching the reference FFT implementation.
        cross = F.conv1d(
            signal,
            self.weight.permute(1, 0, 2).reshape(signal.shape[1] * self.weight.shape[0], 1, self.kernel_size),
            padding=self.padding,
            groups=signal.shape[1],
        )
        product = cross.reshape(signal.shape[0], signal.shape[1], self.weight.shape[0], -1).permute(0, 2, 1, 3)
        standard, offset_branch, trend_branch = torch.split(product, self.out_channels, dim=1)
        weights = torch.split(self.weight, self.out_channels, dim=0)
        biases = torch.split(self.bias, self.out_channels)
        mean, norm, coef, offset, linear_norm = self._local_stats(signal)
        outputs = []
        if self.out_channels[0]:
            outputs.append(standard.sum(dim=2) + biases[0].view(1, -1, 1))
        if self.out_channels[1]:
            kernel_sum = weights[1].sum(dim=-1).view(1, self.out_channels[1], signal.shape[1], 1)
            normalized = offset_branch - kernel_sum * mean.unsqueeze(1)
            normalized = normalized / torch.where(norm.unsqueeze(1) == 0, torch.inf, norm.unsqueeze(1))
            outputs.append(normalized.sum(dim=2) + biases[1].view(1, -1, 1))
        if self.out_channels[2]:
            positions = torch.arange(self.kernel_size, device=signal.device, dtype=signal.dtype)
            weighted_sum = (weights[2] * positions.view(1, 1, -1)).sum(dim=-1)
            kernel_sum = weights[2].sum(dim=-1)
            corrected = trend_branch
            corrected = corrected - weighted_sum.view(1, self.out_channels[2], signal.shape[1], 1) * coef.unsqueeze(1)
            corrected = corrected - kernel_sum.view(1, self.out_channels[2], signal.shape[1], 1) * offset.unsqueeze(1)
            corrected = corrected / torch.where(linear_norm.unsqueeze(1) == 0, torch.inf, linear_norm.unsqueeze(1))
            outputs.append(corrected.sum(dim=2) + biases[2].view(1, -1, 1))
        return torch.cat(outputs, dim=1)


class InvConvNet1D(nn.Module):
    """Reference InvConvNet-4 style one-layer classifier."""

    def __init__(self, in_channels: int, num_classes: int, d_model: int = 64,
                 kernel_size: int = 51, dropout: float = 0.2,
                 sequence_length: int = 256):
        super().__init__()
        if d_model < 3:
            raise ValueError("d_model must be at least 3")
        kernel_size = min(kernel_size, 2 * (int(sequence_length) // 2) + 1)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        first = (d_model + 2) // 3
        second = (d_model - first) // 2
        hidden = (first, second, d_model - first - second)
        self.invconv = InvConv1d(in_channels, hidden, kernel_size, kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.relu(self.bn(self.invconv(x)))
        embedding = tokens.mean(dim=-1)
        return self.fc(self.dropout(embedding)), embedding, tokens.transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _ = self.encode(x)
        return logits
