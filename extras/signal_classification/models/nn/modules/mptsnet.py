"""Length-agnostic MPTSNet adaptation for multivariate PDW sequences.

The model keeps the core ideas of MPTSNet (AAAI 2025): multiscale local
patterns, periodic views, and global dependency modelling.  The original
fixed-length flattened classifier is replaced with attentive statistics
pooling so the same network can later accept variable-length sequences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiScaleLocalMixer(nn.Module):
    def __init__(self, channels: int, kernel_sizes: Sequence[int]):
        super().__init__()
        self.branches = nn.ModuleList(
            DepthwiseConv1d(channels, kernel_size) for kernel_size in kernel_sizes
        )
        self.project = nn.Sequential(
            nn.Conv1d(channels * len(kernel_sizes), channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class PeriodicMixer(nn.Module):
    """Mix neighbouring cycles after folding a sequence by fixed periods."""

    def __init__(self, channels: int, periods: Sequence[int]):
        super().__init__()
        self.periods = tuple(int(period) for period in periods)
        if not self.periods or any(period < 2 for period in self.periods):
            raise ValueError("periods must contain integers greater than one")
        self.mixers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            for _ in self.periods
        )
        self.period_logits = nn.Parameter(torch.zeros(len(self.periods)))
        self.project = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        outputs = []
        for period, mixer in zip(self.periods, self.mixers):
            padded_length = math.ceil(length / period) * period
            folded = F.pad(x, (0, padded_length - length))
            # [B, C, cycles, position-within-cycle]
            folded = folded.reshape(batch, channels, padded_length // period, period)
            mixed = mixer(folded) + folded
            outputs.append(mixed.reshape(batch, channels, padded_length)[..., :length])
        weights = self.period_logits.softmax(dim=0)
        result = sum(weight * output for weight, output in zip(weights, outputs))
        return self.project(result)


class GlobalTokenMixer(nn.Module):
    """Bound global attention cost using a fixed number of pooled tokens."""

    def __init__(self, channels: int, num_heads: int, token_count: int, dropout: float):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.token_count = token_count
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.project = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def _encode_unfused(self, tokens: Tensor) -> Tensor:
        """Equivalent pre-norm encoder without backend-specific fused shortcuts."""
        for layer in self.encoder.layers:
            attention = layer.self_attn
            normalized = layer.norm1(tokens).transpose(0, 1)
            attended, _ = F.multi_head_attention_forward(
                normalized, normalized, normalized,
                attention.embed_dim, attention.num_heads,
                attention.in_proj_weight, attention.in_proj_bias,
                attention.bias_k, attention.bias_v, attention.add_zero_attn,
                attention.dropout, attention.out_proj.weight, attention.out_proj.bias,
                training=attention.training, need_weights=False,
            )
            tokens = tokens + layer.dropout1(attended.transpose(0, 1))
            tokens = tokens + layer._ff_block(layer.norm2(tokens))
        if self.encoder.norm is not None:
            tokens = self.encoder.norm(tokens)
        return tokens

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        token_count = min(length, self.token_count)
        tokens = F.adaptive_avg_pool1d(x, token_count).transpose(1, 2)
        if tokens.device.type == "npu":
            tokens = self._encode_unfused(tokens)
        else:
            tokens = self.encoder(tokens)
        tokens = tokens.transpose(1, 2)
        tokens = F.interpolate(tokens, size=length, mode="linear", align_corners=False)
        return self.project(tokens)


class MPTSBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int],
        periods: Sequence[int],
        num_heads: int,
        token_count: int,
        dropout: float,
    ):
        super().__init__()
        self.local = MultiScaleLocalMixer(channels, kernel_sizes)
        self.periodic = PeriodicMixer(channels, periods)
        self.global_mixer = GlobalTokenMixer(channels, num_heads, token_count, dropout)
        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, 1),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        weights = self.fusion_logits.softmax(dim=0)
        mixed = (
            weights[0] * self.local(x)
            + weights[1] * self.periodic(x)
            + weights[2] * self.global_mixer(x)
        )
        x = self.norm1(x + mixed)
        return self.norm2(x + self.channel_mlp(x))


class AttentiveStatisticsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, max(channels // 2, 8), 1),
            nn.Tanh(),
            nn.Conv1d(max(channels // 2, 8), 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.attention(x).softmax(dim=-1)
        mean = (weights * x).sum(dim=-1)
        variance = (weights * (x - mean.unsqueeze(-1)).square()).sum(dim=-1)
        maximum = x.amax(dim=-1)
        return torch.cat([mean, variance.clamp_min(1e-8).sqrt(), maximum], dim=1)


class MPTSNet1D(nn.Module):
    """PDW-oriented, variable-length MPTSNet classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 64,
        depth: int = 2,
        kernel_sizes: Sequence[int] = (3, 7, 15, 31),
        periods: Sequence[int] = (2, 4, 8, 16),
        num_heads: int = 4,
        token_count: int = 32,
        dropout: float = 0.1,
        norm: str = "batch",
    ):
        super().__init__()
        if depth < 1 or d_model < 8:
            raise ValueError("depth and d_model must be positive")
        if any(kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("all kernel sizes must be odd")
        self.in_channels = in_channels
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                MPTSBlock(
                    d_model,
                    kernel_sizes,
                    periods,
                    num_heads,
                    token_count,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.pool = AttentiveStatisticsPool(d_model)
        # Preserve simple absolute-value and variation statistics from RF/PW.
        raw_stats = in_channels * 6
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3 + raw_stats, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        if norm == "group":
            self._replace_batch_norm(self)
        elif norm != "batch":
            raise ValueError("norm must be 'batch' or 'group'")
        self.apply(self._initialize)

    @classmethod
    def _replace_batch_norm(cls, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                groups = min(8, child.num_features)
                while child.num_features % groups:
                    groups -= 1
                setattr(module, name, nn.GroupNorm(groups, child.num_features))
            else:
                cls._replace_batch_norm(child)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _raw_statistics(x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        minimum = x.amin(dim=-1)
        maximum = x.amax(dim=-1)
        slope = x[..., -1] - x[..., 0]
        variation = (x[..., 1:] - x[..., :-1]).abs().mean(dim=-1)
        return torch.cat([mean, std, minimum, maximum, slope, variation], dim=1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError("MPTSNet1D expects [batch, channels, length]")
        if x.shape[-1] < 2:
            raise ValueError("sequence length must be at least two")
        raw = self._raw_statistics(x)
        tokens = self.blocks(self.embedding(x))
        embedding = torch.cat([self.pool(tokens), raw], dim=1)
        return self.classifier(embedding), embedding, tokens.transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _ = self.encode(x)
        return logits
