"""Compact ModernTCN classifier adapted from the official ICLR 2024 code.

Reference implementation: https://github.com/luodhhh/ModernTCN
The input and output interfaces are kept compatible with this project's shared
window classifier pipeline.
"""

from torch import Tensor, nn


class LargeKernelBlock(nn.Module):
    def __init__(self, nvars: int, d_model: int, large_kernel: int, small_kernel: int,
                 ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        channels = nvars * d_model
        self.nvars = nvars
        self.d_model = d_model
        self.large = nn.Sequential(
            nn.Conv1d(channels, channels, large_kernel, padding=large_kernel // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.small = nn.Sequential(
            nn.Conv1d(channels, channels, small_kernel, padding=small_kernel // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        hidden = channels * ffn_ratio
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, groups=nvars),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, 1, groups=nvars),
            nn.Dropout(dropout),
        )
        self.variable_mixer = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, 1, groups=d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.large(x) + self.small(x)
        x = self.channel_mixer(x)
        batch, _, length = x.shape
        x = x.reshape(batch, self.nvars, self.d_model, length)
        x = x.permute(0, 2, 1, 3).reshape(batch, -1, length)
        x = self.variable_mixer(x)
        x = x.reshape(batch, self.d_model, self.nvars, length)
        x = x.permute(0, 2, 1, 3).reshape(batch, -1, length)
        return residual + x


class ModernTCN1D(nn.Module):
    """Large-kernel patch classifier following the ModernTCN block design."""

    def __init__(self, in_channels: int, num_classes: int, d_model: int = 32,
                 depth: int = 2, large_kernel: int = 31, small_kernel: int = 5,
                 patch_size: int = 8, patch_stride: int = 4,
                 ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, patch_size, stride=patch_stride),
            nn.BatchNorm1d(d_model),
        )
        self.blocks = nn.Sequential(*[
            LargeKernelBlock(in_channels, d_model, large_kernel, small_kernel,
                             ffn_ratio, dropout)
            for _ in range(depth)
        ])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels * d_model, num_classes)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, nvars, length = x.shape
        if nvars != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {nvars}")
        pad = (self.patch_stride - (length - self.patch_size) % self.patch_stride) % self.patch_stride
        if pad:
            x = nn.functional.pad(x, (0, pad), mode="replicate")
        x = self.stem(x.reshape(batch * nvars, 1, -1))
        tokens = self.blocks(x.reshape(batch, nvars * self.d_model, x.shape[-1]))
        embedding = self.dropout(self.activation(tokens)).mean(dim=-1)
        return self.classifier(embedding), embedding, tokens.transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _ = self.encode(x)
        return logits

