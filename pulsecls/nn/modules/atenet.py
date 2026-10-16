"""ATENet-style classifier for irregular pulse trains.

Adapted from the official NeurIPS 2025 implementation:
https://github.com/shlee-labs/ATENet

The third input channel is treated as relative observation time. The remaining
channels are measurements or time-derived covariates.
"""

import math

import torch
from torch import Tensor, nn


class LearnableTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, dimension - 1)

    def forward(self, time: Tensor) -> Tensor:
        time = time.unsqueeze(-1)
        if time.device.type == "cpu":
            # Each projection has one input feature. Broadcasting avoids
            # unsupported ARM CPU matmul shapes while retaining checkpoint keys.
            dtype = time.dtype
            if torch.is_autocast_cpu_enabled() and dtype in {torch.float16, torch.bfloat16, torch.float32}:
                dtype = torch.get_autocast_cpu_dtype()
            linear = (time * self.linear.weight[:, 0] + self.linear.bias).to(dtype)
            periodic = (time * self.periodic.weight[:, 0] + self.periodic.bias).to(dtype)
        else:
            linear, periodic = self.linear(time), self.periodic(time)
        return torch.cat([linear, torch.sin(periodic)], dim=-1)


class ATENet1D(nn.Module):
    """Adaptive reference-time encoder with temporal attention and a GRU head."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 64,
                 embed_time: int = 32, num_heads: int = 4, dropout: float = 0.1,
                 time_channel: int = 2):
        super().__init__()
        if in_channels <= time_channel:
            raise ValueError("ATENet requires a relative-time channel at index 2")
        if embed_time % num_heads:
            raise ValueError("embed_time must be divisible by num_heads")
        self.in_channels = in_channels
        self.time_channel = time_channel
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.head_dim = embed_time // num_heads
        value_channels = in_channels - 1
        self.reference_time = nn.Parameter(torch.linspace(0.0, 1.0, embed_time))
        self.time_embedding = LearnableTimeEmbedding(embed_time)
        self.query = nn.Linear(embed_time, embed_time)
        self.key = nn.Linear(embed_time, embed_time)
        self.value = nn.Linear(value_channels, hidden)
        self.attention_dropout = nn.Dropout(dropout)
        # Ascend CANN 8.5.1 lacks a float32 DynamicGRUV2 kernel. A residual
        # temporal convolution keeps the ordered reference-time aggregation on
        # device while retaining ATENet's adaptive time encoder and attention.
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        time = x[:, self.time_channel, :]
        time = time - time.amin(dim=1, keepdim=True)
        time = time / time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        values = torch.cat(
            [x[:, :self.time_channel, :], x[:, self.time_channel + 1:, :]], dim=1
        ).transpose(1, 2)
        batch, length, _ = values.shape
        reference = self.reference_time.unsqueeze(0).expand(batch, -1)
        q = self.query(self.time_embedding(reference))
        k = self.key(self.time_embedding(time))
        q = q.reshape(batch, self.embed_time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1)).mean(dim=1)
        encoded_values = self.value(values)
        context = torch.matmul(weights, encoded_values)
        state = self.encoder(context.transpose(1, 2)).transpose(1, 2)
        tokens = torch.nn.functional.gelu(state + context)
        embedding = tokens.mean(dim=1)
        return self.classifier(embedding), embedding, tokens

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _ = self.encode(x)
        return logits
