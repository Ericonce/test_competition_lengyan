"""Classic sequence classifiers for fixed-length three-channel pulse streams.

The implementations follow the published/official architecture definitions while
using native PyTorch layers and the project's ``logits`` output contract.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _output(embedding: torch.Tensor, classifier: nn.Module, dropout: nn.Module) -> dict[str, torch.Tensor]:
    return {"logits": classifier(dropout(embedding)), "stream_embedding": embedding}


class _CausalResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.gelu(self.conv1(F.pad(x, (self.left_pad, 0)))))
        y = self.drop(F.gelu(self.conv2(F.pad(y, (self.left_pad, 0)))))
        return F.gelu(y + self.projection(x))


class TCNClassifier(nn.Module):
    """Causal dilated residual TCN based on the Locuslab TCN design."""
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        channels = [int(v) for v in cfg.get("channels", [64, 64, 128, 128])]
        kernel = int(cfg.get("kernel_size", 3))
        dropout = float(cfg.get("dropout", 0.1))
        layers = []
        current = int(cfg.get("in_channels", 3))
        for level, width in enumerate(channels):
            layers.append(_CausalResidualBlock(current, width, kernel, 2 ** level, dropout))
            current = width
        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.network(inputs).mean(dim=-1)
        return _output(embedding, self.classifier, self.dropout)


class _InceptionModule(nn.Module):
    def __init__(self, in_ch: int, filters: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        branch_in = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(branch_in, filters, k, padding=k // 2, bias=False) for k in (41, 21, 11)
        ])
        self.pool_conv = nn.Conv1d(in_ch, filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(filters * 4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        pooled = self.pool_conv(F.max_pool1d(x, kernel_size=3, stride=1, padding=1))
        y = torch.cat([conv(bottleneck) for conv in self.convs] + [pooled], dim=1)
        return self.drop(F.gelu(self.bn(y)))


class _InceptionResidualGroup(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.projection(residual))


class InceptionTimeClassifier(nn.Module):
    """InceptionTime-style multi-scale convolution ensemble for classification."""
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        depth = int(cfg.get("depth", 6))
        filters = int(cfg.get("filters", 32))
        bottleneck = int(cfg.get("bottleneck", 32))
        dropout = float(cfg.get("dropout", 0.1))
        modules = []
        residuals = {}
        current = int(cfg.get("in_channels", 3))
        out_ch = filters * 4
        for index in range(depth):
            modules.append(_InceptionModule(current, filters, bottleneck, dropout))
            current = out_ch
            if (index + 1) % 3 == 0:
                residuals[str(index)] = _InceptionResidualGroup(int(cfg.get("in_channels", 3)) if index == 2 else out_ch, out_ch)
        self.modules_list = nn.ModuleList(modules)
        self.residuals = nn.ModuleDict(residuals)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_ch, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = inputs
        residual = inputs
        for index, module in enumerate(self.modules_list):
            x = module(x)
            if str(index) in self.residuals:
                x = self.residuals[str(index)](x, residual)
                residual = x
        embedding = x.mean(dim=-1)
        return _output(embedding, self.classifier, self.dropout)


class RecurrentClassifier(nn.Module):
    def __init__(self, cfg: dict, num_classes: int, *, bidirectional: bool) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        layers = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            int(cfg.get("in_channels", 3)), hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional,
        )
        width = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, (hidden, _) = self.rnn(inputs.transpose(1, 2))
        embedding = torch.cat((hidden[-2], hidden[-1]), dim=-1) if self.bidirectional else hidden[-1]
        return _output(embedding, self.classifier, self.dropout)


class LSTMClassifier(RecurrentClassifier):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__(cfg, num_classes, bidirectional=False)


class BiLSTMClassifier(RecurrentClassifier):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__(cfg, num_classes, bidirectional=True)


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier with learned temporal position embeddings."""
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        width = int(cfg.get("d_model", 128))
        heads = int(cfg.get("n_heads", 4))
        if width % heads:
            raise ValueError("transformer d_model must be divisible by n_heads")
        depth = int(cfg.get("num_layers", 3))
        dropout = float(cfg.get("dropout", 0.1))
        self.input_projection = nn.Linear(int(cfg.get("in_channels", 3)), width)
        self.position = nn.Parameter(torch.zeros(1, int(cfg.get("sequence_length", 256)), width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=int(cfg.get("dim_feedforward", width * 4)),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, int(num_classes))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_projection(inputs.transpose(1, 2))
        if x.shape[1] > self.position.shape[1]:
            raise ValueError(f"input length {x.shape[1]} exceeds configured sequence_length {self.position.shape[1]}")
        embedding = self.encoder(x + self.position[:, :x.shape[1]]).mean(dim=1)
        return _output(embedding, self.classifier, self.dropout)


class _ProbSparseSelfAttention(nn.Module):
    """Encoder-side ProbSparse attention adapted from Informer2020."""
    def __init__(self, width: int, heads: int, factor: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("informer d_model must be divisible by n_heads")
        self.heads = heads
        self.head_width = width // heads
        self.factor = max(1, factor)
        self.scale = self.head_width ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        sample_count = min(length, self.factor * max(1, math.ceil(math.log(length + 1))))
        top_count = min(length, self.factor * max(1, math.ceil(math.log(length + 1))))
        sample_index = torch.randint(length, (length, sample_count), device=x.device)
        sampled_k = k[:, :, sample_index, :]
        sample_scores = (q.unsqueeze(-2) * sampled_k).sum(dim=-1)
        sparsity = sample_scores.max(dim=-1).values - sample_scores.mean(dim=-1)
        top_index = sparsity.topk(top_count, dim=-1, sorted=False).indices
        q_reduced = q.gather(2, top_index.unsqueeze(-1).expand(-1, -1, -1, self.head_width))
        scores = torch.matmul(q_reduced, k.transpose(-2, -1)) * self.scale
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = v.mean(dim=2, keepdim=True).expand(-1, -1, length, -1).clone()
        updates = torch.matmul(weights, v)
        context.scatter_(2, top_index.unsqueeze(-1).expand(-1, -1, -1, self.head_width), updates)
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, factor: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = _ProbSparseSelfAttention(width, heads, factor, dropout)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ff(x)))


class InformerClassifier(nn.Module):
    """Informer ProbSparse encoder + classification pooling/head (not forecasting decoder)."""
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        width = int(cfg.get("d_model", 128))
        heads = int(cfg.get("n_heads", 4))
        layers = int(cfg.get("e_layers", 3))
        factor = int(cfg.get("factor", 5))
        dropout = float(cfg.get("dropout", 0.1))
        self.input_projection = nn.Linear(int(cfg.get("in_channels", 3)), width)
        self.position = nn.Parameter(torch.zeros(1, int(cfg.get("sequence_length", 256)), width))
        self.encoder = nn.ModuleList([_InformerEncoderLayer(width, heads, factor, int(cfg.get("d_ff", width * 4)), dropout) for _ in range(layers)])
        self.distill = bool(cfg.get("distil", True))
        self.distillers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 3, padding=1, padding_mode="circular", bias=False),
                          nn.BatchNorm1d(width), nn.ELU(), nn.MaxPool1d(3, stride=2, padding=1))
            for _ in range(max(0, layers - 1))
        ]) if self.distill else nn.ModuleList()
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, int(num_classes))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_projection(inputs.transpose(1, 2))
        if x.shape[1] > self.position.shape[1]:
            raise ValueError(f"input length {x.shape[1]} exceeds configured sequence_length {self.position.shape[1]}")
        x = x + self.position[:, :x.shape[1]]
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if self.distill and index < len(self.encoder) - 1:
                x = self.distillers[index](x.transpose(1, 2)).transpose(1, 2)
        embedding = self.norm(x).mean(dim=1)
        return _output(embedding, self.classifier, self.dropout)


class MLPClassifier(nn.Module):
    """Flattened fixed-length MLP baseline."""
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        self.length = int(cfg.get("sequence_length", 256))
        channels = int(cfg.get("in_channels", 3))
        hidden = [int(v) for v in cfg.get("hidden_dims", [512, 256])]
        dropout = float(cfg.get("dropout", 0.2))
        layers: list[nn.Module] = []
        current = self.length * channels
        for width in hidden:
            layers.extend((nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)))
            current = width
        self.encoder = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.shape[-1] != self.length:
            raise ValueError(f"MLP expects sequence_length={self.length}, received {inputs.shape[-1]}")
        embedding = self.encoder(inputs.flatten(1))
        return _output(embedding, self.classifier, self.dropout)
