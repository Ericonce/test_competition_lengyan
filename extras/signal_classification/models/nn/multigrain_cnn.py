from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MultiGrainCNN(nn.Module):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        hidden_channels = list(cfg.get("hidden_channels", [64, 128, 256]))
        stream_embedding_dim = int(cfg.get("stream_embedding_dim", hidden_channels[-1]))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        dropout = float(cfg.get("dropout", 0.1))
        pooling = str(cfg.get("pooling", "avg")).lower()
        if pooling not in {"avg", "avgmax"}:
            raise ValueError(f"Unknown pooling mode {pooling!r}; expected 'avg' or 'avgmax'")
        self.pooling = pooling

        layers: list[nn.Module] = []
        current = in_channels
        for index, channels in enumerate(hidden_channels):
            kernel_size = 7 if index == 0 else 5
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(current, channels, kernel_size, padding=padding, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < len(hidden_channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2))
            current = channels
        self.backbone = nn.Sequential(*layers)
        if pooling == "avg":
            # Keep the original module layout so existing checkpoints remain loadable.
            self.stream_proj = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(current, stream_embedding_dim),
                nn.BatchNorm1d(stream_embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.stream_proj = nn.Sequential(
                nn.Linear(current * 2, stream_embedding_dim),
                nn.BatchNorm1d(stream_embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.classifier = nn.Linear(stream_embedding_dim, num_classes)
        self.stream_head = mlp(stream_embedding_dim, stream_embedding_dim, contrast_dim, dropout)
        self.pulse_head = nn.Sequential(
            nn.Linear(current, current),
            nn.ReLU(inplace=True),
            nn.Linear(current, contrast_dim),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        pulse_features = self.backbone(inputs)
        if self.pooling == "avgmax":
            average_pool = torch.mean(pulse_features, dim=-1)
            pooled = torch.cat(
                [average_pool, torch.amax(pulse_features, dim=-1)],
                dim=1,
            )
            stream_embedding = self.stream_proj(pooled)
        else:
            stream_embedding = self.stream_proj(pulse_features)
        logits = self.classifier(stream_embedding)

        pulse_tokens = pulse_features.transpose(1, 2).contiguous()
        pulse_embedding = self.pulse_head(pulse_tokens)
        stream_contrast = self.stream_head(stream_embedding)

        return {
            "logits": logits,
            "stream_embedding": stream_embedding,
            "stream_contrast": F.normalize(stream_contrast, dim=-1),
            "pulse_contrast": F.normalize(pulse_embedding, dim=-1),
        }
