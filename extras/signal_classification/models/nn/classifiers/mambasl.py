from __future__ import annotations

import torch
from torch import nn

from ..heads.projection import StreamProjectionHead, TokenProjectionHead
from ..modules.mamba import build_mamba_block


class MambaSLClassifier(nn.Module):
    """Single-layer Mamba classifier for PDW sequences.

    The trainer contract is ``[B, C, L]`` in and a dict of logits plus optional
    contrast embeddings out. Internally the Mamba block still sees ``[B, L, C]``.
    """

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        d_model = int(cfg.get("d_model", 128))
        d_state = int(cfg.get("d_state", 16))
        d_conv = int(cfg.get("d_conv", 4))
        expand = int(cfg.get("expand", 2))
        dropout = float(cfg.get("dropout", 0.1))
        embedding_dim = int(cfg.get("stream_embedding_dim", d_model))
        contrast_dim = int(cfg.get("contrast_dim", 128))

        self.input_proj = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.block = build_mamba_block(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend=str(cfg.get("mamba_backend", "auto")),
        )
        self.stream_proj = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, int(num_classes))
        self.stream_head = StreamProjectionHead(
            embedding_dim,
            embedding_dim,
            contrast_dim,
            dropout,
        )
        self.pulse_head = TokenProjectionHead(
            d_model,
            d_model,
            contrast_dim,
            dropout,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # Trainer feeds [B, C, L]; Mamba scans over length.
        sequence = inputs.transpose(1, 2).contiguous()
        tokens = self.norm(self.input_proj(sequence))
        tokens = self.block(tokens)
        stream_embedding = self.stream_proj(tokens.mean(dim=1))
        return {
            "logits": self.classifier(stream_embedding),
            "stream_embedding": stream_embedding,
            "stream_contrast": self.stream_head(stream_embedding),
            "pulse_contrast": self.pulse_head(tokens),
        }
