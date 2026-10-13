from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.heads.projection import StreamProjectionHead, TokenProjectionHead


class ContrastHeads(nn.Module):
    """Attach stream/pulse contrast heads to a logits-only sequence backbone."""

    def __init__(
        self,
        stream_dim: int,
        token_dim: int,
        contrast_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stream_head = StreamProjectionHead(
            stream_dim,
            stream_dim,
            contrast_dim,
            dropout,
        )
        self.pulse_head = TokenProjectionHead(
            token_dim,
            token_dim,
            contrast_dim,
            dropout,
        )

    def pack(
        self,
        logits: torch.Tensor,
        embedding: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "logits": logits,
            "stream_embedding": embedding,
            "stream_contrast": self.stream_head(embedding),
            "pulse_contrast": self.pulse_head(tokens),
        }
