from __future__ import annotations

from torch import nn

from .outputs import ContrastHeads
from ..modules.invconv import InvConvNet1D


class InvConvClassifier(nn.Module):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        d_model = int(cfg.get("d_model", 64))
        dropout = float(cfg.get("dropout", 0.2))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        self.core = InvConvNet1D(
            in_channels=in_channels,
            num_classes=int(num_classes),
            d_model=d_model,
            kernel_size=int(cfg.get("kernel_size", 51)),
            dropout=dropout,
            sequence_length=int(cfg.get("sequence_length", 256)),
        )
        self.heads = ContrastHeads(d_model, d_model, contrast_dim, dropout)

    def forward(self, inputs):
        logits, embedding, tokens = self.core.encode(inputs)
        return self.heads.pack(logits, embedding, tokens)
