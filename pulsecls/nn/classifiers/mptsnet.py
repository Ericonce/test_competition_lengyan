from __future__ import annotations

from torch import nn

from pulsecls.nn.classifiers.outputs import ContrastHeads
from pulsecls.nn.modules.mptsnet import MPTSNet1D


class MPTSNetClassifier(nn.Module):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        d_model = int(cfg.get("d_model", 64))
        dropout = float(cfg.get("dropout", 0.1))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        self.core = MPTSNet1D(
            in_channels=in_channels,
            num_classes=int(num_classes),
            d_model=d_model,
            depth=int(cfg.get("depth", 2)),
            kernel_sizes=tuple(cfg.get("kernel_sizes", (3, 7, 15, 31))),
            periods=tuple(cfg.get("periods", (2, 4, 8, 16))),
            num_heads=int(cfg.get("num_heads", 4)),
            token_count=int(cfg.get("token_count", 32)),
            dropout=dropout,
            norm=str(cfg.get("norm", "batch")),
        )
        stream_dim = d_model * 3 + in_channels * 6
        self.heads = ContrastHeads(stream_dim, d_model, contrast_dim, dropout)

    def forward(self, inputs):
        logits, embedding, tokens = self.core.encode(inputs)
        return self.heads.pack(logits, embedding, tokens)
