from __future__ import annotations

from torch import nn

from pulsecls.nn.classifiers.outputs import ContrastHeads
from pulsecls.nn.modules.modern_tcn import ModernTCN1D


class ModernTCNClassifier(nn.Module):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        d_model = int(cfg.get("d_model", 32))
        dropout = float(cfg.get("dropout", 0.1))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        self.core = ModernTCN1D(
            in_channels=in_channels,
            num_classes=int(num_classes),
            d_model=d_model,
            depth=int(cfg.get("depth", 2)),
            large_kernel=int(cfg.get("large_kernel", 31)),
            small_kernel=int(cfg.get("small_kernel", 5)),
            patch_size=int(cfg.get("patch_size", 8)),
            patch_stride=int(cfg.get("patch_stride", 4)),
            ffn_ratio=int(cfg.get("ffn_ratio", 2)),
            dropout=dropout,
        )
        stream_dim = in_channels * d_model
        self.heads = ContrastHeads(stream_dim, stream_dim, contrast_dim, dropout)

    def forward(self, inputs):
        logits, embedding, tokens = self.core.encode(inputs)
        return self.heads.pack(logits, embedding, tokens)
