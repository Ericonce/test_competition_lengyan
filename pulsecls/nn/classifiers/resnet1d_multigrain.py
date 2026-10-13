from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.backbones.resnet1d import ResNet1DBackbone
from pulsecls.nn.heads.adversarial import gradient_reversal
from pulsecls.nn.heads.projection import (
    MLPProjection,
    StreamProjectionHead,
    TokenProjectionHead,
)
from pulsecls.nn.hypothesis.conditioned_branch import ConditionedIQBranch


class ModularResNet1DClassifier(nn.Module):
    """ResNet backbone with optional hypothesis-IQ and two contrast levels.

    Raw path:
        ``[B, 3, T] -> backbone -> [B, 512, T/32] -> [B, 256]``

    Hypothesis path:
        ``[B, 3, T] -> K PDWs -> [B*K, 2, L_iq] -> IQ encoder
        -> [B*K, 64] -> mean -> [B, 256]``.
    """

    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        in_channels = int(cfg.get("in_channels", 2))
        base_channels = int(cfg.get("base_channels", 64))
        embedding_dim = int(cfg.get("stream_embedding_dim", 256))
        contrast_dim = int(cfg.get("contrast_dim", 128))
        dropout = float(cfg.get("dropout", 0.1))

        self.backbone = ResNet1DBackbone(in_channels, base_channels)
        self.stream_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
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
            self.backbone.out_channels,
            self.backbone.out_channels,
            contrast_dim,
            dropout,
        )

        self.hypothesis_for_classification = bool(
            cfg.get("hypothesis_for_classification", False)
        )
        self.contrast_hypothesis_randomize = bool(
            cfg.get("contrast_hypothesis_randomize", True)
        )
        self.classification_hypothesis_randomize = bool(
            cfg.get("classification_hypothesis_randomize", False)
        )
        self.conditioned_iq = ConditionedIQBranch(
            cfg,
            stream_dim=embedding_dim,
            contrast_dim=contrast_dim,
            dropout=dropout,
        )
        self.raw_pulse_head = MLPProjection(
            self.backbone.out_channels,
            self.backbone.out_channels,
            contrast_dim,
            dropout,
        )

        self.fuse_hypothesis_features = self.hypothesis_for_classification
        self.hypothesis_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.adversarial_alignment = bool(
            cfg.get("adversarial_alignment", False)
        )
        self.grl_lambda = float(cfg.get("grl_lambda", 0.1))
        if self.adversarial_alignment:
            hidden = int(cfg.get("adversarial_hidden_dim", max(contrast_dim // 2, 32)))
            self.modality_discriminator = nn.Sequential(
                nn.Linear(contrast_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        else:
            self.modality_discriminator = None

    def _pair_contrastive_views(
        self,
        inputs: torch.Tensor,
        pulse_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.conditioned_iq.pulses_per_sequence <= 0:
            return {}
        raw_features, iq_features = self.conditioned_iq.encode_paired_pulses(
            inputs,
            pulse_features,
            randomize=self.contrast_hypothesis_randomize,
        )
        raw_embedding = self.raw_pulse_head(raw_features)
        iq_embedding = iq_features
        outputs: dict[str, torch.Tensor] = {}
        outputs["pulse_pair_contrast_raw"] = raw_embedding
        outputs["pulse_pair_contrast_iq"] = iq_embedding
        if self.adversarial_alignment and self.modality_discriminator is not None:
            embeddings = torch.cat([raw_embedding, iq_embedding], dim=0)
            embeddings = gradient_reversal(embeddings, self.grl_lambda)
            outputs["modality_domain_logits"] = self.modality_discriminator(embeddings)
            outputs["modality_domain_labels"] = torch.cat(
                [
                    torch.zeros(raw_embedding.size(0), device=inputs.device, dtype=torch.long),
                    torch.ones(iq_embedding.size(0), device=inputs.device, dtype=torch.long),
                ],
                dim=0,
            )
        return outputs

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        pulse_features = self.backbone(inputs)
        raw_stream_embedding = self.stream_proj(pulse_features)
        stream_embedding = raw_stream_embedding

        if self.fuse_hypothesis_features:
            hypothesis_embedding = self.conditioned_iq.encode_stream(
                inputs,
                randomize=(
                    self.training and self.classification_hypothesis_randomize
                ),
            )
            stream_embedding = self.hypothesis_fusion(
                torch.cat([raw_stream_embedding, hypothesis_embedding], dim=-1)
            )

        pulse_tokens = pulse_features.transpose(1, 2).contiguous()
        outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {
            "logits": self.classifier(stream_embedding),
            "stream_embedding": stream_embedding,
            "stream_contrast": self.stream_head(stream_embedding),
            "pulse_contrast": self.pulse_head(pulse_tokens),
        }

        if self.training:
            pair = self._pair_contrastive_views(inputs, pulse_features)
            if "pulse_pair_contrast_raw" in pair:
                outputs["pulse_pair_contrast"] = (
                    pair["pulse_pair_contrast_raw"],
                    pair["pulse_pair_contrast_iq"],
                )
            outputs.update(pair)
        return outputs
