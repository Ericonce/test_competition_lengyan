from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.hypothesis.iq_encoder import IQEncoder
from pulsecls.signals.generator import HypothesisSignalGenerator


class ConditionedIQBranch(nn.Module):
    """Sample PDWs, generate hypothesis IQ, and aggregate pulse features."""

    def __init__(
        self,
        cfg: dict,
        *,
        stream_dim: int,
        contrast_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pulses_per_sequence = int(
            cfg.get("hypothesis_pulses_per_sequence", 8)
        )
        self.condition_alignment = str(
            cfg.get("hypothesis_condition_alignment", "paired")
        )
        if self.condition_alignment not in {"paired", "mismatched_batch"}:
            raise ValueError(
                "hypothesis_condition_alignment must be 'paired' or "
                f"'mismatched_batch', got {self.condition_alignment!r}"
            )

        modulation_types = cfg.get("hypothesis_modulation_types")
        if modulation_types is not None and not isinstance(modulation_types, (list, tuple)):
            raise TypeError("hypothesis_modulation_types must be a list or tuple")
        self.generator = HypothesisSignalGenerator(
            signal_length=int(cfg.get("hypothesis_signal_length", 64)),
            modulation_type=str(cfg.get("hypothesis_modulation_type", "random")),
            modulation_types=modulation_types,
            num_modes=int(cfg.get("hypothesis_num_modes", 6)),
            frequency_jitter=float(cfg.get("hypothesis_frequency_jitter", 0.08)),
            width_jitter=float(cfg.get("hypothesis_width_jitter", 0.04)),
            phase_jitter=bool(cfg.get("hypothesis_phase_jitter", True)),
            signal_mode=str(cfg.get("hypothesis_signal_mode", "full")),
            envelope_steepness=float(cfg.get("hypothesis_envelope_steepness", 48.0)),
        )
        self.encoder = IQEncoder(
            hidden_dim=int(cfg.get("hypothesis_hidden_dim", 64)),
            contrast_dim=contrast_dim,
            dropout=dropout,
        )
        self.stream_projection = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, int(stream_dim)),
            nn.BatchNorm1d(int(stream_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

    def _sample_indices(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        count = min(max(self.pulses_per_sequence, 1), sequence_length)
        if self.training:
            return torch.randint(
                0,
                sequence_length,
                (batch_size, count),
                device=device,
            )
        indices = torch.linspace(
            0,
            sequence_length - 1,
            count,
            device=device,
        ).round().long()
        return indices.unsqueeze(0).expand(batch_size, -1)

    def sample_conditions(
        self,
        inputs: torch.Tensor,
        pulse_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, _channels, sequence_length = inputs.shape
        indices = self._sample_indices(
            batch_size,
            sequence_length,
            device=inputs.device,
        )
        sequence = inputs.transpose(1, 2).contiguous()
        batch_indices = torch.arange(
            batch_size,
            device=inputs.device,
        )[:, None]
        conditions = sequence[batch_indices, indices].reshape(-1, inputs.size(1))

        sampled_raw = None
        if pulse_tokens is not None:
            token_sequence = pulse_tokens.transpose(1, 2).contiguous()
            token_length = token_sequence.size(1)
            token_indices = torch.div(
                indices * token_length,
                sequence_length,
                rounding_mode="floor",
            ).clamp_max(token_length - 1)
            sampled_raw = token_sequence[
                batch_indices,
                token_indices,
            ].reshape(-1, pulse_tokens.size(1))

        if self.condition_alignment == "mismatched_batch":
            count = conditions.size(0) // batch_size
            conditions = torch.roll(
                conditions.reshape(batch_size, count, -1),
                shifts=1,
                dims=0,
            ).reshape_as(conditions)
        return conditions, sampled_raw

    def encode_stream(
        self,
        inputs: torch.Tensor,
        *,
        randomize: bool,
    ) -> torch.Tensor:
        conditions, _ = self.sample_conditions(inputs)
        signals = self.generator(conditions, randomize=randomize)
        features = self.encoder.encode_features(signals)
        batch_size = inputs.size(0)
        count = min(max(self.pulses_per_sequence, 1), inputs.size(-1))
        features = features.reshape(batch_size, count, -1).mean(dim=1)
        return self.stream_projection(features)

    def encode_paired_pulses(
        self,
        inputs: torch.Tensor,
        pulse_tokens: torch.Tensor,
        *,
        randomize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditions, sampled_raw = self.sample_conditions(inputs, pulse_tokens)
        if sampled_raw is None:
            raise RuntimeError("Pulse tokens are required for paired encoding")
        signals = self.generator(conditions, randomize=randomize)
        iq_features = self.encoder(signals)
        return sampled_raw, iq_features
