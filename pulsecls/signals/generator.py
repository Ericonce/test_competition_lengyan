from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from pulsecls.signals.conditions import map_conditions_to_parameters
from pulsecls.signals.envelope import soft_rectangular_envelope
from pulsecls.signals.registry import PHASE_BUILDERS, get_phase_builder
from pulsecls.signals.timebase import normalized_time


class HypothesisSignalGenerator(nn.Module):
    """Generate configurable I/Q hypotheses from normalized CF/PW/TOA triplets.

    Output shape is ``[N, 2, signal_length]``. ``modulation_type="random"``
    samples from the configured modulation list only when ``randomize=True``.
    With ``randomize=False`` it uses the first configured type, which makes
    deterministic ablations reproducible.
    """

    DEFAULT_MODULATIONS = (
        "lfm",
        "nlfm",
        "fsk",
        "bpsk",
        "lfm_fsk",
        "lfm_bpsk",
    )

    def __init__(
        self,
        signal_length: int = 64,
        *,
        modulation_type: str = "random",
        modulation_types: Sequence[str] | None = None,
        num_modes: int = 6,
        frequency_jitter: float = 0.08,
        width_jitter: float = 0.04,
        phase_jitter: bool = True,
        signal_mode: str = "full",
        envelope_steepness: float = 48.0,
    ) -> None:
        super().__init__()
        self.signal_length = int(signal_length)
        self.modulation_type = str(modulation_type).lower()
        self.frequency_jitter = float(frequency_jitter)
        self.width_jitter = float(width_jitter)
        self.phase_jitter = bool(phase_jitter)
        self.signal_mode = str(signal_mode).lower()
        self.envelope_steepness = float(envelope_steepness)

        if self.signal_mode not in {"full", "envelope", "phase"}:
            raise ValueError(
                f"Unknown signal_mode={self.signal_mode!r}; "
                "expected 'full', 'envelope', or 'phase'."
            )
        if self.modulation_type != "random":
            get_phase_builder(self.modulation_type)

        configured = tuple(modulation_types or self.DEFAULT_MODULATIONS)
        if not configured:
            raise ValueError("modulation_types must contain at least one type")
        limit = min(max(int(num_modes), 1), len(configured))
        self.modulation_types = tuple(configured[:limit])
        for name in self.modulation_types:
            get_phase_builder(name)

        time = normalized_time(
            self.signal_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("time", time, persistent=False)

    def _select_modulations(
        self,
        batch_size: int,
        *,
        randomize: bool,
        device: torch.device,
    ) -> tuple[list[str], torch.Tensor]:
        if self.modulation_type != "random":
            return [self.modulation_type], torch.zeros(
                batch_size,
                device=device,
                dtype=torch.long,
            )

        if randomize:
            indices = torch.randint(
                0,
                len(self.modulation_types),
                (batch_size,),
                device=device,
            )
        else:
            indices = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.long,
            )
        return list(self.modulation_types), indices

    def generate(
        self,
        conditions: torch.Tensor,
        *,
        randomize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = map_conditions_to_parameters(
            conditions,
            randomize=randomize,
            frequency_jitter=self.frequency_jitter,
            width_jitter=self.width_jitter,
            phase_jitter=self.phase_jitter,
        )
        time = self.time.to(device=conditions.device, dtype=conditions.dtype)
        envelope = soft_rectangular_envelope(
            time,
            params.width,
            steepness=self.envelope_steepness,
        )

        names, mode_indices = self._select_modulations(
            conditions.size(0),
            randomize=randomize,
            device=conditions.device,
        )
        phases = torch.stack(
            [
                get_phase_builder(name)(time, params)
                for name in names
            ],
            dim=1,
        )
        row_indices = torch.arange(
            conditions.size(0),
            device=conditions.device,
        )
        phase = phases[row_indices, mode_indices]

        if self.signal_mode == "envelope":
            signal_i = envelope
            signal_q = torch.zeros_like(envelope)
        elif self.signal_mode == "phase":
            signal_i = torch.cos(phase)
            signal_q = torch.sin(phase)
        else:
            signal_i = envelope * torch.cos(phase)
            signal_q = envelope * torch.sin(phase)
        return torch.stack([signal_i, signal_q], dim=1), mode_indices

    def forward(
        self,
        conditions: torch.Tensor,
        randomize: bool = False,
    ) -> torch.Tensor:
        signals, _ = self.generate(conditions, randomize=randomize)
        return signals
