from __future__ import annotations

import torch
from torch import nn

from pulsecls.nn.classifiers.resnet1d_multigrain import (
    ModularResNet1DClassifier,
)
from pulsecls.nn.heads.adversarial import gradient_reversal
from pulsecls.nn.hypothesis.iq_encoder import IQEncoder
from pulsecls.signals.generator import HypothesisSignalGenerator


class ModulationHypothesisGenerator(HypothesisSignalGenerator):
    """Backward-compatible generator backed by the modular signal package."""

    def __init__(
        self,
        signal_length: int = 64,
        num_modes: int = 6,
        frequency_jitter: float = 0.08,
        width_jitter: float = 0.04,
        phase_jitter: bool = True,
        signal_mode: str = "full",
    ) -> None:
        super().__init__(
            signal_length=signal_length,
            modulation_type="random",
            modulation_types=self.DEFAULT_MODULATIONS,
            num_modes=num_modes,
            frequency_jitter=frequency_jitter,
            width_jitter=width_jitter,
            phase_jitter=phase_jitter,
            signal_mode=signal_mode,
        )


class PulseIQEncoder(IQEncoder):
    """Backward-compatible name for the modular I/Q encoder."""


class PulseConditionedContrastiveCNN(ModularResNet1DClassifier):
    """Compatibility entry point routed to the modular implementation.

    Older pipeline files used this class name.  Keeping the name allows old
    checkpoints and scripts to be inspected while ensuring there is one
    implementation of hypothesis-IQ and contrastive learning.
    """


__all__ = [
    "ModulationHypothesisGenerator",
    "PulseIQEncoder",
    "PulseConditionedContrastiveCNN",
    "gradient_reversal",
]
