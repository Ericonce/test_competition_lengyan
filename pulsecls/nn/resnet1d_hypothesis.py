from __future__ import annotations

from pulsecls.nn.classifiers.resnet1d_multigrain import (
    ModularResNet1DClassifier,
)
from pulsecls.nn.hypothesis.iq_encoder import IQEncoder


class PulseIQEncoder1D(IQEncoder):
    """Backward-compatible name for the modular I/Q pulse encoder."""


class ResNet1DHypothesisClassifier(ModularResNet1DClassifier):
    """Backward-compatible name for the modular ResNet hypothesis model."""


__all__ = ["PulseIQEncoder1D", "ResNet1DHypothesisClassifier"]
