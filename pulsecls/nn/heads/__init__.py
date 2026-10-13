"""Projection and alignment heads."""

from pulsecls.nn.heads.adversarial import gradient_reversal
from pulsecls.nn.heads.projection import (
    MLPProjection,
    StreamProjectionHead,
    TokenProjectionHead,
)

__all__ = [
    "MLPProjection",
    "StreamProjectionHead",
    "TokenProjectionHead",
    "gradient_reversal",
]
