"""Compatibility shim. Prefer pulsecls.engine.predictor."""

from pulsecls.engine.predictor import BasePredictor, Inferencer

__all__ = ["BasePredictor", "Inferencer"]
