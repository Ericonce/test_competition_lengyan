"""Compatibility shim. Prefer pulsecls.nn.tasks."""

from pulsecls.nn.tasks import build_model

__all__ = ["build_model"]
