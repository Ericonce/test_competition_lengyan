"""Compatibility shim. Prefer pulsecls.nn.tasks."""

from pulsecls.nn.tasks import (
    MODEL_REGISTRY,
    build_registered_model,
    register_model,
)

__all__ = ["MODEL_REGISTRY", "build_registered_model", "register_model"]
