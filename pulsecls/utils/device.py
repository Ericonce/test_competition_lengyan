from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def npu_available() -> bool:
    return torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()


def parse_device_spec(spec: str | int | None = None) -> torch.device:
    """Resolve a device; auto prefers CUDA, then Ascend NPU, then CPU."""
    if spec is None:
        spec = os.environ.get("PULSECLS_DEVICE", "auto")
    if isinstance(spec, int):
        if torch.cuda.is_available():
            return torch.device("cuda", spec)
        if npu_available():
            return torch.device("npu", spec)
        raise ValueError(f"Integer device {spec} requested, but no accelerator is available")

    text = str(spec).strip().lower()
    if text in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if npu_available():
            return torch.device("npu", 0)
        return torch.device("cpu")

    if text.isdigit():
        return parse_device_spec(int(text))

    device = torch.device(text)
    if device.type == "npu":
        if not npu_available():
            raise RuntimeError("NPU requested, but torch_npu is not available")
        index = 0 if device.index is None else int(device.index)
        count = torch.npu.device_count()
        if index < 0 or index >= count:
            raise ValueError(f"NPU index {index} is out of range; {count} device(s) visible")
        return torch.device("npu", index)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but no GPU is available")
        index = 0 if device.index is None else int(device.index)
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(f"CUDA index {index} is out of range; {count} device(s) visible")
        return torch.device("cuda", index)
    return torch.device("cpu")


def get_device(spec: str | int | None = None) -> torch.device:
    device = parse_device_spec(spec)
    if device.type == "npu":
        torch.npu.set_device(device.index or 0)
    elif device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    return device


def device_name(device: torch.device) -> str | None:
    index = 0 if device.index is None else device.index
    if device.type == "npu":
        return torch.npu.get_device_name(index)
    if device.type == "cuda":
        return torch.cuda.get_device_name(index)
    return None


def use_amp_for(device: torch.device, enabled: bool) -> bool:
    return bool(enabled) and device.type in {"cuda", "npu"}


def pin_memory_for(device: torch.device) -> bool:
    return device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "amp"):
        return torch.npu.amp.autocast(enabled=True)
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=True,
    )


def build_grad_scaler(device: torch.device, enabled: bool) -> Any:
    enabled = use_amp_for(device, enabled)
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "amp"):
        return torch.npu.amp.GradScaler(enabled=enabled)
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=enabled)
        except TypeError:
            return grad_scaler_cls(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
