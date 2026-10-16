from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pulsecls.utils import device as devices


@pytest.mark.parametrize("spec", ["auto", "", None])
@pytest.mark.parametrize(
    "cuda,npu,expected",
    [(True, True, "cuda"), (True, False, "cuda"),
     (False, True, "npu"), (False, False, "cpu")],
)
def test_auto_accelerator_priority(monkeypatch, spec, cuda, npu, expected):
    monkeypatch.delenv("PULSECLS_DEVICE", raising=False)
    with patch.object(devices.torch.cuda, "is_available", return_value=cuda), \
         patch.object(devices, "npu_available", return_value=npu) as check_npu, \
         patch.object(devices.torch, "device", side_effect=lambda kind, index=None: SimpleNamespace(type=kind, index=index)):
        selected = devices.parse_device_spec(spec)
    assert selected.type == expected
    if cuda:
        check_npu.assert_not_called()


@pytest.mark.parametrize("spec", [1, "1"])
@pytest.mark.parametrize("cuda,expected", [(True, "cuda"), (False, "npu")])
def test_numeric_accelerator_priority(spec, cuda, expected):
    with patch.object(devices.torch.cuda, "is_available", return_value=cuda), \
         patch.object(devices, "npu_available", return_value=True), \
         patch.object(devices.torch, "device", side_effect=lambda kind, index=None: SimpleNamespace(type=kind, index=index)):
        selected = devices.parse_device_spec(spec)
    assert (selected.type, selected.index) == (expected, 1)


def test_explicit_cpu_and_environment_override(monkeypatch):
    monkeypatch.setenv("PULSECLS_DEVICE", "cpu")
    with patch.object(devices.torch.cuda, "is_available", return_value=True):
        assert devices.parse_device_spec(None).type == "cpu"
        assert devices.parse_device_spec("cpu").type == "cpu"
        assert devices.parse_device_spec("auto").type == "cuda"


def test_explicit_cuda_does_not_fall_back_to_npu():
    with patch.object(devices.torch.cuda, "is_available", return_value=False), \
         patch.object(devices, "npu_available", return_value=True):
        with pytest.raises(RuntimeError, match="CUDA requested"):
            devices.parse_device_spec("cuda:0")


@pytest.mark.parametrize("supported,expected", [(True, torch.bfloat16), (False, torch.float16)])
def test_auto_amp_uses_bf16_only_on_supported_npu(monkeypatch, supported, expected):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_bf16_supported=lambda: supported), raising=False)
    assert devices.amp_dtype_for(SimpleNamespace(type="npu"), "auto") == expected
    assert devices.amp_dtype_for(torch.device("cuda"), "auto") == torch.float16


@pytest.mark.parametrize("dtype,enabled", [("bfloat16", False), ("float16", True)])
def test_gradient_scaling_with_unified_amp(monkeypatch, dtype, enabled):
    # torch.amp.GradScaler was introduced after the competition's PyTorch 2.1.
    monkeypatch.setattr(torch, "amp", SimpleNamespace())
    with patch.object(torch.amp, "GradScaler", create=True) as scaler:
        result = devices.build_grad_scaler(torch.device("cuda"), True, dtype)
        scaler.assert_called_once_with("cuda", enabled=enabled)
        assert result is scaler.return_value


@pytest.mark.parametrize("dtype,enabled", [("bfloat16", False), ("float16", True)])
def test_gradient_scaling_with_legacy_amp(monkeypatch, dtype, enabled):
    monkeypatch.setattr(torch, "amp", SimpleNamespace())
    with patch.object(torch.cuda.amp, "GradScaler") as scaler:
        result = devices.build_grad_scaler(torch.device("cuda"), True, dtype)
        scaler.assert_called_once_with(enabled=enabled)
        assert result is scaler.return_value


def test_invalid_amp_dtype_fails_clearly():
    with pytest.raises(ValueError, match="amp_dtype"):
        devices.amp_dtype_for(torch.device("cpu"), "float64")
