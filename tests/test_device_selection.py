from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
