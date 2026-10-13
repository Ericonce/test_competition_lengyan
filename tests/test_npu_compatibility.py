from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from pulsecls.nn.modules.mptsnet import GlobalTokenMixer
from pulsecls.nn.modules.mamba import PureMambaBlock, build_mamba_block
from pulsecls.utils.seed import seed_everything


def test_seed_preserves_selected_npu():
    npu = SimpleNamespace(manual_seed_all=Mock())
    with patch("pulsecls.utils.seed.get_device", side_effect=AssertionError("device reset")), \
         patch.object(torch, "npu", npu, create=True), \
         patch.object(torch.cuda, "is_available", return_value=False):
        seed_everything(2026, device=SimpleNamespace(type="npu", index=1))
    npu.manual_seed_all.assert_called_once_with(2026)


def test_mamba_backend_detection_does_not_change_current_device():
    with patch("pulsecls.utils.device.get_device", side_effect=AssertionError("device reset")), \
         patch("pulsecls.utils.device.parse_device_spec", return_value=SimpleNamespace(type="npu")):
        assert isinstance(build_mamba_block(8, backend="auto"), PureMambaBlock)


def test_unfused_encoder_matches_outputs_and_input_gradients():
    torch.manual_seed(2026)
    mixer = GlobalTokenMixer(16, 4, 8, dropout=0.0).eval()
    original = torch.randn(2, 8, 16, requires_grad=True)
    portable = original.detach().clone().requires_grad_(True)
    expected = mixer.encoder(original)
    actual = mixer._encode_unfused(portable)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(portable.grad, original.grad)
    with torch.no_grad():
        torch.testing.assert_close(mixer._encode_unfused(original), mixer.encoder(original))
