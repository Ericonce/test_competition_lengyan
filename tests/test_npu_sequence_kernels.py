"""Check that device-specialized paths preserve the original models' mathematics."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pulsecls.nn.modules.npu_sequence import causal_linear_conv1d, causal_phase_conv1d, encode_sequence


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
@pytest.mark.parametrize("length", [31, 32, 35])
@pytest.mark.parametrize("convolution", [causal_phase_conv1d, causal_linear_conv1d])
def test_phase_convolution_preserves_causality_outputs_and_gradients(dilation, length, convolution):
    torch.manual_seed(42)
    conv = nn.Conv1d(3, 5, 3, dilation=dilation)
    other = copy.deepcopy(conv)
    x = torch.randn(2, 3, length, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = conv(F.pad(x, (2 * dilation, 0)))
    actual = convolution(other, y)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(conv.weight.grad, other.weight.grad, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(conv.bias.grad, other.bias.grad, rtol=1e-5, atol=3e-6)


def test_explicit_encoder_preserves_existing_weights_outputs_and_gradients():
    torch.manual_seed(42)
    layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True, norm_first=True, activation="gelu")
    encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(16), enable_nested_tensor=False)
    other = copy.deepcopy(encoder)
    x = torch.randn(2, 13, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = encoder(x)
    actual = encode_sequence(other, y, backend="math")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=3e-6)
    for (_, p), (_, q) in zip(encoder.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=8e-6)
