from copy import deepcopy

import pytest
import torch

from pulsecls.nn.modules.atenet import LearnableTimeEmbedding


@pytest.mark.parametrize("expanded", [False, True])
def test_cpu_time_embedding_matches_linear_outputs_and_gradients(expanded):
    torch.manual_seed(2026)
    embedding = LearnableTimeEmbedding(8)
    reference = deepcopy(embedding).double()
    time = torch.linspace(0, 1, 8).unsqueeze(0)
    time = (time.expand(3, -1) if expanded else time.repeat(3, 1)).requires_grad_()
    reference_time = time.detach().double().requires_grad_()

    actual = embedding(time)
    # Float64 uses the reference Linear implementation without the ARM FP32
    # primitive restriction. Compare both input and parameter derivatives.
    inputs = reference_time.unsqueeze(-1)
    expected = torch.cat([reference.linear(inputs), torch.sin(reference.periodic(inputs))], dim=-1)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.double())
    torch.testing.assert_close(time.grad.double(), reference_time.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in embedding.named_parameters():
        torch.testing.assert_close(parameter.grad.double(), reference.get_parameter(name).grad, rtol=1e-5, atol=1e-6)
    # Existing checkpoints must continue to load without key/shape migration.
    embedding.load_state_dict(reference.float().state_dict(), strict=True)


def test_cpu_time_embedding_preserves_autocast_output_dtype():
    embedding = LearnableTimeEmbedding(8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = embedding(torch.linspace(0, 1, 8).repeat(2, 1))
    assert actual.dtype == torch.bfloat16
    actual.float().square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in embedding.parameters())
