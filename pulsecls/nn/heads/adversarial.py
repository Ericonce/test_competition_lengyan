from __future__ import annotations

import torch


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * grad_output, None


def gradient_reversal(
    inputs: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    return _GradientReversal.apply(inputs, float(coefficient))
