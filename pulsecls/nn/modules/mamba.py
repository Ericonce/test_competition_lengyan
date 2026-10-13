"""Mamba blocks for PDW sequences.

``ssm`` prefers CUDA ``mamba-ssm`` kernels. ``pure`` is a PyTorch selective-scan
fallback for machines without those kernels, including Ascend NPU.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PureMambaBlock(nn.Module):
    """Lightweight selective SSM-style block without CUDA kernels."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _batch, length, _dim = x.shape
        xz = self.in_proj(x)
        x_branch, gate = xz.chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :length]
        x_branch = F.silu(x_branch.transpose(1, 2))
        outputs = self._selective_scan(x_branch)
        outputs = outputs * F.silu(gate)
        return self.out_proj(outputs)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, inner = x.shape
        projected = self.x_proj(x)
        delta, state_b, state_c = torch.split(
            projected,
            [inner, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta)).clamp(min=1e-4, max=20.0)
        decay = -torch.exp(self.A_log.float())
        hidden = torch.zeros(batch, inner, self.d_state, device=x.device, dtype=x.dtype)
        steps = []
        for time in range(length):
            step = delta[:, time].unsqueeze(-1)
            discrete_a = torch.exp(step * decay)
            discrete_b = step * state_b[:, time].unsqueeze(1)
            hidden = discrete_a * hidden + discrete_b * x[:, time].unsqueeze(-1)
            readout = (hidden * state_c[:, time].unsqueeze(1)).sum(-1)
            steps.append(readout + self.D * x[:, time])
        return torch.stack(steps, dim=1)


class SSMMambaBlock(nn.Module):
    """Prefer Mamba2, then Mamba1, from mamba-ssm."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.mamba = self._build(d_model, d_state, d_conv, expand)

    @staticmethod
    def _build(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2

            inner = int(expand * d_model)
            headdim = 64
            while headdim > 8 and inner % headdim != 0:
                headdim //= 2
            return Mamba2(
                d_model=d_model,
                d_state=max(d_state, 16),
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
        except Exception as mamba2_error:
            first_error = mamba2_error
        try:
            from mamba_ssm.modules.mamba_simple import Mamba

            return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        except Exception as mamba1_error:
            raise ImportError(
                f"mamba-ssm Mamba2/Mamba unavailable: mamba2={first_error}; mamba1={mamba1_error}"
            ) from mamba1_error

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


def build_mamba_block(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    backend: str = "auto",
) -> nn.Module:
    choice = str(backend).lower()
    if choice == "auto":
        try:
            from pulsecls.utils.device import parse_device_spec

            # Backend detection must not reset the trainer's current NPU index.
            if parse_device_spec().type == "npu":
                choice = "pure"
            else:
                choice = "ssm"
        except Exception:
            choice = "ssm"
    if choice == "ssm":
        try:
            return SSMMambaBlock(d_model, d_state, d_conv, expand)
        except Exception as error:
            print(f"[MambaSL] mamba-ssm unavailable ({error}); falling back to pure.")
            return PureMambaBlock(d_model, d_state, d_conv, expand)
    if choice == "pure":
        return PureMambaBlock(d_model, d_state, d_conv, expand)
    raise ValueError(f"Unknown mamba_backend: {backend}")
