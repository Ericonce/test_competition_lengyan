"""Weight-preserving device kernels for sequence classifiers.

These functions never move model tensors to CPU. CUDA keeps the existing
PyTorch modules; the explicit encoder path avoids NPU-unsupported fused
TransformerEncoderLayer/MultiheadAttention shortcuts.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def sequence_attention(query, key, value, *, dropout: float, backend: str):
    if backend not in {"auto", "fusion", "math"}:
        raise ValueError(f"Unknown NPU attention backend: {backend}")
    fusion = None
    if query.device.type == "npu" and backend != "math":
        import torch_npu
        fusion = getattr(torch_npu, "npu_fusion_attention", None)
    supported_fusion = fusion is not None and query.dtype in {torch.float16, torch.bfloat16}
    if backend == "fusion" and not supported_fusion:
        raise RuntimeError("NPU fusion attention requires torch_npu.npu_fusion_attention and FP16/BF16 inputs")
    if supported_fusion:
        # Keep failures visible: do not silently run an unsupported kernel on CPU.
        return fusion(
            query.contiguous(), key.contiguous(), value.contiguous(),
            query.size(1), "BNSD", scale=query.size(-1) ** -0.5,
            keep_prob=1.0 - dropout,
        )[0]
    # Accumulate the unfused half-precision path in FP32, like PyTorch's math
    # attention backend. All intermediate tensors remain on the same device.
    half = query.dtype in {torch.float16, torch.bfloat16}
    context = (torch.npu.amp.autocast(enabled=False) if query.device.type == "npu"
               else torch.autocast(device_type=query.device.type, enabled=False)) if half else nullcontext()
    with context:
        q, k, v = (query.float(), key.float(), value.float()) if half else (query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) * query.size(-1) ** -0.5
        weights = F.dropout(scores.softmax(dim=-1), p=dropout, training=dropout > 0)
        return torch.matmul(weights, v).to(query.dtype)


def encode_sequence(encoder: nn.TransformerEncoder, tokens: torch.Tensor, *, backend="auto"):
    """Execute the project's pre-norm encoder using its original parameters."""
    for layer in encoder.layers:
        if not layer.norm_first:
            raise ValueError("The sequence NPU encoder requires norm_first=True")
        attention = layer.self_attn
        normalized = layer.norm1(tokens)
        batch, length, width = normalized.shape
        heads = attention.num_heads
        projected = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias)
        query, key, value = projected.reshape(batch, length, 3, heads, width // heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = sequence_attention(query, key, value,
                                     dropout=attention.dropout if layer.training else 0.0,
                                     backend=backend)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        attended = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
        tokens = tokens + layer.dropout1(attended)
        feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(tokens)))))
        tokens = tokens + layer.dropout2(feedforward)
    return encoder.norm(tokens) if encoder.norm is not None else tokens


def causal_phase_conv1d(conv: nn.Conv1d, inputs: torch.Tensor):
    """Exactly factor a stride-1 dilated causal convolution into phase batches."""
    if conv.stride != (1,) or conv.padding != (0,) or conv.groups != 1:
        raise ValueError("Phase convolution requires stride=1, padding=0, groups=1")
    dilation = conv.dilation[0]
    kernel = conv.kernel_size[0]
    batch, channels, length = inputs.shape
    if dilation == 1:
        return conv(F.pad(inputs, (kernel - 1, 0)))
    extra = (-length) % dilation
    phases = F.pad(inputs, (0, extra)) if extra else inputs
    phase_length = phases.size(-1) // dilation
    phases = phases.reshape(batch, channels, phase_length, dilation).permute(0, 3, 1, 2)
    phases = phases.reshape(batch * dilation, channels, phase_length)
    outputs = F.conv1d(F.pad(phases, (kernel - 1, 0)), conv.weight, conv.bias)
    outputs = outputs.reshape(batch, dilation, conv.out_channels, phase_length).permute(0, 2, 3, 1)
    return outputs.reshape(batch, conv.out_channels, length + extra)[..., :length]


def causal_linear_conv1d(conv: nn.Conv1d, inputs: torch.Tensor):
    """Exact causal convolution as a small window matrix times unchanged weights."""
    if conv.stride != (1,) or conv.padding != (0,) or conv.groups != 1:
        raise ValueError("Linear convolution requires stride=1, padding=0, groups=1")
    dilation, kernel = conv.dilation[0], conv.kernel_size[0]
    batch, channels, length = inputs.shape
    padded = F.pad(inputs, ((kernel - 1) * dilation, 0)) if kernel > 1 else inputs
    taps = [padded[..., index * dilation:index * dilation + length].transpose(1, 2) for index in range(kernel)]
    windows = torch.stack(taps, dim=-1).reshape(batch * length, channels * kernel)
    outputs = F.linear(windows, conv.weight.flatten(1), conv.bias)
    return outputs.reshape(batch, length, conv.out_channels).transpose(1, 2)
