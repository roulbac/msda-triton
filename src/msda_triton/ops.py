"""Public MSDA operator backed by the Triton kernels in kernels.py."""

from __future__ import annotations

import torch
import triton
from torch.autograd.function import once_differentiable

from .kernels import BWD_BLOCK_Q, BWD_NUM_WARPS, _msda_backward_kernel, _msda_forward_kernel

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _use_fp32_grad_accum(value: torch.Tensor) -> bool:
    """Paper Section 4.1: accumulate grad_value in FP32 on GPUs whose atomic
    add for the value dtype is an emulated compare-and-swap retry loop, which
    collapses atomic throughput. Only BF16 lacks a native atomic add before
    SM 9.0 (PTX ISA 7.8); FP16 has had ``red.add.noftz.f16x2`` since SM 6.0,
    and forcing it through an FP32 scratch buffer measures ~2-4x slower than
    native FP16 atomics on Ampere/Ada."""
    if value.dtype != torch.bfloat16:
        return False
    return torch.cuda.get_device_capability(value.device)[0] < 9


class _MSDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, fp32_grad_accum):
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        out = value.new_empty(B, Q, M, D)

        grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_Q"]), B * M)
        _msda_forward_kernel[grid](
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, out,
            Q, S, M=M, D=D, L=L, K=K, BLOCK_D=triton.next_power_of_2(D),
        )

        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        ctx.fp32_grad_accum = fp32_grad_accum
        return out.view(B, Q, M * D)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
            ctx.saved_tensors
        )
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        grad_out = grad_out.reshape(B, Q, M, D).contiguous()

        acc_dtype = torch.float32 if ctx.fp32_grad_accum else value.dtype
        grad_value = torch.zeros_like(value, dtype=acc_dtype)
        grad_loc = torch.empty_like(sampling_locations)
        grad_attn = torch.empty_like(attention_weights)

        grid = (triton.cdiv(Q, BWD_BLOCK_Q), B * M)
        _msda_backward_kernel[grid](
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
            grad_out, grad_value, grad_loc, grad_attn,
            Q, S, M=M, D=D, L=L, K=K,
            BLOCK_Q=BWD_BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
            num_warps=BWD_NUM_WARPS,
        )

        if grad_value.dtype != value.dtype:
            grad_value = grad_value.to(value.dtype)
        return grad_value, None, None, grad_loc, grad_attn, None


def multi_scale_deformable_attention(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor | None,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    fp32_grad_accum: bool | None = None,
) -> torch.Tensor:
    """Multi-scale deformable attention (Zhu et al. 2021, Eq. 1), Triton-backed.

    For each query, head, feature level l and sampling point k, bilinearly
    samples ``value`` at ``sampling_locations`` and aggregates the samples
    weighted by ``attention_weights``. Sampling uses the reference CUDA
    convention ``x_im = x * W_l - 0.5`` (== ``grid_sample`` with
    ``align_corners=False, padding_mode="zeros"``).

    Args:
        value: ``(B, S, M, D)`` flattened multi-scale feature maps, where
            ``S = sum_l H_l * W_l``, ``M`` attention heads, head dim ``D``.
        spatial_shapes: ``(L, 2)`` integer ``(H_l, W_l)`` per level.
        level_start_index: ``(L,)`` integer offset of each level in the ``S``
            axis, or ``None`` to compute it from ``spatial_shapes``.
        sampling_locations: ``(B, Q, M, L, K, 2)`` normalized ``(x, y)`` in
            ``[0, 1]``; locations outside sample zeros.
        attention_weights: ``(B, Q, M, L, K)``, typically softmax-normalized
            over the ``(L, K)`` axes (not enforced here).
        fp32_grad_accum: accumulate ``grad_value`` in an FP32 scratch buffer
            instead of native-dtype atomics. ``None`` (default) auto-selects:
            FP32 accumulation for bf16 inputs on SM < 9.0 (where the bf16
            atomic add is an emulated compare-and-swap), native atomics
            otherwise — fp16 atomics are hardware-native from SM 6.0 on
            (paper Section 4.1).

    Returns:
        ``(B, Q, M * D)`` attention output, same dtype as ``value``.

    Note:
        ``grad_value`` is accumulated with atomics, so its backward is not
        bitwise deterministic (FP32 accumulation is order-independent to FP32
        rounding; native half accumulation is looser).
    """
    if value.dim() != 4:
        raise ValueError(f"value must be (B, S, M, D), got {tuple(value.shape)}")
    if sampling_locations.dim() != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError(
            f"sampling_locations must be (B, Q, M, L, K, 2), got {tuple(sampling_locations.shape)}"
        )
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    if sampling_locations.shape[0] != B or sampling_locations.shape[2] != M:
        raise ValueError(
            f"sampling_locations {tuple(sampling_locations.shape)} inconsistent with value {tuple(value.shape)}"
        )
    if attention_weights.shape != (B, Q, M, L, K):
        raise ValueError(
            f"attention_weights must be {(B, Q, M, L, K)}, got {tuple(attention_weights.shape)}"
        )
    if spatial_shapes.shape != (L, 2):
        raise ValueError(f"spatial_shapes must be ({L}, 2), got {tuple(spatial_shapes.shape)}")
    if not (value.dtype == sampling_locations.dtype == attention_weights.dtype):
        raise ValueError(
            f"value/sampling_locations/attention_weights dtypes must match, got "
            f"{value.dtype}/{sampling_locations.dtype}/{attention_weights.dtype}"
        )
    if value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"unsupported dtype {value.dtype}; use fp32, fp16 or bf16")
    if not value.is_cuda:
        raise ValueError(
            "the Triton operator requires CUDA tensors; for CPU use the "
            "reference implementation in tests/reference_impls.py"
        )

    spatial_shapes = spatial_shapes.to(device=value.device, dtype=torch.int64).contiguous()
    if level_start_index is None:
        hw = spatial_shapes.prod(-1)
        level_start_index = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    else:
        if level_start_index.shape != (L,):
            raise ValueError(
                f"level_start_index must be ({L},), got {tuple(level_start_index.shape)}"
            )
        level_start_index = level_start_index.to(
            device=value.device, dtype=torch.int64
        ).contiguous()

    if B == 0 or Q == 0:
        return value.new_zeros(B, Q, M * D)

    if fp32_grad_accum is None:
        fp32_grad_accum = _use_fp32_grad_accum(value)

    return _MSDAFunction.apply(
        value.contiguous(),
        spatial_shapes,
        level_start_index,
        sampling_locations.contiguous(),
        attention_weights.contiguous(),
        fp32_grad_accum,
    )
