"""Lab 10 solution: the shipped operator — autograd wrapper, validation, and
the final exam (running the repo's own test suite against your op).

Mirrors src/msda_triton/ops.py, built on the lab05/lab08 kernels through the
stage loader (so it runs on *your* kernels if you've saved them to labs/my/).
"""

import sys
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from labs.common.loader import stage
from labs.solutions.lab09 import use_fp32_grad_accum

_forward, FORWARD_SOURCE = stage(5, "msda_forward")
_backward, BACKWARD_SOURCE = stage(8, "msda_backward")

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class MSDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, fp32_grad_accum):
        out = _forward(value, spatial_shapes, level_start_index,
                       sampling_locations, attention_weights)
        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)
        ctx.fp32_grad_accum = fp32_grad_accum
        return out

    @staticmethod
    @once_differentiable  # no analytic double-backward: fail loudly, not wrong
    def backward(ctx, grad_out):
        value, spatial_shapes, level_start_index, loc, attn = ctx.saved_tensors
        grad_value, grad_loc, grad_attn = _backward(
            value, spatial_shapes, level_start_index, loc, attn, grad_out,
            fp32_grad_accum=ctx.fp32_grad_accum,
        )
        return grad_value, None, None, grad_loc, grad_attn, None


def msda_student(value, spatial_shapes, level_start_index, sampling_locations,
                 attention_weights, *, fp32_grad_accum=None):
    """Drop-in replacement for msda_triton.multi_scale_deformable_attention."""
    if value.dim() != 4:
        raise ValueError(f"value must be (B, S, M, D), got {tuple(value.shape)}")
    if sampling_locations.dim() != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError(
            f"sampling_locations must be (B, Q, M, L, K, 2), "
            f"got {tuple(sampling_locations.shape)}"
        )
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    if attention_weights.shape != (B, Q, M, L, K):
        raise ValueError(
            f"attention_weights must be {(B, Q, M, L, K)}, "
            f"got {tuple(attention_weights.shape)}"
        )
    if not (value.dtype == sampling_locations.dtype == attention_weights.dtype):
        raise ValueError(
            f"value/sampling_locations/attention_weights dtypes must match, got "
            f"{value.dtype}/{sampling_locations.dtype}/{attention_weights.dtype}"
        )
    if value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"unsupported dtype {value.dtype}; use fp32, fp16 or bf16")
    if not value.is_cuda:
        raise ValueError("the Triton operator requires CUDA tensors")
    if level_start_index is not None and level_start_index.shape != (L,):
        raise ValueError(
            f"level_start_index must be ({L},), got {tuple(level_start_index.shape)}"
        )
    if B == 0 or Q == 0:
        return value.new_zeros(B, Q, M * D)

    if fp32_grad_accum is None:
        fp32_grad_accum = use_fp32_grad_accum(
            value.dtype, torch.cuda.get_device_capability(value.device)
        )
    return MSDAFunction.apply(
        value.contiguous(), spatial_shapes, level_start_index,
        sampling_locations.contiguous(), attention_weights.contiguous(),
        fp32_grad_accum,
    )


def run_repo_suite(op=msda_student, extra_args=()):
    """The final exam: run tests/test_msda.py with ``op`` swapped in for the
    installed operator. Returns pytest's exit code (0 = all green).

    On a CUDA machine this runs the full suite; on CPU only the two
    reference-vs-naive tests execute (the rest skip — the same subset CI runs).
    """
    import pytest

    import msda_triton

    repo = Path(__file__).resolve().parent.parent.parent
    original = msda_triton.multi_scale_deformable_attention
    msda_triton.multi_scale_deformable_attention = op
    # test_msda.py does `from msda_triton import ...` at import time, so it
    # must not be sitting in sys.modules from an earlier run with the old op.
    sys.modules.pop("test_msda", None)
    try:
        return pytest.main(
            [str(repo / "tests" / "test_msda.py"), "-q", *extra_args]
        )
    finally:
        msda_triton.multi_scale_deformable_attention = original
        sys.modules.pop("test_msda", None)
