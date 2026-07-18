"""Correctness tests for the Triton MSDA operator.

Ground-truth chain:
  1. ``msda_naive`` (pure Python loops, hand-rolled bilinear) validates
     ``msda_reference`` (grid_sample-based) on CPU — no GPU needed.
  2. ``msda_reference`` in FP32 then validates the Triton kernels (forward
     output and all three analytic gradients) on GPU across dtypes, both
     grad_value accumulator paths, and edge cases (out-of-bounds sampling,
     exact grid corners, non-power-of-two head dims, non-contiguous inputs).
"""

import pytest
import torch

from reference_impls import msda_naive, msda_reference

try:  # requires triton, which CPU-only torch builds do not ship
    from msda_triton import multi_scale_deformable_attention
except ImportError:
    multi_scale_deformable_attention = None

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or multi_scale_deformable_attention is None,
    reason="CUDA GPU and triton required for the Triton kernels",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# Forward-output tolerances vs the FP32 reference. The kernel accumulates in
# FP32, so half-precision error comes from input/output storage rounding.
FWD_TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=5e-3, atol=5e-3),
    torch.bfloat16: dict(rtol=3e-2, atol=3e-2),
}
# Gradient tolerances are looser: grad_value is accumulated with atomics
# (order-nondeterministic; in native half dtype on the non-fp32-accum path)
# and grad_loc is scaled by the level resolution.
BWD_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=3e-2, atol=3e-2),
    torch.bfloat16: dict(rtol=8e-2, atol=8e-2),
}

CASES = [
    dict(id="decoder", B=2, Q=100, M=4, D=32, K=4, shapes=[(24, 32), (12, 16), (6, 8)]),
    dict(id="single_level_point", B=1, Q=17, M=1, D=16, K=1, shapes=[(13, 9)]),
    dict(id="nonpow2_D", B=2, Q=33, M=2, D=24, K=3, shapes=[(10, 14), (5, 7)]),
    dict(id="wide", B=1, Q=64, M=8, D=64, K=4, shapes=[(16, 16), (8, 8)]),
]


def make_inputs(B, Q, M, D, shapes, K, dtype, device, loc_lo=0.0, loc_hi=1.0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    L = len(shapes)
    S = sum(h * w for h, w in shapes)
    value = torch.randn(B, S, M, D, generator=gen).to(device, dtype)
    loc = loc_lo + (loc_hi - loc_lo) * torch.rand(B, Q, M, L, K, 2, generator=gen)
    loc = loc.to(device, dtype)
    attn = torch.rand(B, Q, M, L, K, generator=gen)
    attn = attn.flatten(3).softmax(-1).view(B, Q, M, L, K).to(device, dtype)
    spatial_shapes = torch.tensor(shapes, dtype=torch.long, device=device)
    hw = spatial_shapes.prod(-1)
    starts = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    return value, spatial_shapes, starts, loc, attn


def run_reference_fp32(value, spatial_shapes, loc, attn, grad_out=None):
    """Reference forward (and optionally backward) with FP32 leaf copies of
    the given inputs. Returns (out, grads or None)."""
    v = value.detach().float().requires_grad_(True)
    l = loc.detach().float().requires_grad_(True)
    a = attn.detach().float().requires_grad_(True)
    out = msda_reference(v, spatial_shapes, l, a)
    if grad_out is None:
        return out, None
    out.backward(grad_out.float())
    return out, (v.grad, l.grad, a.grad)


# ---------------------------------------------------------------------------
# Reference validation (CPU, no GPU required)
# ---------------------------------------------------------------------------


def test_reference_matches_naive_cpu():
    value, spatial_shapes, _, loc, attn = make_inputs(
        B=2, Q=5, M=2, D=4, shapes=[(5, 7), (3, 4)], K=3,
        dtype=torch.float64, device="cpu", loc_lo=-0.3, loc_hi=1.3,
    )
    ref = msda_reference(value, spatial_shapes, loc, attn)
    naive = msda_naive(value, spatial_shapes, loc, attn)
    torch.testing.assert_close(ref, naive, rtol=1e-10, atol=1e-12)


def test_reference_matches_naive_cpu_exact_corners():
    value, spatial_shapes, _, loc, attn = make_inputs(
        B=1, Q=4, M=1, D=3, shapes=[(4, 6)], K=2, dtype=torch.float64, device="cpu"
    )
    # Locations exactly on {0, 0.5, 1} exercise floor/boundary handling.
    grid_vals = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    loc = grid_vals[torch.randint(0, 3, loc.shape, generator=torch.Generator().manual_seed(1))]
    ref = msda_reference(value, spatial_shapes, loc, attn)
    naive = msda_naive(value, spatial_shapes, loc, attn)
    torch.testing.assert_close(ref, naive, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Triton forward
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_forward_matches_reference(case, dtype):
    case = {k: v for k, v in case.items() if k != "id"}
    value, spatial_shapes, starts, loc, attn = make_inputs(
        **case, dtype=dtype, device="cuda"
    )
    out = multi_scale_deformable_attention(value, spatial_shapes, starts, loc, attn)
    ref, _ = run_reference_fp32(value, spatial_shapes, loc, attn)
    torch.testing.assert_close(out.float(), ref, **FWD_TOL[dtype])


@requires_cuda
def test_forward_out_of_bounds_locations():
    value, spatial_shapes, starts, loc, attn = make_inputs(
        B=2, Q=64, M=4, D=32, shapes=[(14, 20), (7, 10)], K=4,
        dtype=torch.float32, device="cuda", loc_lo=-0.5, loc_hi=1.5,
    )
    out = multi_scale_deformable_attention(value, spatial_shapes, starts, loc, attn)
    ref, _ = run_reference_fp32(value, spatial_shapes, loc, attn)
    torch.testing.assert_close(out, ref, **FWD_TOL[torch.float32])


@requires_cuda
def test_forward_level_start_index_optional_and_noncontiguous():
    value, spatial_shapes, starts, loc, attn = make_inputs(
        B=2, Q=50, M=4, D=32, shapes=[(12, 16), (6, 8)], K=4,
        dtype=torch.float32, device="cuda",
    )
    out = multi_scale_deformable_attention(value, spatial_shapes, starts, loc, attn)
    out_nostarts = multi_scale_deformable_attention(
        value, spatial_shapes, None, loc, attn
    )
    torch.testing.assert_close(out, out_nostarts)
    # Non-contiguous inputs must be handled (contiguous copies inside the op).
    value_nc = value.transpose(1, 2).contiguous().transpose(1, 2)
    assert not value_nc.is_contiguous()
    torch.testing.assert_close(value_nc, value)
    out_nc = multi_scale_deformable_attention(value_nc, spatial_shapes, starts, loc, attn)
    torch.testing.assert_close(out, out_nc)


# ---------------------------------------------------------------------------
# Triton backward
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("fp32_grad_accum", [False, True], ids=["native_acc", "fp32_acc"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "case", [CASES[0], CASES[2]], ids=[CASES[0]["id"], CASES[2]["id"]]
)
def test_backward_matches_reference(case, dtype, fp32_grad_accum):
    case = {k: v for k, v in case.items() if k != "id"}
    value, spatial_shapes, starts, loc, attn = make_inputs(
        **case, dtype=dtype, device="cuda"
    )
    v = value.clone().requires_grad_(True)
    l = loc.clone().requires_grad_(True)
    a = attn.clone().requires_grad_(True)
    out = multi_scale_deformable_attention(
        v, spatial_shapes, starts, l, a, fp32_grad_accum=fp32_grad_accum
    )
    grad_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(7)).to(
        out.device, out.dtype
    )
    out.backward(grad_out)

    _, (gv_ref, gl_ref, ga_ref) = run_reference_fp32(
        value, spatial_shapes, loc, attn, grad_out
    )
    tol = BWD_TOL[dtype]
    torch.testing.assert_close(v.grad.float(), gv_ref, **tol)
    torch.testing.assert_close(l.grad.float(), gl_ref, **tol)
    torch.testing.assert_close(a.grad.float(), ga_ref, **tol)


@requires_cuda
def test_backward_out_of_bounds_locations():
    value, spatial_shapes, starts, loc, attn = make_inputs(
        B=2, Q=64, M=4, D=32, shapes=[(14, 20), (7, 10)], K=4,
        dtype=torch.float32, device="cuda", loc_lo=-0.5, loc_hi=1.5,
    )
    v = value.clone().requires_grad_(True)
    l = loc.clone().requires_grad_(True)
    a = attn.clone().requires_grad_(True)
    out = multi_scale_deformable_attention(v, spatial_shapes, starts, l, a)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    _, (gv_ref, gl_ref, ga_ref) = run_reference_fp32(
        value, spatial_shapes, loc, attn, grad_out
    )
    tol = BWD_TOL[torch.float32]
    torch.testing.assert_close(v.grad, gv_ref, **tol)
    torch.testing.assert_close(l.grad, gl_ref, **tol)
    torch.testing.assert_close(a.grad, ga_ref, **tol)


@requires_cuda
def test_fp32_accum_paths_agree_at_fp32():
    """At FP32 both accumulator paths use hardware FP32 atomics and must agree
    to rounding/reordering noise."""
    value, spatial_shapes, starts, loc, attn = make_inputs(
        B=2, Q=100, M=4, D=32, shapes=[(24, 32), (12, 16)], K=4,
        dtype=torch.float32, device="cuda",
    )
    grads = []
    for fp32_acc in (False, True):
        v = value.clone().requires_grad_(True)
        out = multi_scale_deformable_attention(
            v, spatial_shapes, starts, loc, attn, fp32_grad_accum=fp32_acc
        )
        out.backward(torch.ones_like(out))
        grads.append(v.grad)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@requires_cuda
def test_rejects_mismatched_shapes_and_dtypes():
    value, spatial_shapes, starts, loc, attn = make_inputs(
        B=1, Q=8, M=2, D=16, shapes=[(6, 6)], K=2, dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError, match="attention_weights"):
        multi_scale_deformable_attention(
            value, spatial_shapes, starts, loc, attn[:, :, :, :, :1]
        )
    with pytest.raises(ValueError, match="dtypes"):
        multi_scale_deformable_attention(
            value, spatial_shapes, starts, loc.half(), attn
        )
    with pytest.raises(ValueError, match="CUDA"):
        multi_scale_deformable_attention(
            value.cpu(), spatial_shapes.cpu(), starts.cpu(), loc.cpu(), attn.cpu()
        )
