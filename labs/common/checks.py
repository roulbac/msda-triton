"""The labs' "supplied test scripts" (NAND2Tetris style).

Every lab checks the thing you just built against the repo's reference
implementations in ``tests/reference_impls.py`` — the exact same ground truth
the repo's own pytest suite uses, with the same tolerances. The assertion
helpers here are plain functions (also used by ``tests/test_labs.py``); the
marimo rendering lives in :func:`run_checks`.
"""

import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))

from reference_impls import msda_naive, msda_reference  # noqa: E402

__all__ = [
    "msda_naive", "msda_reference", "make_inputs",
    "FWD_TOL", "BWD_TOL", "assert_msda_matches", "assert_grads_match",
    "reference_grads", "run_checks", "NotDoneYet",
]

# Tolerances mirroring tests/test_msda.py: the kernel accumulates in FP32, so
# half-precision forward error is storage rounding; gradients are looser
# because grad_value is accumulated with atomics (order-nondeterministic).
FWD_TOL = {
    torch.float64: dict(rtol=1e-9, atol=1e-10),
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=5e-3, atol=5e-3),
    torch.bfloat16: dict(rtol=3e-2, atol=3e-2),
}
BWD_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=3e-2, atol=3e-2),
    torch.bfloat16: dict(rtol=8e-2, atol=8e-2),
}


class NotDoneYet(NotImplementedError):
    """Raise (or leave a bare ``raise NotDoneYet()`` in a skeleton) to mark an
    exercise as not attempted; checkers render it as 🚧 rather than ❌."""


def make_inputs(B, Q, M, D, shapes, K, dtype=torch.float32, device="cpu",
                loc_lo=0.0, loc_hi=1.0, seed=0):
    """Random MSDA inputs (mirrors tests/test_msda.py::make_inputs).

    Returns ``(value, spatial_shapes, level_start_index, loc, attn)``.
    """
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


# Small, CPU-friendly cases covering the repo suite's edge-case axes:
# multi-level, single-level/point, non-power-of-two D, out-of-bounds locs.
DEFAULT_CASES = [
    dict(B=2, Q=5, M=2, D=4, shapes=[(5, 7), (3, 4)], K=3),
    dict(B=1, Q=7, M=1, D=3, shapes=[(4, 6)], K=1),
    dict(B=1, Q=4, M=2, D=5, shapes=[(6, 5), (3, 3)], K=2, loc_lo=-0.4, loc_hi=1.4),
]


def assert_msda_matches(fn, *, cases=None, dtype=torch.float64, device="cpu",
                        tol=None, takes_starts=False):
    """Assert ``fn(value, spatial_shapes, loc, attn) -> (B, Q, M*D)`` agrees
    with the repo's grid_sample reference on every case (incl. out-of-bounds
    sampling locations). ``takes_starts``: fn also wants level_start_index.
    """
    tol = tol or FWD_TOL[dtype]
    for case in cases or DEFAULT_CASES:
        value, shapes, starts, loc, attn = make_inputs(
            **case, dtype=dtype, device=device
        )
        ref = msda_reference(value, shapes, loc, attn)
        out = (fn(value, shapes, starts, loc, attn) if takes_starts
               else fn(value, shapes, loc, attn))
        assert out.shape == ref.shape, (
            f"shape {tuple(out.shape)} != expected {tuple(ref.shape)} "
            f"for case {case}"
        )
        torch.testing.assert_close(out, ref, **tol)


def reference_grads(value, spatial_shapes, loc, attn, grad_out):
    """(grad_value, grad_loc, grad_attn) of the FP32 reference via autograd —
    the machine-derived truth your hand-derived gradients must match."""
    v = value.detach().float().requires_grad_(True)
    l = loc.detach().float().requires_grad_(True)
    a = attn.detach().float().requires_grad_(True)
    out = msda_reference(v, spatial_shapes, l, a)
    out.backward(grad_out.float())
    return v.grad, l.grad, a.grad


def assert_grads_match(backward_fn, *, cases=None, dtype=torch.float32,
                       device="cpu", tol=None):
    """Assert ``backward_fn(value, spatial_shapes, starts, loc, attn,
    grad_out) -> (grad_value, grad_loc, grad_attn)`` matches autograd on the
    FP32 reference."""
    tol = tol or BWD_TOL[dtype]
    for case in cases or DEFAULT_CASES:
        value, shapes, starts, loc, attn = make_inputs(
            **case, dtype=dtype, device=device
        )
        B, Q, M, D = value.shape[0], loc.shape[1], value.shape[2], value.shape[3]
        gen = torch.Generator().manual_seed(7)
        grad_out = torch.randn(B, Q, M * D, generator=gen).to(device, dtype)
        gv, gl, ga = backward_fn(value, shapes, starts, loc, attn, grad_out)
        gv_ref, gl_ref, ga_ref = reference_grads(value, shapes, loc, attn, grad_out)
        torch.testing.assert_close(gv.float(), gv_ref, **tol)
        torch.testing.assert_close(gl.float(), gl_ref, **tol)
        torch.testing.assert_close(ga.float(), ga_ref, **tol)


def run_checks(checks):
    """Run ``{label: zero-arg callable}`` and render a marimo results table.

    Callables should raise ``AssertionError`` on failure and ``NotDoneYet``
    (or ``NotImplementedError``) when the exercise is still a stub. Reactive
    re-runs are marimo's doing: edit your implementation cell and this cell
    re-executes automatically.
    """
    import marimo as mo

    rows, all_green = [], True
    for label, fn in checks.items():
        try:
            fn()
        except (NotDoneYet, NotImplementedError):
            rows.append(f"| 🚧 | **{label}** | not implemented yet |")
            all_green = False
        except AssertionError as e:
            msg = str(e).strip().splitlines()
            head = msg[0] if msg else "assertion failed"
            rows.append(f"| ❌ | **{label}** | {head} |")
            all_green = False
        except Exception:
            err = traceback.format_exc().strip().splitlines()[-1]
            rows.append(f"| 💥 | **{label}** | `{err}` |")
            all_green = False
        else:
            rows.append(f"| ✅ | **{label}** | passed |")
    table = "\n".join(["| | check | result |", "|---|---|---|", *rows])
    if all_green:
        return mo.md(f"{table}\n\n**All checks passed — stage complete.** 🏁").callout(
            kind="success"
        )
    return mo.md(table).callout(kind="neutral")
