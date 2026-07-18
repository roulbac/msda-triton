"""CI for the labs: reference solutions must satisfy the labs' own checkers,
kernel solutions must run (CPU interpreter) and compile (AOT), and every
notebook must execute headlessly with its exercises still stubbed.

Triton-dependent pieces run in subprocesses because TRITON_INTERPRET must be
chosen *before* ``import triton``, and interpreter mode and AOT compilation
need opposite settings.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from labs.common import checks  # noqa: E402
from labs.common import memsim  # noqa: E402

HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_MARIMO = importlib.util.find_spec("marimo") is not None

requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
requires_marimo = pytest.mark.skipif(not HAS_MARIMO, reason="marimo not installed")


def run_py(code: str, extra_env=None, timeout=900) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, (
        f"subprocess failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Pure-torch solutions (labs 0, 1, 2, 3, 7) — in-process
# ---------------------------------------------------------------------------


def test_lab00_solution_matches_reference():
    from labs.solutions import lab00

    checks.assert_msda_matches(
        lambda v, s, l, a: lab00.msda(v, s, l, a), dtype=torch.float64
    )
    checks.assert_msda_matches(
        lambda v, s, l, a: lab00.msda(v, s, l, a), dtype=torch.float32
    )


def test_lab00_layers_collapse_and_train():
    """With offset_proj/weight_proj zeroed, the lab-0 layers must degenerate
    to plain linear reads (the notebooks' 'collapse' checks), and gradients
    must reach the offset predictors through the bilinear sampling path."""
    from labs.solutions import lab00

    torch.manual_seed(0)
    layer = lab00.DeformableAttention(8, 2, 3)
    with torch.no_grad():
        layer.offset_proj.weight.zero_()
        layer.offset_proj.bias.zero_()
        layer.weight_proj.weight.zero_()
        layer.weight_proj.bias.zero_()
    x = torch.randn(2, 5, 6, 8)
    torch.testing.assert_close(
        layer(x), layer.out_proj(layer.value_proj(x)), rtol=1e-5, atol=1e-6
    )

    cross = lab00.MSDACrossAttention(8, 2, 2, 3)
    shapes = torch.tensor([[5, 7], [3, 4]])
    query = torch.randn(2, 5, 8)
    ref = torch.rand(2, 5, 2, requires_grad=True)
    value = torch.randn(2, 47, 8)
    out = cross(query, ref, value, shapes)
    out.sum().backward()
    assert cross.offset_proj.weight.grad.abs().sum() > 0
    assert ref.grad.abs().sum() > 0

    with torch.no_grad():
        cross.offset_proj.weight.zero_()
        cross.offset_proj.bias.zero_()
        cross.weight_proj.weight.zero_()
        cross.weight_proj.bias.zero_()
    out = cross(query, ref, value, shapes)
    v = cross.value_proj(value).reshape(2, 47, 2, 4)
    locs = ref.detach()[:, :, None, None, None, :].expand(2, 5, 2, 2, 3, 2)
    w = torch.full((2, 5, 2, 2, 3), 1 / 6)
    expected = cross.out_proj(checks.msda_reference(v, shapes, locs, w))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_lab01_solution_matches_reference():
    from labs.solutions import lab01

    checks.assert_msda_matches(
        lambda v, s, l, a: lab01.msda_naive_student(v, s, l, a),
        dtype=torch.float64,
    )


def test_lab02_solution_matches_reference():
    from labs.solutions import lab02

    checks.assert_msda_matches(
        lab02.flat_msda, dtype=torch.float64, takes_starts=True
    )
    checks.assert_msda_matches(
        lab02.flat_msda, dtype=torch.float32, takes_starts=True
    )


def test_lab03_solution_counts_sectors():
    from labs.solutions import lab03

    assert lab03.efficiency(memsim.contiguous(1024, 4)) == 1.0
    assert lab03.efficiency(memsim.strided(16, 1024, 4)) <= 0.125 + 1e-9
    qb = lab03.efficiency(memsim.msda_query_block())
    pp = lab03.efficiency(memsim.msda_point_parallel())
    assert qb / pp > 4, "query-block tiling must dominate in the model"


def test_lab07_solution_matches_autograd():
    from labs.solutions import lab07

    checks.assert_grads_match(lab07.msda_backward_torch, dtype=torch.float32)


def test_lab09_switch_is_pure_and_correct():
    from labs.solutions.lab09 import use_fp32_grad_accum

    assert use_fp32_grad_accum(torch.bfloat16, (8, 0)) is True
    assert use_fp32_grad_accum(torch.bfloat16, (9, 0)) is False
    assert use_fp32_grad_accum(torch.float16, (8, 0)) is False
    assert use_fp32_grad_accum(torch.float32, (8, 0)) is False


# ---------------------------------------------------------------------------
# Kernel solutions (labs 4, 5, 8, 10) — CPU interpreter, subprocess
# ---------------------------------------------------------------------------

_INTERPRETED_SUITE = """
import torch
from labs.common import setup, checks
assert setup.INTERPRETED or setup.HAS_CUDA
from labs.solutions import lab04, lab05, lab07, lab08

x, y = torch.randn(400), torch.randn(400)
torch.testing.assert_close(lab04.add(x, y), x + y)
a = torch.randn(19, 11)
torch.testing.assert_close(lab04.copy2d(a), a)
src, idx = torch.randn(100), torch.randint(0, 100, (37,))
torch.testing.assert_close(lab04.gather(src, idx), src[idx])

case = dict(B=1, Q=4, M=2, D=4, shapes=[(5, 7), (3, 4)], K=2)
v, s, st, loc, attn = checks.make_inputs(**case, loc_lo=-0.3, loc_hi=1.3)
out = lab05.msda_forward(v, s, st, loc, attn, BLOCK_Q=4)
torch.testing.assert_close(out, checks.msda_reference(v, s, loc, attn),
                           rtol=1e-5, atol=1e-5)

g = torch.randn(*out.shape)
gv, gl, ga = lab08.msda_backward(v, s, st, loc, attn, g, BLOCK_Q=4)
gv_t, gl_t, ga_t = lab07.msda_backward_torch(v, s, st, loc, attn, g)
torch.testing.assert_close(gv, gv_t, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(gl, gl_t, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(ga, ga_t, rtol=1e-4, atol=1e-4)

idx = torch.randint(0, 8, (500,))
torch.testing.assert_close(lab08.histogram(idx, 8, atomic=True, BLOCK=64),
                           torch.bincount(idx, minlength=8).float())

from labs.solutions.lab10 import MSDAFunction
vr = v.clone().requires_grad_(True)
lr = loc.clone().requires_grad_(True)
ar = attn.clone().requires_grad_(True)
out2 = MSDAFunction.apply(vr, s, st, lr, ar, False)
out2.backward(g)
gv_r, gl_r, ga_r = checks.reference_grads(v, s, loc, attn, g)
torch.testing.assert_close(vr.grad, gv_r, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(lr.grad, gl_r, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(ar.grad, ga_r, rtol=1e-4, atol=1e-4)
print("interpreted kernel suite OK")
"""


@requires_triton
def test_kernel_solutions_run_interpreted():
    out = run_py(_INTERPRETED_SUITE, extra_env={"TRITON_INTERPRET": "1"})
    assert "interpreted kernel suite OK" in out


_AOT_SUITE = """
import os
assert os.environ["TRITON_INTERPRET"] == "0"
from labs.solutions import lab08, lab09

report = lab09.atomic_report(lab08.msda_backward_kernel,
                             cc_list=(80, 90), dtypes=("fp32", "bf16"))
for key, atoms in sorted(report.items()):
    assert atoms, f"no atomics found for {key} — grad_value scatter missing?"
    print(key, "->", atoms)
print("aot compile suite OK")
"""


@requires_triton
def test_kernel_solutions_compile_aot():
    """Compiles the solution backward for SM 8.0/9.0 x {native, fp32acc} and
    checks each PTX contains atomic instructions. The exact lowering (CAS vs
    native) is printed, not asserted — same policy as the repo's own
    compile_kernels_check.py: a future Triton may change its spelling."""
    out = run_py(_AOT_SUITE, extra_env={"TRITON_INTERPRET": "0"})
    assert "aot compile suite OK" in out


# ---------------------------------------------------------------------------
# Notebooks execute headlessly with exercises stubbed
# ---------------------------------------------------------------------------

_PURE_NOTEBOOKS = [
    "00_the_idea.py",
    "01_the_spec.py",
    "02_tensors_are_pointers.py",
    "03_the_machine.py",
    "07_the_other_direction.py",
]
_TRITON_NOTEBOOKS = [
    "04_hello_triton.py",
    "05_forward_correct.py",
    "06_make_it_fast.py",
    "08_concurrent_writes.py",
    "09_hardware_detective.py",
    "10_ship_it.py",
]


def _run_notebook(name):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run(
        [sys.executable, str(REPO / "labs" / name)], cwd=REPO, env=env,
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"{name} failed headlessly:\n{result.stdout[-2000:]}\n"
        f"{result.stderr[-2000:]}"
    )


@requires_marimo
@pytest.mark.parametrize("name", _PURE_NOTEBOOKS)
def test_notebook_runs_headless(name):
    _run_notebook(name)


@requires_marimo
@requires_triton
@pytest.mark.parametrize("name", _TRITON_NOTEBOOKS)
def test_triton_notebook_runs_headless(name):
    _run_notebook(name)


# ---------------------------------------------------------------------------
# The labs/my escape hatch
# ---------------------------------------------------------------------------


def test_loader_falls_back_to_solutions():
    from labs.common import loader

    fn, kind = loader.stage(1, "msda_naive_student")
    assert "reference" in kind and callable(fn)
    with pytest.raises(ImportError):
        loader.stage(1, "no_such_artifact")
