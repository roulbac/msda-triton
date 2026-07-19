import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys

    import marimo as mo

    _repo = mo.notebook_dir().parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    import torch

    from labs.common import checks, setup
    from labs.common.loader import stage, stage_banner

    triton, tl = setup.import_triton()
    return checks, mo, setup, stage, stage_banner, torch


@app.cell(hide_code=True)
def _(mo, setup):
    setup.banner(mo)
    return


@app.cell(hide_code=True)
def _(mo, stage, stage_banner):
    msda_forward, _f_src = stage(5, "msda_forward")
    msda_backward, _b_src = stage(8, "msda_backward")
    use_fp32_grad_accum, _s_src = stage(9, "use_fp32_grad_accum")
    stage_banner(mo, {
        "msda_forward": _f_src,
        "msda_backward": _b_src,
        "use_fp32_grad_accum": _s_src,
    })
    return msda_backward, msda_forward, use_fp32_grad_accum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 10 — Ship it

    You have kernels. A *library* is kernels plus three unglamorous layers that
    decide whether anyone can actually use them:

    1. **autograd integration** — so `loss.backward()` just works;
    2. **input validation** — so mistakes fail loudly at the boundary with real
       error messages, not as garbage reads inside a kernel;
    3. **honest benchmarks** — so the numbers you tell people are true.

    Then the final exam: this repo ships a pytest suite for its operator. Your
    operator has the same contract. **You will run the repo's own tests against
    your implementation.** *From NAND to Tetris* graded every chip against
    supplied test scripts; this is ours.

    ## 1. ✏️ Plugging into autograd

    Custom ops register with autograd via `torch.autograd.Function`: a `forward`
    that stashes what the backward will need, and a `backward` returning one
    gradient per `forward` argument (or `None` for non-differentiable ones —
    count them carefully, it's the classic mistake).
    """)
    return


@app.cell
def _(checks, msda_backward, msda_forward, torch):
    from torch.autograd.function import once_differentiable

    class MSDAFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights, fp32_grad_accum):
            # ================= YOUR CODE =================
            # 1. out = msda_forward(...)
            # 2. ctx.save_for_backward(<the five tensors the backward re-reads>)
            #    ctx.fp32_grad_accum = fp32_grad_accum
            # 3. return out
            raise checks.NotDoneYet()
            # =============================================

        @staticmethod
        @once_differentiable  # no analytic double-backward: error > silent garbage
        def backward(ctx, grad_out):
            # ================= YOUR CODE =================
            # 1. unpack ctx.saved_tensors
            # 2. call msda_backward(..., fp32_grad_accum=ctx.fp32_grad_accum)
            # 3. return SIX things: grad_value, None, None, grad_loc,
            #    grad_attn, None   (one per forward argument, same order)
            raise checks.NotDoneYet()
            # =============================================
    return (MSDAFunction,)


@app.cell(hide_code=True)
def _(MSDAFunction, checks, setup, torch):
    def _gradients_flow():
        _dev = "cuda" if setup.HAS_CUDA else "cpu"
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=4, M=2, D=4, shapes=[(5, 6)], K=2,
            dtype=torch.float32, device=_dev,
        )
        _vr = _v.clone().requires_grad_(True)
        _lr = _l.clone().requires_grad_(True)
        _ar = _a.clone().requires_grad_(True)
        _out = MSDAFunction.apply(_vr, _s, _st, _lr, _ar, False)
        _g = torch.randn_like(_out)
        _out.backward(_g)
        _gv, _gl, _ga = checks.reference_grads(_v, _s, _l, _a, _g)
        torch.testing.assert_close(_vr.grad, _gv, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(_lr.grad, _gl, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(_ar.grad, _ga, rtol=1e-4, atol=1e-4)

    checks.run_checks({
        "`.backward()` produces all three correct gradients": _gradients_flow,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint — both methods": mo.md(
            "```python\n# forward\nout = msda_forward(value, spatial_shapes, "
            "level_start_index,\n                   sampling_locations, "
            "attention_weights)\nctx.save_for_backward(value, spatial_shapes, "
            "level_start_index,\n                      sampling_locations, "
            "attention_weights)\nctx.fp32_grad_accum = fp32_grad_accum\n"
            "return out\n\n# backward\nvalue, shapes, starts, loc, attn = "
            "ctx.saved_tensors\ngv, gl, ga = msda_backward(value, shapes, starts, "
            "loc, attn, grad_out,\n                           fp32_grad_accum="
            "ctx.fp32_grad_accum)\nreturn gv, None, None, gl, ga, None\n```"
        ),
        "Why save inputs instead of the corner values?": mo.md(
            "Recompute-vs-store: the backward re-derives corner indices and "
            "weights from the saved inputs (cheap arithmetic) rather than "
            "storing 4×D floats per sample point from the forward (a lot of "
            "memory *and* a lot of extra traffic — the scarce resource, as "
            "Lab 3 taught)."
        ),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ✏️ The front door

    Wrap `MSDAFunction.apply` in the user-facing function: validate everything,
    default `level_start_index` when the caller passes `None`, auto-select the
    Lab 9 switch, handle empty batches without launching. Kernels *trust* their
    layout contract — this function is where that trust is earned.
    """)
    return


@app.cell
def _(MSDAFunction, checks, torch, use_fp32_grad_accum):
    def msda_student(value, spatial_shapes, level_start_index,
                     sampling_locations, attention_weights, *,
                     fp32_grad_accum=None):
        """Drop-in for msda_triton.multi_scale_deformable_attention."""
        # ================= YOUR CODE =================
        # Raise ValueError (with a useful message!) unless:
        #   value.dim() == 4;  sampling_locations is (B, Q, M, L, K, 2);
        #   attention_weights is (B, Q, M, L, K);  the three dtypes match and
        #   are one of fp32/fp16/bf16;  value.is_cuda.
        # Then:
        #   if B == 0 or Q == 0: return value.new_zeros(B, Q, M * D)
        #   if fp32_grad_accum is None:
        #       fp32_grad_accum = use_fp32_grad_accum(value.dtype,
        #           torch.cuda.get_device_capability(value.device))
        #   return MSDAFunction.apply(value.contiguous(), spatial_shapes,
        #       level_start_index, sampling_locations.contiguous(),
        #       attention_weights.contiguous(), fp32_grad_accum)
        raise checks.NotDoneYet()
        # =============================================
    return (msda_student,)


@app.cell(hide_code=True)
def _(checks, msda_student, setup, torch):
    def _rejects_garbage():
        _dev = "cuda" if setup.HAS_CUDA else "cpu"
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=8, M=2, D=16, shapes=[(6, 6)], K=2,
            dtype=torch.float32, device=_dev,
        )
        for _bad_call, _label in [
            (lambda: msda_student(_v[0], _s, _st, _l, _a), "3-D value"),
            (lambda: msda_student(_v, _s, _st, _l, _a[..., :1]), "truncated attn"),
            (lambda: msda_student(_v, _s, _st, _l.half(), _a), "mixed dtypes"),
            (lambda: msda_student(_v.double(), _s, _st, _l.double(),
                                  _a.double()), "float64"),
        ]:
            try:
                _bad_call()
            except ValueError:
                continue
            except checks.NotDoneYet:
                raise
            raise AssertionError(f"{_label}: should have raised ValueError")

    def _cpu_tensor_rejected_or_runs():
        # On GPU sessions CPU input must raise; under the interpreter your
        # wrapper may accept CPU tensors — then it must be *correct* instead.
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=4, M=1, D=4, shapes=[(4, 5)], K=1, dtype=torch.float32
        )
        try:
            _out = msda_student(_v, _s, _st, _l, _a)
        except ValueError:
            return
        torch.testing.assert_close(
            _out, checks.msda_reference(_v, _s, _l, _a), rtol=1e-5, atol=1e-5
        )

    checks.run_checks({
        "bad inputs raise ValueError at the boundary": _rejects_garbage,
        "CPU tensors: rejected (or handled correctly)": _cpu_tensor_rejected_or_runs,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 🎓 The final exam

    `tests/test_msda.py` validates the installed operator: forward and all three
    gradients, across dtypes, both accumulator paths, out-of-bounds locations,
    exact grid corners, non-power-of-two head dims, non-contiguous inputs.
    The cell below monkeypatches **your** `msda_student` into the `msda_triton`
    package and runs that exact suite.

    On a CUDA machine, everything runs. On CPU, the GPU tests skip and only the
    reference self-checks execute — do the full run on a GPU for the real
    diploma. *(The `raise ValueError("...CUDA...")` your wrapper does for CPU
    tensors is itself under test here — `test_rejects_mismatched_shapes_and_
    dtypes` expects it.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    exam_button = mo.ui.run_button(label="🎓 Run the repo's test suite against MY operator")
    exam_button
    return (exam_button,)


@app.cell(hide_code=True)
def _(exam_button, mo, msda_student):
    if exam_button.value:
        from labs.solutions.lab10 import run_repo_suite

        _code = run_repo_suite(op=msda_student)
        if _code == 0:
            _msg = mo.md(
                "## ✅ PASSED\n\nThe repo's own test suite just certified an "
                "operator you built from bilinear interpolation up. That's the "
                "whole course."
            ).callout(kind="success")
        else:
            _msg = mo.md(
                f"❌ pytest exit code {_code} — scroll the terminal output for "
                f"the failing case; the parametrized IDs (dtype, case, "
                f"accumulator path) tell you which lab to revisit."
            ).callout(kind="danger")
    else:
        _msg = mo.md("*(click when ready — output appears in the terminal "
                     "running marimo)*")
    _msg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Epilogue: benchmarking honestly, and what's left on the table

    Two facts before you quote numbers to anyone (the full treatment is
    [course chapter 6](https://roulbac.github.io/msda-triton/course/06-benchmarking/),
    and the harness in `benchmarks/modal_benchmark.py` implements it):

    - **`torch.cuda.synchronize()` inside your timing loop measures the
      launcher, not the kernel** — at decoder scale it inflated this operator
      ~4× and *inverted* a comparison against mmcv's CUDA kernel. Sync once
      outside the loop; for kernels this small, trust `torch.profiler`'s CUDA
      self-time, cross-checked with CUDA-graph replay.
    - **Triton's Python dispatch (~40µs) can exceed the kernel (~10µs).** In
      training the queue hides it; for isolated-latency use, CUDA graphs replay
      the whole thing with zero Python.

    And the honest gap between your rebuild and `src/msda_triton/`: the shipped
    forward autotunes (your Lab 6 robot, wired on), the backward launch config
    came from a three-GPU sweep, and one known optimization is *deliberately
    left un-shipped* — getting Triton to emit 8-wide vectorized atomics
    (`red.add.noftz.v8.bf16`) instead of scalar ones. The README's last caveat
    documents it as open headroom. **You now know enough to go take it.**

    | you built | it became |
    |---|---|
    | bilinear interpolation in loops (Lab 1) | the spec everything answers to |
    | flat offsets + overflow math (Lab 2) | every pointer in your kernels |
    | a sector-counting paper GPU (Lab 3) | the tiling decision, predicted |
    | four small kernels (Lab 4) | fluency in programs/blocks/masks |
    | the forward kernel (Lab 5) | correct on the repo's edge cases |
    | a controlled tiling race + autotuning (Lab 6) | the prediction, confirmed |
    | three hand-derived gradients (Lab 7) | graded by autograd itself |
    | atomics + the backward kernel (Lab 8) | the race, witnessed and fixed |
    | PTX forensics + the precision switch (Lab 9) | a 10× cliff, dodged in 4 lines |
    | autograd + validation + the exam (Lab 10) | **a library** |

    🏁 **Course complete.** Save `MSDAFunction` and `msda_student` into
    `labs/my/lab10.py`, then go read `src/msda_triton/` one last time — it
    should read like your own code now.
    """)
    return


if __name__ == "__main__":
    app.run()
