import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    import sys

    # AOT compilation needs the real compiler, not the CPU interpreter — set
    # BEFORE anything imports triton (this is the first cell to run).
    os.environ["TRITON_INTERPRET"] = "0"

    import marimo as mo

    _repo = mo.notebook_dir().parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    import torch

    from labs.common import checks, setup
    from labs.common.loader import stage, stage_banner

    triton, tl = setup.import_triton()
    return checks, mo, setup, stage, stage_banner, torch, triton


@app.cell(hide_code=True)
def _(mo, setup):
    setup.banner(mo)
    return


@app.cell(hide_code=True)
def _(mo, stage, stage_banner):
    msda_backward_kernel, _k_src = stage(8, "msda_backward_kernel")
    stage_banner(mo, {"msda_backward_kernel": _k_src})
    return (msda_backward_kernel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 9 — Hardware detective: PTX and the precision switch

    A true story, and the best plot twist in this repo. Train a bf16 model with
    the backward you just wrote: on an H100 it's fast; on an A100, the *identical
    source code* runs the backward ~10× slower. Nothing in your kernel changed.
    What did?

    **The instruction it compiles to.** `tl.atomic_add` is one line of Triton,
    but which machine instruction it lowers to depends on the *(dtype, GPU
    generation)* pair:

    | `grad_value` dtype | A100 (SM 8.0) | H100 (SM 9.0) |
    |---|---|---|
    | fp32 | ✅ native atomic add | ✅ native |
    | fp16 | ✅ native (since SM 6.0!) | ✅ native |
    | **bf16** | ❌ **compare-and-swap emulation** | ✅ native (new in SM 9.0) |

    BF16 arrived late to CUDA; its atomic add only became a hardware instruction
    with SM 9.0. On older GPUs the compiler silently emits a **CAS retry loop**:
    read the word, compute the new value, atomically swap *if unchanged*, retry
    on failure — a full round trip with a loop around it, versus fire-and-forget.
    Under a million-atomic stream, ~10× slower (paper, Table 2).

    Today you (1) *see* this in compiled code without owning either GPU, and
    (2) build the four-line fix the repo ships.

    ## Compilers don't need GPUs

    Triton compiles to **PTX** (NVIDIA's portable assembly) for an explicit
    target — `triton.compile(..., target=GPUTarget("cuda", 80, 32))` runs fine
    on a laptop. This is how the repo's CI type-checks every kernel variant and
    prints its atomic lowering *on every PR, with no GPU* — see
    `scripts/compile_kernels_check.py`, which this lab is a guided tour of.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ✏️ Exercise 1 — the PTX detector

    Given a PTX listing (a big string), find every atomic instruction: they look
    like `atom.global.gpu.relaxed.add.f32`, `red.global.add.f32`,
    `atom.relaxed.gpu.global.cas.b32`, ... i.e. `atom.` or `red.` followed by
    dot-separated qualifiers. Return them deduplicated and sorted.
    """)
    return


@app.cell
def _(checks):
    import re

    def find_atomics(ptx: str) -> list:
        """Sorted, distinct atomic/reduction instructions in a PTX listing."""
        _ = re  # regex is the intended tool: r"\b(?:red|atom)\.[\w.:]+"
        # ================= YOUR CODE =================
        raise checks.NotDoneYet()
        # =============================================
    return (find_atomics,)


@app.cell(hide_code=True)
def _(checks, find_atomics):
    _FAKE_PTX = """
    .visible .entry kern(
        ld.global.f32 %f1, [%rd1];
        atom.global.gpu.relaxed.add.f32 %f2, [%rd2], %f1;
    $L__BB0_2:
        atom.relaxed.gpu.global.cas.b32 %r5, [%rd3], %r3, %r4;
        setp.ne.s32 %p2, %r5, %r3;
        @%p2 bra $L__BB0_2;          // <- the retry loop, in the flesh
        red.global.add.f32 [%rd4], %f3;
        atom.global.gpu.relaxed.add.f32 %f9, [%rd9], %f8;
    """

    def _finds_all_three():
        _got = find_atomics(_FAKE_PTX)
        assert _got == [
            "atom.global.gpu.relaxed.add.f32",
            "atom.relaxed.gpu.global.cas.b32",
            "red.global.add.f32",
        ], f"got {_got}"

    checks.run_checks({"detector finds add, cas, and red forms once each":
                       _finds_all_three})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The experiment: your backward kernel, four ways

    We AOT-compile **your Lab 8 kernel** for SM 8.0 and SM 9.0, with
    `grad_value` in native bf16 vs an fp32 buffer, and run your detector on each
    PTX. (The compile plumbing — building the signature dict and calling
    `triton.compile` — is given; it's `scripts/compile_kernels_check.py`
    machinery, worth reading once but not worth retyping.)
    """)
    return


@app.cell(hide_code=True)
def _(find_atomics, mo, msda_backward_kernel):
    from labs.solutions.lab09 import compile_backward

    try:
        _rows = []
        for _cc in (80, 90):
            for _gv in ("bf16", "fp32"):
                _compiled = compile_backward(msda_backward_kernel, "bf16", _gv, _cc)
                _atoms = find_atomics(_compiled.asm["ptx"])
                _verdict = ("🐌 CAS emulation" if any(".cas." in a for a in _atoms)
                            else "⚡ native add")
                _rows.append(
                    f"| SM {_cc // 10}.{_cc % 10} | {_gv} | "
                    f"`{', '.join(_atoms)}` | {_verdict} |"
                )
        atomic_table = (
            "| target | grad_value dtype | atomic instructions | verdict |\n"
            "|---|---|---|---|\n" + "\n".join(_rows)
        )
        mo.md(
            f"{atomic_table}\n\n**There's the 10× cliff, in writing**: same "
            f"kernel source, and only the (SM 8.0, bf16) cell compiles to a "
            f"`cas` loop. Note the fp32-buffer rows: native everywhere."
        ).callout(kind="success")
    except (Exception,) as _e:  # noqa: BLE001 — surface partial progress kindly
        atomic_table = None
        mo.md(
            f"*(finish Exercise 1 to run the four-way compile — "
            f"currently: `{type(_e).__name__}: {_e}`)*"
        ).callout(kind="neutral")
    return (atomic_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercise 2 — the switch

    The fix writes itself from the table: **if the value dtype's atomic would be
    emulated on this GPU, accumulate `grad_value` in an fp32 scratch buffer
    instead, and downcast once at the end.** Your Lab 8 wrapper already accepts
    `fp32_grad_accum` and your kernel already casts to
    `grad_value_ptr.dtype.element_ty` — the caller picks the instruction by
    picking the buffer's dtype. All that's missing is the decision rule:

    - bf16 **and** SM < 9.0 → `True` (dodge the CAS loop; bonus: fp32
      accumulation is also more accurate);
    - fp16 → `False` *always*: its atomics have been native since SM 6.0, and
      routing fp16 through the fp32 buffer doubles the atomic bytes + adds a
      downcast pass — measured 2–4× *slower* on Ampere/Ada. **A fix applied
      wider than its cause becomes a regression.**
    - fp32 → `False` (it already is fp32).
    """)
    return


@app.cell
def _(checks, torch):
    def use_fp32_grad_accum(dtype, capability):
        """dtype: torch dtype of value; capability: (major, minor).
        True iff grad_value should be accumulated in an fp32 scratch buffer."""
        _ = torch
        # ================= YOUR CODE ================= (one or two lines)
        raise checks.NotDoneYet()
        # =============================================
    return (use_fp32_grad_accum,)


@app.cell(hide_code=True)
def _(checks, torch, use_fp32_grad_accum):
    def _truth_table():
        _t = {
            (torch.bfloat16, (8, 0)): True,   # A100: dodge the CAS loop
            (torch.bfloat16, (8, 9)): True,   # Ada too (shipped rule: major < 9)
            (torch.bfloat16, (9, 0)): False,  # H100: native bf16 atomics
            (torch.float16, (8, 0)): False,   # fp16: native since SM 6.0
            (torch.float16, (9, 0)): False,
            (torch.float32, (8, 0)): False,   # already fp32
        }
        for (_dt, _cc), _want in _t.items():
            _got = use_fp32_grad_accum(_dt, _cc)
            assert _got == _want, f"({_dt}, SM {_cc}): got {_got}, want {_want}"

    checks.run_checks({"decision rule matches ops.py's shipped switch":
                       _truth_table})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *(One genuinely fuzzy row: SM 8.9 — Ada/L40S. The native bf16 atomic-add
    instruction is documented as PTX ISA 7.8 / SM 9.0, so the shipped rule
    `major < 9` routes Ada through the fp32 buffer; yet the repo's L40S
    benchmarks show bf16 behaving healthily there. Hardware boundaries are
    messier than version tables — which is exactly why the switch is
    user-overridable (`fp32_grad_accum=True/False`) instead of hard-coded.)*

    The complete shipped decision, for reference — this is
    `src/msda_triton/ops.py::_use_fp32_grad_accum`, all four lines of it:

    ```python
    if value.dtype != torch.bfloat16:
        return False
    return torch.cuda.get_device_capability(value.device)[0] < 9
    ```

    A 10× hardware cliff, dodged by choosing a buffer's dtype. The transferable
    lesson: **when identical code is mysteriously slow on one GPU generation,
    read what it *lowers to*, not what it says** — and when a hardware fast path
    is missing, restructuring around it can cost four lines.

    ---
    ### 🏁 Stage complete — save your work

    Copy `find_atomics` and `use_fp32_grad_accum` into **`labs/my/lab09.py`**.
    Every piece now exists: forward kernel, backward kernel, and the
    hardware-aware dispatch rule. One lab remains: bolt it into PyTorch's
    autograd, validate inputs like a grown-up library, and face the repo's own
    test suite. Next: **Lab 10 — ship it.**
    """)
    return


if __name__ == "__main__":
    app.run()
