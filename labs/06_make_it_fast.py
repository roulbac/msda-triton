import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
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
    return checks, mo, setup, stage, stage_banner, tl, torch, triton


@app.cell
def _(mo, setup):
    setup.banner(mo)
    return


@app.cell
def _(mo, setup):
    if not setup.HAS_CUDA:
        _msg = mo.md(
            "⛔ **This lab's measurements need a real GPU.** The correctness "
            "exercise below still works under the interpreter, but every timing "
            "cell will refuse to run (timing interpreted Python would teach you "
            "nothing). Cheapest paths to a GPU: any CUDA machine with "
            "`uv sync --group labs`, or the repo's Modal harness "
            "(`MSDA_GPU=L4 uv run modal run benchmarks/modal_benchmark.py`) to "
            "see the shipped kernel's numbers while you read along."
        ).callout(kind="danger")
    else:
        _msg = mo.md("🟢 GPU detected — all cells in this lab are live.").callout(
            kind="success"
        )
    _msg
    return


@app.cell
def _(mo, stage, stage_banner):
    msda_forward, _fwd_src = stage(5, "msda_forward")
    stage_banner(mo, {"msda_forward": _fwd_src})
    return (msda_forward,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lab 6 — Make it fast (and prove it)

    Lab 3 *predicted* the tiling war; Lab 5 built the winner. Today you build the
    loser too, race them on silicon, and then hand the remaining knob to a robot.

    ## 1. The controlled experiment: narrow the loads, keep the math

    Rather than write a from-scratch point-parallel kernel, we'll do what a good
    experiment does — **change one variable**. Take your Lab 5 kernel and add a
    third grid axis that splits the channel dimension into chunks of `CHUNK_D`:

    - `CHUNK_D = 32` (= D): identical to Lab 5. One program per query block reads
      whole 64-byte rows. *(wide)*
    - `CHUNK_D = 2`: 16× more programs (occupancy fans love it!), each reading 4
      bytes per gathered row — Lab 3's point-parallel sector waste, plus every
      per-query parameter re-loaded 16×. *(narrow)*

    ### ✏️ Exercise 1 — the chunked kernel

    Three edits to your Lab 5 kernel — everything else is copy-paste:

    1. new `constexpr` `CHUNK_D` replaces `BLOCK_D`; accumulator is
       `(BLOCK_Q, CHUNK_D)`;
    2. `pid_d = tl.program_id(2)` and `offs_d = pid_d * CHUNK_D +
       tl.arange(0, CHUNK_D)`;
    3. launch grid gains `triton.cdiv(D, CHUNK_D)` as its third element.
    """)
    return


@app.cell
def _(tl, triton):
    @triton.jit
    def msda_forward_chunked_kernel(
        value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, out_ptr,
        Q, S,
        M: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
        BLOCK_Q: tl.constexpr, CHUNK_D: tl.constexpr,
    ):
        # ================= YOUR CODE =================
        # Paste your Lab 5 kernel body and make edits 1-2 from the list above.
        pass
        # =============================================

    def msda_forward_chunked(value, spatial_shapes, level_start_index,
                             sampling_locations, attention_weights,
                             BLOCK_Q=32, CHUNK_D=None):
        import torch as _torch

        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        CHUNK_D = triton.next_power_of_2(D) if CHUNK_D is None else CHUNK_D
        spatial_shapes = spatial_shapes.to(value.device, _torch.int64).contiguous()
        if level_start_index is None:
            _hw = spatial_shapes.prod(-1)
            level_start_index = _torch.cat([_hw.new_zeros(1), _hw.cumsum(0)[:-1]])
        level_start_index = level_start_index.to(
            value.device, _torch.int64
        ).contiguous()
        out = value.new_zeros(B, Q, M, D)
        _grid = (triton.cdiv(Q, BLOCK_Q), B * M, triton.cdiv(D, CHUNK_D))
        msda_forward_chunked_kernel[_grid](
            value.contiguous(), spatial_shapes, level_start_index,
            sampling_locations.contiguous(), attention_weights.contiguous(), out,
            Q, S, M=M, D=D, L=L, K=K, BLOCK_Q=BLOCK_Q, CHUNK_D=CHUNK_D,
        )
        return out.view(B, Q, M * D)
    return (msda_forward_chunked,)


@app.cell
def _(checks, msda_forward_chunked, torch):
    def _correct_at_every_width():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=5, M=2, D=8, shapes=[(5, 6), (3, 3)], K=2,
            dtype=torch.float32, loc_lo=-0.2, loc_hi=1.2,
        )
        _ref = checks.msda_reference(_v, _s, _l, _a)
        for _cd in (2, 4, 8):
            _out = msda_forward_chunked(_v, _s, _st, _l, _a,
                                        BLOCK_Q=4, CHUNK_D=_cd)
            assert not torch.equal(_out, torch.zeros_like(_out)), (
                f"CHUNK_D={_cd}: kernel wrote nothing yet"
            )
            torch.testing.assert_close(_out, _ref, rtol=1e-5, atol=1e-5,
                                       msg=f"CHUNK_D={_cd} disagrees")

    checks.run_checks({
        "chunked kernel is exact at CHUNK_D ∈ {2, 4, 8}": _correct_at_every_width,
    })
    return


@app.cell
def _(mo):
    mo.accordion({
        "Hint — the three edits, exactly": mo.md(
            "```python\npid_d = tl.program_id(2)\n"
            "offs_d = pid_d * CHUNK_D + tl.arange(0, CHUNK_D)\n"
            "...\nacc = tl.zeros((BLOCK_Q, CHUNK_D), dtype=tl.float32)\n```\n"
            "Everything else — `val_base`, masks, corner loads, the store — "
            "already uses `offs_d`/`mask_d` and needs no change. That's the point: "
            "the *tiling* changed, the *math* didn't."
        ),
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The race (GPU)

    Decoder-preset-sized inputs, fp16. For each width we time with
    `triton.testing.do_bench` — which handles warmup (important: the first call
    JIT-compiles!) and returns robust medians — and convert to *effective
    bandwidth* using Lab 3's `forward_min_bytes` accounting: the bytes the
    operator *needs*, divided by the time the kernel *took*.
    """)
    return


@app.cell
def _(checks, mo, msda_forward_chunked, setup, torch, triton):
    if setup.HAS_CUDA:
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=4, Q=300, M=8, D=32, shapes=[(100, 167), (50, 84), (25, 42), (13, 21)],
            K=4, dtype=torch.float16, device="cuda",
        )
        _min_bytes = (4 * 300 * 8 * 4 * 4 * 4 * 32 * 2
                      + 4 * 300 * 8 * 4 * 4 * 3 * 2 + 4 * 300 * 8 * 32 * 2)
        sweep_rows = []
        for _cd in (2, 4, 8, 16, 32):
            _ms = triton.testing.do_bench(
                lambda: msda_forward_chunked(_v, _s, _st, _l, _a, CHUNK_D=_cd),
                warmup=25, rep=100,
            )
            sweep_rows.append((_cd, _ms * 1e3, _min_bytes / (_ms * 1e-3) / 1e9))
        _tbl = "\n".join(
            f"| {c} | {us:.1f} | {bw:.0f} |" for c, us, bw in sweep_rows
        )
        _speedup = sweep_rows[0][1] / sweep_rows[-1][1]
        mo.md(
            f"| CHUNK_D | µs | effective GB/s |\n|---|---|---|\n{_tbl}\n\n"
            f"**Wide over narrow: {_speedup:.1f}× on your GPU.** Lab 3 predicted "
            f"the ceiling (~16×); the paper measured ~7× on A100; your number "
            f"sits wherever your GPU's L2 cache and clocks put it. All three "
            f"agree on the only decision that matters: *load wide.*"
        )
    else:
        sweep_rows = None
        mo.md("*(GPU required — see the callout at the top.)*").callout(
            kind="neutral"
        )
    return (sweep_rows,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Occupancy is a proxy — say it with numbers

    The narrow configuration launches **16× more programs** than the wide one.
    Every occupancy-centric heuristic ranks it better; your table just showed it
    losing by roughly an order of magnitude. This is the paper's §3.2 finding —
    85% occupancy / 5.1% of peak bandwidth vs 17% / 36% — reproduced by an
    experiment you built. When someone (or a profiler) tells you "increase
    occupancy," translate it as: *"keep the memory system busy"* — and check
    whether your accesses are wide before making your threads small.

    ## 4. Hand the last knob to a robot: autotuning

    `BLOCK_Q` (and `num_warps` — how many warps execute each program) is left.
    The honest answer to "which is best?" is *it depends on the GPU and the
    shapes*, so Triton ships a measuring robot:

    ```python
    tuned_kernel = triton.autotune(
        configs=[triton.Config({"BLOCK_Q": bq}, num_warps=nw)
                 for bq in (16, 32, 64) for nw in (2, 4)],
        key=["Q", "M", "D", "L", "K"],        # re-tune when these change
    )(msda_forward_kernel)
    ```

    First call with new `key` values: it times all six configs, caches the winner.
    That decorator (on this exact config list) is the *only* difference between
    your Lab 5 kernel and the one shipped in `src/msda_triton/kernels.py`.
    """)
    return


@app.cell
def _(checks, mo, setup, stage, torch, triton):
    if setup.HAS_CUDA:
        _kernel, _src = stage(5, "msda_forward_kernel")
        tuned_kernel = triton.autotune(
            configs=[triton.Config({"BLOCK_Q": bq}, num_warps=nw)
                     for bq in (16, 32, 64) for nw in (2, 4)],
            key=["Q", "M", "D", "L", "K"],
        )(_kernel)

        _v, _s, _st, _l, _a = checks.make_inputs(
            B=4, Q=300, M=8, D=32, shapes=[(100, 167), (50, 84), (25, 42), (13, 21)],
            K=4, dtype=torch.float16, device="cuda",
        )
        _B, _S, _M, _D = _v.shape
        _Q = _l.shape[1]
        _out = _v.new_zeros(_B, _Q, _M, _D)
        _grid = lambda meta: (triton.cdiv(_Q, meta["BLOCK_Q"]), _B * _M)
        for _ in range(10):  # triggers tuning on the first call
            tuned_kernel[_grid](_v, _s, _st, _l, _a, _out, _Q, _S,
                                M=_M, D=_D, L=4, K=4,
                                BLOCK_D=triton.next_power_of_2(_D))
        mo.md(
            f"Autotuner verdict on **{torch.cuda.get_device_name()}** "
            f"(kernel source: {_src}):\n\n```\n{tuned_kernel.best_config}\n```\n\n"
            f"Compare with the shipped forward kernel's tuned config on the "
            f"same shapes — they should agree; it uses this exact search space."
        )
    else:
        tuned_kernel = None
        mo.md("*(GPU required.)*").callout(kind="neutral")
    return (tuned_kernel,)


@app.cell
def _(mo):
    mo.md(r"""
    ### ⚠️ Why the *backward* kernel must not be autotuned this way

    File this away for Lab 8: the autotuner **runs the kernel repeatedly to time
    it**. The forward is pure — run it 100×, same result. The backward
    *accumulates into* `grad_value` with atomic adds: every timing run would add
    another copy of the gradient. (Triton has `reset_to_zero` hooks for this;
    this repo instead fixes the backward launch config from an offline sweep —
    `BWD_BLOCK_Q = 16`, 4 warps, chosen on L40S/A100/H100. See
    `benchmarks/modal_variants.py::bwd_sweep` to rerun that sweep yourself.)
    **Kernels with side effects don't compose naively with tooling that reruns
    them.**

    ---
    ### 🏁 Stage complete — save your work

    Copy `msda_forward_chunked` (kernel + wrapper) into **`labs/my/lab06.py`**
    if you want it around; the artifact that carries forward is the *lesson*.
    The forward story is complete: correct (Lab 5), fast, and measured honestly
    (this lab). Next: **Lab 7 — the other direction** — training needs
    gradients, and gradients turn our gathers into scatters.
    """)
    return


if __name__ == "__main__":
    app.run()
