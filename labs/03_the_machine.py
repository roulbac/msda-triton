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

    import numpy as np

    from labs.common import checks, memsim
    return checks, memsim, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Lab 3 — The machine: a paper GPU you can reason about

    *From NAND to Tetris* has you build the computer before programming it. We
    can't fab an H100, but we can do the next best thing: **build the model of GPU
    memory that determines this operator's performance**, and use it to *predict*
    the design of the kernel you'll write in Lab 5 — before touching hardware.

    The three facts our paper GPU is built on (the full story is
    [course chapter 2](https://roulbac.github.io/msda-triton/course/02-gpu-programming-model/)):

    1. **Threads execute in warps of 32.** A warp issues one instruction across
       all 32 threads at once — including memory loads: 32 addresses hit the
       memory system *simultaneously, as a group*.
    2. **DRAM sells 32-byte sectors, not bytes.** Any touched sector is fetched
       whole. Want 2 bytes from it? You pay for 32.
    3. **Compute is ~free here.** MSDA does ~4 multiply-adds per loaded value; an
       A100 can do ~40 per value at peak bandwidth. Runtime ≈ bytes moved ÷
       bandwidth, full stop.

    So the entire performance question collapses to: **of the bytes each warp's
    sectors deliver, how many did the program actually want?** That ratio is
    *coalescing efficiency*, and you're about to compute it.

    The scaffold (`labs/common/memsim.py`) defines an `AccessPattern`: a list of
    per-warp address arrays plus the `itemsize` each lane wanted. Your job is the
    accounting.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ✏️ Exercise 1 — the sector counter

    `sectors_touched`: for every warp access, count the *distinct* 32-byte sectors
    its addresses land in (`address // 32`; assume aligned elements), and sum over
    all accesses. `efficiency`: useful bytes ÷ delivered bytes.
    """)
    return


@app.cell
def _(checks, memsim, np):
    def sectors_touched(pattern):
        """Total sectors fetched to serve every warp access in the pattern."""
        _ = (memsim.SECTOR_BYTES, np)  # you'll want these
        # ================= YOUR CODE =================
        raise checks.NotDoneYet()
        # =============================================

    def efficiency(pattern):
        """pattern.useful_bytes / bytes actually delivered (1.0 = perfect)."""
        # ================= YOUR CODE ================= (one line)
        raise checks.NotDoneYet()
        # =============================================
    return efficiency, sectors_touched


@app.cell
def _(mo):
    mo.accordion({
        "Hint — one warp": mo.md(
            "```python\nlen(np.unique(np.asarray(warp) // memsim.SECTOR_BYTES))\n```\n"
            "Sum that over `pattern.warps`. For `efficiency`, delivered bytes are "
            "`sectors_touched(pattern) * memsim.SECTOR_BYTES`."
        ),
    })
    return


@app.cell
def _(checks, efficiency, memsim, sectors_touched):
    def _contiguous_is_perfect():
        _p = memsim.contiguous(n=1024, itemsize=4)
        # 32 lanes x 4B = 128B = exactly 4 sectors per warp
        assert sectors_touched(_p) == 1024 * 4 // 32, sectors_touched(_p)
        assert efficiency(_p) == 1.0

    def _stride_burns_bandwidth():
        _p = memsim.strided(stride_elems=16, n=1024, itemsize=4)
        _e = efficiency(_p)
        assert _e <= 0.125 + 1e-9, f"stride-16 fp32 should waste >=7/8, got {_e:.3f}"

    def _random_is_worst():
        _p = memsim.random_gather(n=1024, itemsize=4)
        assert efficiency(_p) <= 0.126, "random ~= one sector per lane"

    checks.run_checks({
        "contiguous access: efficiency 1.0": _contiguous_is_perfect,
        "strided access wastes proportionally": _stride_burns_bandwidth,
        "random gather ≈ 4 useful bytes per 32": _random_is_worst,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🔬 Play: watch a stride destroy your bandwidth

    Same 1024 elements, same warp width — only the stride changes.
    """)
    return


@app.cell
def _(mo):
    stride_slider = mo.ui.slider(1, 32, step=1, value=1, label="stride (elements)")
    stride_slider
    return (stride_slider,)


@app.cell
def _(checks, efficiency, memsim, mo, stride_slider):
    try:
        _p = memsim.strided(stride_slider.value, n=1024, itemsize=4)
        _e = efficiency(_p)
        _bar = "█" * max(1, round(_e * 40))
        mo.md(
            f"stride **{stride_slider.value}** → efficiency **{_e:.3f}** "
            f"`{_bar}`\n\n(2 TB/s of paper DRAM delivers **{2000 * _e:.0f} GB/s** "
            f"of *useful* bytes at this stride.)"
        )
    except (checks.NotDoneYet, NotImplementedError):
        mo.md("*(finish Exercise 1 to enable this)*").callout(kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The main event: predicting the MSDA kernel design

    Now point the simulator at *our* problem. Recall the forward pass: for each
    query, read 4 corner *rows* of `D` consecutive channels each, from
    data-dependent (≈random) pixels. The bytes wanted are fixed. The only choice —
    **the tiling** — is which lanes read what:

    - **Point-parallel** (`memsim.msda_point_parallel`): one *thread* per sample
      point; each walks its row's `D` channels serially. A warp = 32 unrelated
      rows, reading channel `d` of each at every step. Maximum parallelism — it's
      how most people first parallelize this operator.
    - **Query-block** (`memsim.msda_query_block`): one *warp* per row; its 32
      lanes read the row's 32 consecutive fp16 channels in one go. Fewer, fatter
      units of work.

    Same rows, same useful bytes. Run both through **your** counter:
    """)
    return


@app.cell
def _(checks, efficiency, memsim, mo):
    try:
        _qb = memsim.msda_query_block(n_queries=1024, D=32, itemsize=2)
        _pp = memsim.msda_point_parallel(n_queries=1024, D=32, itemsize=2)
        _eq, _ep = efficiency(_qb), efficiency(_pp)
        tiling_ratio = _eq / _ep
        mo.md(
            f"| tiling | efficiency | useful GB/s at 2 TB/s |\n|---|---|---|\n"
            f"| query-block | **{_eq:.3f}** | {2000 * _eq:.0f} |\n"
            f"| point-parallel | **{_ep:.3f}** | {2000 * _ep:.0f} |\n\n"
            f"**Predicted advantage: {tiling_ratio:.1f}× effective bandwidth** "
            f"— from an accounting identity, before writing any GPU code."
        ).callout(kind="success")
    except (checks.NotDoneYet, NotImplementedError):
        tiling_ratio = None
        mo.md("*(finish Exercise 1 to reveal the punchline)*").callout(kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reality check

    The paper measured this on an A100 (§3.2): the point-parallel kernel reached
    **85% occupancy but 5.1% of peak bandwidth**; query-block tiling reached **17%
    occupancy and 36% of peak** — about **7× more useful bytes per second**. Your
    simulator predicts an even bigger ratio (ours says ~16×) because it's an upper
    bound: the real GPU's L2 cache rescues some wasted sectors when threads
    accidentally share them, and neither kernel hits 100% of its model. The
    *direction and order of magnitude* are what the model buys you — enough to
    choose the design before benchmarking it (which you'll do properly in Lab 6).

    Two morals to carry forward:

    - **Occupancy is a proxy, not the objective.** The winning tiling has *fewer*
      threads. What matters is whether each unavoidable scattered access moves the
      widest useful contiguous chunk the problem offers.
    - **You just designed Lab 5's kernel.** One program per block of queries; the
      channel dimension laid across lanes; parameters loaded once per query. That
      is precisely `_msda_forward_kernel` in this repo.

    ### ✏️ Exercise 2 — the roofline sanity check

    If the operator is memory-bound, its best possible time is
    `bytes ÷ bandwidth`. Compute the forward's minimum bytes for the decoder
    preset and its ideal time on an A100 (~2 TB/s).
    """)
    return


@app.cell
def _(checks):
    def forward_min_bytes(B=4, Q=300, M=8, D=32, L=4, K=4, itemsize=2):
        """Algorithmic-minimum DRAM traffic of the MSDA forward, in bytes:
        4 corner rows of D channels per (b,q,m,l,k) sample, plus the sampling
        parameters (2 coords + 1 weight per sample), plus the (B,Q,M,D) output.
        (Ignore the tiny spatial_shapes/starts metadata.)"""
        # ================= YOUR CODE =================
        raise checks.NotDoneYet()
        # =============================================
    return (forward_min_bytes,)


@app.cell
def _(checks, forward_min_bytes, mo):
    def _counts_the_traffic():
        _b = forward_min_bytes()
        _corners = 4 * 300 * 8 * 4 * 4 * 4 * 32 * 2
        _params = 4 * 300 * 8 * 4 * 4 * 3 * 2
        _out = 4 * 300 * 8 * 32 * 2
        assert _b == _corners + _params + _out, (
            f"got {_b:,}, expected {_corners + _params + _out:,} "
            f"(corners {_corners:,} + params {_params:,} + output {_out:,})"
        )

    _result = checks.run_checks({"byte accounting": _counts_the_traffic})
    _result
    return


@app.cell
def _(checks, forward_min_bytes, mo):
    try:
        _b = forward_min_bytes()
        _t = _b / 2e12 * 1e6
        mo.md(
            f"**{_b / 1e6:.0f} MB** must move → ideal A100 time ≈ **{_t:.0f} µs** "
            f"at full 2 TB/s. The repo's measured fp16 forward on A100 is ~18 µs — "
            f"so the shipped kernel runs at a large fraction of the theoretical "
            f"peak, *and* you now know that a kernel 10× slower than your roofline "
            f"number is leaving bandwidth on the table, not compute."
        ).callout(kind="info")
    except (checks.NotDoneYet, NotImplementedError):
        mo.md("*(finish Exercise 2 to see the roofline)*").callout(kind="neutral")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### 🏁 Stage complete — save your work

    Copy `sectors_touched`, `efficiency`, and `forward_min_bytes` into
    **`labs/my/lab03.py`**. You have a machine model and a predicted design.
    Next: **Lab 4 — hello, Triton**, where you finally write code that runs on
    the real machine (or its CPU interpreter), one small kernel at a time.
    """)
    return


if __name__ == "__main__":
    app.run()
