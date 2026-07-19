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

    triton, tl = setup.import_triton()
    return checks, mo, setup, tl, torch, triton


@app.cell(hide_code=True)
def _(mo, setup):
    setup.banner(mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4 — Hello, Triton

    Time to program the machine from Lab 3. [Triton](https://triton-lang.org) is a
    Python-embedded GPU language: you decorate a function with `@triton.jit`,
    launch a **grid of program instances**, and each program operates on whole
    **blocks** of values (little tensors) instead of single scalars. The compiler
    maps your block operations onto warps and picks the wide/coalesced
    instructions — *which* bytes each program touches remains 100% your decision,
    which is why Lab 3 mattered.

    You'll climb four rungs, each adding exactly one idea:

    | rung | kernel | new idea |
    |---|---|---|
    | 1 | vector add *(worked example)* | programs, blocks, masks |
    | 2 | 2-D tile copy | 2-D blocks via broadcasting, two masks |
    | 3 | gather | **data-dependent addresses** |
    | 4 | bilinear sample | everything above + Lab 1's math = a mini-MSDA |

    *(Deeper background: [course chapter 3](https://roulbac.github.io/msda-triton/course/03-triton-basics/).
    No GPU? Everything here runs on the CPU interpreter — see the banner above.)*

    ## Rung 1 (worked example): vector add

    Read every line; every later kernel is this skeleton with more inside.
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr,           # tensors arrive as POINTERS
                   n,                                # runtime scalar
                   BLOCK: tl.constexpr):             # compile-time constant
        pid = tl.program_id(0)                       # which program am I?
        offs = pid * BLOCK + tl.arange(0, BLOCK)     # my slice of the problem
        mask = offs < n                              # guard the ragged edge
        x = tl.load(x_ptr + offs, mask=mask)         # BLOCK loads in one op
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)   # masked lanes write nothing

    def add(x, y, BLOCK=256):
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, BLOCK),)              # ceil(n / BLOCK) programs
        add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
        return out
    return (add,)


@app.cell(hide_code=True)
def _(add, checks, torch):
    def _adds():
        _x, _y = torch.randn(1000), torch.randn(1000)
        torch.testing.assert_close(add(_x, _y), _x + _y)

    checks.run_checks({"worked example runs on your machine": _adds})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Five things to internalize before rung 2:

    1. **Pointers, not tensors** — `x_ptr + offs` is Lab 2's flat-index
       arithmetic, verbatim. There are no shapes inside a kernel.
    2. **`tl.arange(0, BLOCK)` needs a power-of-2 `BLOCK`**, so real sizes are
       handled by the `mask`, not by the block size.
    3. **Masked loads take `other=`** — the value delivered to switched-off lanes
       (default 0). Remember Lab 2's clamp-then-zero gather trick? `mask=`/`other=`
       is its native form — and it will implement zeros-padding for free.
    4. **`tl.constexpr` args specialize the compilation** — a new binary per
       distinct value, letting the compiler unroll and allocate registers exactly.
    5. **The grid is just a tuple** (up to 3-D) of how many programs to launch;
       each finds its slice from `tl.program_id(axis)`.

    ### ✏️ Rung 2 — 2-D tile copy

    Copy an `(R, C)` matrix through a kernel where each program owns a
    `(BLOCK_R, BLOCK_C)` tile. The new move is building a 2-D block of flat
    offsets from two 1-D ranges by broadcasting — the idiom at the heart of the
    MSDA kernel's corner loads:

    ```python
    ptrs = offs_r[:, None] * C + offs_c[None, :]      # (BLOCK_R, BLOCK_C)
    mask = mask_r[:, None] & mask_c[None, :]
    ```
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def copy2d_kernel(x_ptr, out_ptr, R, C,
                      BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        # ================= YOUR CODE =================
        # offs_r/offs_c from the two pids; masks; 2-D ptrs; load; store.
        pass
        # =============================================

    def copy2d(x, BLOCK_R=16, BLOCK_C=16):
        R, C = x.shape
        out = torch.zeros_like(x)  # zeros: a do-nothing kernel is caught below
        _grid = (triton.cdiv(R, BLOCK_R), triton.cdiv(C, BLOCK_C))
        copy2d_kernel[_grid](x, out, R, C, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C)
        return out
    return (copy2d,)


@app.cell(hide_code=True)
def _(checks, copy2d, torch):
    def _copies_exactly():
        _x = torch.randn(37, 21)  # deliberately not multiples of 16
        _y = copy2d(_x)
        assert not torch.equal(_y, torch.zeros_like(_x)), "kernel wrote nothing yet"
        torch.testing.assert_close(_y, _x)

    checks.run_checks({"ragged 37×21 copy is exact": _copies_exactly})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint — full body": mo.md(
            "```python\noffs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)\n"
            "offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)\n"
            "mask = (offs_r < R)[:, None] & (offs_c < C)[None, :]\n"
            "ptrs = offs_r[:, None] * C + offs_c[None, :]\n"
            "tile = tl.load(x_ptr + ptrs, mask=mask, other=0.0)\n"
            "tl.store(out_ptr + ptrs, tile, mask=mask)\n```"
        ),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ✏️ Rung 3 — gather: addresses that come from data

    Everything so far computed addresses from `program_id` — knowable at launch.
    MSDA's addresses come from the *network's output*. The primitive:
    `out[i] = src[idx[i]]` — load the indices, then load *through* them.
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        # ================= YOUR CODE =================
        # 1. idx = tl.load(idx_ptr + offs, ...)   (other=0 keeps masked lanes legal)
        # 2. val = tl.load(src_ptr + idx, ...)    <- the data-dependent load
        # 3. store to out_ptr + offs
        pass
        # =============================================

    def gather(src, idx, BLOCK=256):
        out = torch.zeros(idx.shape, dtype=src.dtype, device=src.device)
        n = idx.numel()
        gather_kernel[(triton.cdiv(n, BLOCK),)](src, idx, out, n, BLOCK=BLOCK)
        return out
    return (gather,)


@app.cell(hide_code=True)
def _(checks, gather, torch):
    def _gathers():
        _src = torch.randn(500)
        _idx = torch.randint(0, 500, (123,))
        _out = gather(_src, _idx)
        assert not torch.equal(_out, torch.zeros(123)), "kernel wrote nothing yet"
        torch.testing.assert_close(_out, _src[_idx])

    checks.run_checks({"out[i] == src[idx[i]]": _gathers})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pause on what you just wrote: `tl.load(src_ptr + idx)` where `idx` was itself
    loaded. *This is the memory pattern Lab 3 said is expensive* — the hardware
    can't coalesce what it can't predict. The fix isn't avoiding the gather (we
    can't; it's the operator); it's making each gathered access **wide**, which is
    rung 4.

    ### ✏️ Rung 4 — bilinear sampling: mini-MSDA

    One level, one point per query, no heads, no attention weights: given a
    flattened `(H·W, D)` image and `(Q, 2)` normalized locations, produce the
    `(Q, D)` bilinear samples. Each program owns `BLOCK_Q` queries × all `D`
    channels — Lab 3's query-block tiling: the *row* addresses are gathered, but
    each row read is `D` consecutive channels (rung 2's broadcast idiom).

    Everything you need, you've already written once:

    | piece | where you built it |
    |---|---|
    | `x·W − 0.5`, floor, fractional parts, 4 corner weights | Lab 1 |
    | flat offset of a pixel row: `(y·W + x) · D` | Lab 2 |
    | validity per corner → masked loads with `other=0.0` | Lab 2's clamp trick, rung 1's masks |
    | 2-D pointer block `rows[:, None] + offs_d[None, :]` | rung 2 |
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def bilinear_kernel(value_ptr,   # (H*W, D) contiguous
                        loc_ptr,     # (Q, 2) normalized (x, y)
                        out_ptr,     # (Q, D)
                        Q, H, W,
                        D: tl.constexpr,
                        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr):
        pid = tl.program_id(0)
        offs_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, BLOCK_D)
        mask_q = offs_q < Q
        mask_d = offs_d < D
        # ================= YOUR CODE =================
        # 1. load loc_x (loc_ptr + 2*offs_q) and loc_y (+1), mask=mask_q
        # 2. x = loc_x * W - 0.5, y = loc_y * H - 0.5  (use .to(tl.float32))
        # 3. x0f = tl.math.floor(x) ... lx, ly; x0/y0 via .to(tl.int32); x1, y1
        # 4. per-corner validity: vx0 = mask_q & (x0 >= 0) & (x0 < W), etc.
        # 5. four loads:  base = value_ptr + offs_d[None, :]
        #      v00 = tl.load(base + ((y0*W + x0) * D)[:, None],
        #                    mask=(vy0 & vx0)[:, None] & mask_d[None, :], other=0.0)
        # 6. blend with the four weights (broadcast [:, None]), store like rung 2
        pass
        # =============================================

    def bilinear(value_hw_d, loc, BLOCK_Q=32):
        H, W, D = value_hw_d.shape
        Q = loc.shape[0]
        out = torch.zeros(Q, D, dtype=value_hw_d.dtype, device=value_hw_d.device)
        bilinear_kernel[(triton.cdiv(Q, BLOCK_Q),)](
            value_hw_d.reshape(H * W, D).contiguous(), loc.contiguous(), out,
            Q, H, W, D=D, BLOCK_Q=BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
        )
        return out
    return (bilinear,)


@app.cell(hide_code=True)
def _(bilinear, checks, torch):
    def _matches_reference():
        _v, _shapes, _st, _loc, _attn = checks.make_inputs(
            B=1, Q=6, M=1, D=4, shapes=[(5, 7)], K=1, loc_lo=-0.2, loc_hi=1.2
        )
        _img = _v[0, :, 0].view(5, 7, 4)
        _out = bilinear(_img, _loc[0, :, 0, 0, 0], BLOCK_Q=8)
        assert not torch.equal(_out, torch.zeros_like(_out)), "kernel wrote nothing yet"
        _ref = checks.msda_reference(_v, _shapes, _loc, torch.ones_like(_attn))
        torch.testing.assert_close(
            _out, _ref[0].view(6, 1, 4)[:, 0], rtol=1e-5, atol=1e-5
        )

    checks.run_checks({
        "bilinear kernel matches the reference (incl. out-of-bounds)":
            _matches_reference,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint — corner loads, concretely": mo.md(
            "```python\nbase = value_ptr + offs_d[None, :]\n"
            "v00 = tl.load(base + ((y0 * W + x0) * D)[:, None],\n"
            "              mask=(vy0 & vx0)[:, None] & mask_d[None, :],\n"
            "              other=0.0).to(tl.float32)\n```\n"
            "and the blend:\n```python\nout = (v00 * ((1.0-lx)*(1.0-ly))[:, None]\n"
            "       + v01 * (lx*(1.0-ly))[:, None]\n"
            "       + v10 * ((1.0-lx)*ly)[:, None]\n"
            "       + v11 * (lx*ly)[:, None])\n"
            "tl.store(out_ptr + offs_q[:, None] * D + offs_d[None, :],\n"
            "         out.to(out_ptr.dtype.element_ty),\n"
            "         mask=mask_q[:, None] & mask_d[None, :])\n```"
        ),
        "Why `.to(tl.float32)` on the loads?": mo.md(
            "Kernels compute in registers; casting loads up to fp32 and casting "
            "once at the store gives fp32 *accumulation* regardless of the "
            "storage dtype — half-precision I/O traffic, full-precision math. "
            "Lab 5 makes this pattern official with a `tl.zeros(..., dtype="
            "tl.float32)` accumulator."
        ),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 🏁 Stage complete — save your work

    Copy `copy2d`/`gather`/`bilinear` (kernels + wrappers) into
    **`labs/my/lab04.py`**. Rung 4 *is* the inner loop of the real forward kernel
    — Lab 5 wraps it in the `(batch, head)` grid, the level/point loops, and the
    attention weights, and suddenly you'll be holding the whole thing.
    """)
    return


if __name__ == "__main__":
    app.run()
