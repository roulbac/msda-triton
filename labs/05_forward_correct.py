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
    return checks, mo, setup, stage, stage_banner, tl, torch, triton


@app.cell(hide_code=True)
def _(mo, setup):
    setup.banner(mo)
    return


@app.cell(hide_code=True)
def _(mo, stage, stage_banner):
    msda_naive, _naive_src = stage(1, "msda_naive_student")
    stage_banner(mo, {"msda_naive_student": _naive_src})
    return (msda_naive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 5 — The forward kernel, correct

    This is the summit of the forward half of the course: the **complete MSDA
    forward kernel**, with the tiling your Lab 3 simulator chose. Rung 4 of Lab 4
    was one level and one point; the real thing adds:

    - a **2-D grid**: axis 0 tiles queries into `BLOCK_Q`-blocks, axis 1
      enumerates `(batch, head)` pairs — `grid = (cdiv(Q, BLOCK_Q), B * M)`;
    - loops over levels and points, written `tl.static_range(L)` / `(K)`. Because
      `L`, `K` are `constexpr`, these unroll **at compile time**: the kernel
      becomes straight-line code and all 16 iterations' loads can be in flight
      at once (memory-level parallelism — Lab 3's "keep the memory system busy");
    - an explicit **fp32 accumulator** (`tl.zeros((BLOCK_Q, BLOCK_D), dtype=
      tl.float32)`) so fp16/bf16 inputs still sum 128 terms at full precision;
    - **int64 index math** — Lab 2's overflow exercise, now enforced with
      `.to(tl.int64)` at the two places indices are born;
    - **disjoint pointers** for locations and weights: the standard PyTorch
      fallback concatenates them into one `(B, Q, L, K, 3)` buffer (an extra
      round-trip through DRAM); we read the two tensors where they already live.

    The setup (grid decode, offsets, masks) and the closing store are **given** —
    they're bookkeeping you've written twice already. You fill the physics: the
    parameter loads, the bilinear setup, the four corner gathers, the blend.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ✏️ The kernel

    Layouts (all contiguous — the wrapper guarantees it):
    `value (B,S,M,D)` · `shapes (L,2) i64` · `starts (L,) i64` ·
    `loc (B,Q,M,L,K,2)` · `attn (B,Q,M,L,K)` · `out (B,Q,M,D)`.

    Offsets you already know: `p = pq·(L·K) + l·K + k` slots into `loc`/`attn`
    (with `loc_x` at `2p`, `loc_y` at `2p+1`); a corner's channel row starts at
    `((b·S + start + y·W + x)·M + m)·D` — and `val_base` below has already
    absorbed the `b`, `m`, `d` parts, so you only add `((row + x) · M·D)`.
    """)
    return


@app.cell
def _(tl, triton):
    @triton.jit
    def msda_forward_kernel(
        value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, out_ptr,
        Q, S,
        M: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # ---- given: who am I, and where is my data ----
        pid_q = tl.program_id(0)
        pid_bm = tl.program_id(1)
        b = (pid_bm // M).to(tl.int64)
        m = pid_bm % M

        offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask_q = offs_q < Q
        mask_d = offs_d < D

        pq = (b * Q + offs_q) * M + m                      # into loc/attn
        val_base = value_ptr + (b * S * M + m) * D + offs_d[None, :]

        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for l in tl.static_range(L):
            H = tl.load(shapes_ptr + 2 * l)
            W = tl.load(shapes_ptr + 2 * l + 1)
            start = tl.load(starts_ptr + l)
            fh = H.to(tl.float32)
            fw = W.to(tl.float32)
            for k in tl.static_range(K):
                p = pq * (L * K) + l * K + k
                # ================= YOUR CODE =================
                # 1. load loc_x, loc_y, attn (mask=mask_q, other=0.0, ->fp32)
                # 2. bilinear setup: x = loc_x*fw - 0.5, y = loc_y*fh - 0.5,
                #    floor, lx/ly, x0/y0 via .to(tl.int64), x1/y1
                # 3. validities  (fold mask_q into the x ones):
                #    vx0 = mask_q & (x0 >= 0) & (x0 < W)   ... vy0, vx1, vy1
                # 4. row0 = start + y0 * W;  row1 = row0 + W
                #    four corner gathers from val_base + ((row + x)*(M*D))[:, None]
                #    with mask=(vy & vx)[:, None] & mask_d[None, :], other=0.0
                # 5. acc += the four corners, each times
                #    (attn * bilinear_weight)[:, None]
                pass
                # =============================================

        # ---- given: one coalesced tile store ----
        out_ptrs = out_ptr + pq[:, None] * D + offs_d[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
                 mask=mask_q[:, None] & mask_d[None, :])

    def msda_forward(value, spatial_shapes, level_start_index,
                     sampling_locations, attention_weights, BLOCK_Q=32):
        import torch as _torch

        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        spatial_shapes = spatial_shapes.to(value.device, _torch.int64).contiguous()
        if level_start_index is None:
            _hw = spatial_shapes.prod(-1)
            level_start_index = _torch.cat([_hw.new_zeros(1), _hw.cumsum(0)[:-1]])
        level_start_index = level_start_index.to(
            value.device, _torch.int64
        ).contiguous()
        out = value.new_zeros(B, Q, M, D)
        _grid = (triton.cdiv(Q, BLOCK_Q), B * M)
        msda_forward_kernel[_grid](
            value.contiguous(), spatial_shapes, level_start_index,
            sampling_locations.contiguous(), attention_weights.contiguous(), out,
            Q, S, M=M, D=D, L=L, K=K,
            BLOCK_Q=BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
        )
        return out.view(B, Q, M * D)
    return (msda_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint 1 — parameter loads": mo.md(
            "```python\nloc_x = tl.load(loc_ptr + 2 * p, mask=mask_q, other=0.0).to(tl.float32)\n"
            "loc_y = tl.load(loc_ptr + 2 * p + 1, mask=mask_q, other=0.0).to(tl.float32)\n"
            "attn = tl.load(attn_ptr + p, mask=mask_q, other=0.0).to(tl.float32)\n```\n"
            "Note the shape: for fixed `(l, k)`, `p` varies only through `offs_q` — "
            "consecutive queries read (nearly) consecutive addresses. Coalesced, "
            "loaded once per query. Point-parallel tilings re-fetch these `D` times."
        ),
        "Hint 2 — one corner gather": mo.md(
            "```python\nrow0 = start + y0 * W\n"
            "v00 = tl.load(val_base + ((row0 + x0) * (M * D))[:, None],\n"
            "              mask=(vy0 & vx0)[:, None] & mask_d[None, :],\n"
            "              other=0.0).to(tl.float32)\n```"
        ),
        "Hint 3 — the blend": mo.md(
            "```python\nacc += (\n    v00 * (attn * (1.0 - lx) * (1.0 - ly))[:, None]\n"
            "    + v01 * (attn * lx * (1.0 - ly))[:, None]\n"
            "    + v10 * (attn * (1.0 - lx) * ly)[:, None]\n"
            "    + v11 * (attn * lx * ly)[:, None]\n)\n```"
        ),
        "Completely stuck?": mo.md(
            "`labs/solutions/lab05.py` is the full kernel, and "
            "`src/msda_triton/kernels.py` is the production version of the same "
            "thing (plus autotuning). Diff your attempt against them line by line."
        ),
    })
    return


@app.cell(hide_code=True)
def _(checks, mo, msda_forward, msda_naive, setup, torch):
    _cases = [
        dict(B=2, Q=5, M=2, D=4, shapes=[(5, 7), (3, 4)], K=3),
        dict(B=1, Q=7, M=1, D=3, shapes=[(4, 6)], K=1),                # non-pow2 D
        dict(B=1, Q=4, M=2, D=5, shapes=[(6, 5), (3, 3)], K=2,
             loc_lo=-0.4, loc_hi=1.4),                                  # OOB locs
    ] if setup.INTERPRETED else None  # tiny under the interpreter; defaults on GPU

    def _matches_reference():
        checks.assert_msda_matches(
            lambda v, s, st, l, a: msda_forward(v, s, st, l, a, BLOCK_Q=4),
            cases=_cases, dtype=torch.float32, takes_starts=True,
        )

    def _matches_your_lab1():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=3, M=2, D=4, shapes=[(5, 6)], K=2, dtype=torch.float32
        )
        _mine = msda_forward(_v, _s, _st, _l, _a, BLOCK_Q=4)
        assert not torch.equal(_mine, torch.zeros_like(_mine)), (
            "kernel wrote only zeros — the YOUR CODE block is still empty?"
        )
        torch.testing.assert_close(
            _mine, msda_naive(_v, _s, _l, _a), rtol=1e-5, atol=1e-5
        )

    def _grid_corner_locations():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=4, M=1, D=4, shapes=[(4, 6)], K=2, dtype=torch.float32
        )
        _grid_vals = torch.tensor([0.0, 0.5, 1.0])
        _l = _grid_vals[torch.randint(0, 3, _l.shape,
                                      generator=torch.Generator().manual_seed(1))]
        _ref = checks.msda_reference(_v, _s, _l, _a)
        torch.testing.assert_close(
            msda_forward(_v, _s, _st, _l, _a, BLOCK_Q=4), _ref,
            rtol=1e-5, atol=1e-5,
        )

    checks.run_checks({
        "matches the repo reference (multi-level, non-pow-2 D, OOB)":
            _matches_reference,
        "matches YOUR Lab 1 ground truth": _matches_your_lab1,
        "exact grid-corner locations (floor/boundary knots)":
            _grid_corner_locations,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you should notice about your own kernel

    - **The scatter never went away — it got amortized.** Corner *rows* are still
      at unpredictable addresses (data-dependent; nothing can fix that). But each
      row read is `BLOCK_D` consecutive channels: for `D=32` fp16 that's 64
      contiguous bytes = 2 full sectors, **zero wasted bytes** — precisely the
      access your Lab 3 simulator scored at efficiency ≈ 1.0.
    - **The same masks do triple duty**: ragged last query block (`mask_q`),
      non-power-of-2 head dims (`mask_d`), *and* zeros-padding for out-of-bounds
      corners (the `vx·vy` validities). One mechanism, three requirements —
      that's why your OOB check passed without any special-case code.
    - **`BLOCK_Q` is a free parameter** you had to pick blind (we handed you 32,
      and 4 in the checkers so tiny test cases still span multiple programs).
      Which value is fastest? On which GPU? That's Lab 6, and you won't guess —
      you'll measure.

    ---
    ### 🏁 Stage complete — save your work

    Copy `msda_forward_kernel` **and** `msda_forward` into **`labs/my/lab05.py`**
    — Lab 10 will wire whatever lives there into a full autograd operator and run
    the repo's own test suite over it. Next: **Lab 6 — make it fast**, where the
    simulator's prediction meets silicon.
    """)
    return


if __name__ == "__main__":
    app.run()
