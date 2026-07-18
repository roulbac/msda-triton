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
def _(mo, stage, stage_banner):
    msda_backward_torch, _bwd_src = stage(7, "msda_backward_torch")
    stage_banner(mo, {"msda_backward_torch": _bwd_src})
    return (msda_backward_torch,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lab 8 — Concurrent writes: races, atomics, and the backward kernel

    Lab 7 ended on a cliffhanger: on the GPU, thousands of programs will add into
    `grad_value` **at the same time**. Today you watch that go wrong, fix it with
    hardware atomics, and ship the full backward kernel.

    ## 1. The crime scene: a lost-update race

    A histogram is scatter-add at its purest: `out[idx[i]] += 1`. Here it is
    written the *plausible* way — load the bin, add one, store it back:
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def histogram_racy_kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        cur = tl.load(out_ptr + idx, mask=mask, other=0.0)   # read...
        tl.store(out_ptr + idx, cur + 1.0, mask=mask)        # ...modify... write

    def histogram_racy(idx, n_bins, BLOCK=256):
        out = torch.zeros(n_bins, dtype=torch.float32, device=idx.device)
        n = idx.numel()
        histogram_racy_kernel[(triton.cdiv(n, BLOCK),)](idx, out, n, BLOCK=BLOCK)
        return out
    return (histogram_racy,)


@app.cell
def _(mo):
    mo.md(r"""
    Two programs holding the same bin can both read `5`, both write `6` — one
    increment vanishes. (It's even racy *within* a program: two lanes of the same
    block sharing a bin collide in the vectorized store.) The interleaving is up
    to the scheduler, so the answer changes run to run.

    ### ✏️ Exercise 1 — the fix

    `tl.atomic_add(ptr, val, mask=..., sem="relaxed")` makes each lane's
    read-modify-write **indivisible**: the memory system serializes collisions,
    nothing is lost. Rewrite the histogram with it (note: atomic_add *replaces*
    the load/store pair — it's one operation, not a guard around two).
    """)
    return


@app.cell
def _(tl, torch, triton):
    @triton.jit
    def histogram_atomic_kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        # ================= YOUR CODE ================= (two lines)
        pass
        # =============================================

    def histogram_atomic(idx, n_bins, BLOCK=256):
        out = torch.zeros(n_bins, dtype=torch.float32, device=idx.device)
        n = idx.numel()
        histogram_atomic_kernel[(triton.cdiv(n, BLOCK),)](idx, out, n, BLOCK=BLOCK)
        return out
    return (histogram_atomic,)


@app.cell
def _(checks, histogram_atomic, torch):
    def _exact_counts():
        _idx = torch.randint(0, 8, (2000,))
        _h = histogram_atomic(_idx, 8)
        assert not torch.equal(_h, torch.zeros(8)), "kernel wrote nothing yet"
        torch.testing.assert_close(_h, torch.bincount(_idx, minlength=8).float())

    checks.run_checks({"atomic histogram: exact counts, high contention":
                       _exact_counts})
    return


@app.cell
def _(histogram_atomic, histogram_racy, mo, setup, torch):
    if setup.HAS_CUDA:
        _idx = torch.randint(0, 64, (1_000_000,), device="cuda")
        _true = float(_idx.numel())
        _racy = [float(histogram_racy(_idx, 64).sum()) for _ in range(3)]
        try:
            _atomic = float(histogram_atomic(_idx, 64).sum())
        except Exception:
            _atomic = float("nan")
        _lost = [f"{100 * (1 - r / _true):.1f}%" for r in _racy]
        mo.md(
            f"**1,000,000 increments on your GPU.** Racy kernel kept "
            f"{_racy[0]:,.0f} / {_true:,.0f} — losing {_lost[0]}, then "
            f"{_lost[1]}, then {_lost[2]} across three runs (different every "
            f"time: that's the scheduler). Atomic kernel: "
            f"**{_atomic:,.0f} — exact, every run.**"
        ).callout(kind="danger")
    else:
        mo.md(
            "🟡 **Interpreter note:** the CPU interpreter runs programs "
            "*serially*, so the racy kernel appears correct here — a concurrency "
            "bug needs concurrency to show itself (itself a lesson: your tests "
            "must run on hardware that can actually race). On an L4 GPU this "
            "demo loses ~90% of increments: `histogram_racy` keeps ~100k of 1M, "
            "differently each run, while the atomic version is exact."
        ).callout(kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. `sem="relaxed"` — buying speed with algebra

    By default GPU atomics come with *memory-ordering* guarantees (fences), so
    you can build locks and queues from them. Fences constrain how the memory
    subsystem may buffer and combine traffic — expensive when you're about to
    issue millions of atomics. Gradient accumulation needs none of it:

    - addition **commutes** — any arrival order gives the same sum (to rounding);
    - nobody **reads** `grad_value` mid-kernel; consumers wait for kernel
      completion, and kernel completion is itself a full barrier.

    So we demote to `sem="relaxed"`: indivisibility only. On SM 9.0+ hardware
    this lowers to a fire-and-forget reduction instruction (`red.add`) rather
    than an ordered read-modify-write — Lab 9 will show you this in the PTX.

    ## 3. ✏️ Exercise 2 — the backward kernel

    Same scaffold as Lab 5 (grid, offsets, masks, and now also the corner
    gathers are given — recomputing them from saved inputs is cheaper than
    storing them in the forward). You write the *new* physics, straight from
    your Lab 7 derivation:

    1. `grad_attn`: $g \cdot \mathrm{sampled}$, stored to `grad_attn_ptr + p`;
    2. `grad_loc`: corner differences × $A \cdot W$ (resp. $H$), stored;
    3. `grad_value`: **four relaxed atomic adds** through the corner masks.
    """)
    return


@app.cell
def _(tl, triton):
    @triton.jit
    def msda_backward_kernel(
        value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr,
        grad_out_ptr, grad_value_ptr, grad_loc_ptr, grad_attn_ptr,
        Q, S,
        M: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # ---- given: identical bookkeeping to your forward ----
        pid_q = tl.program_id(0)
        pid_bm = tl.program_id(1)
        b = (pid_bm // M).to(tl.int64)
        m = pid_bm % M

        offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask_q = offs_q < Q
        mask_d = offs_d < D
        mask_qd = mask_q[:, None] & mask_d[None, :]

        pq = (b * Q + offs_q) * M + m
        val_base = value_ptr + (b * S * M + m) * D + offs_d[None, :]
        gval_base = grad_value_ptr + (b * S * M + m) * D + offs_d[None, :]

        g = tl.load(grad_out_ptr + pq[:, None] * D + offs_d[None, :],
                    mask=mask_qd, other=0.0).to(tl.float32)

        for l in tl.static_range(L):
            H = tl.load(shapes_ptr + 2 * l)
            W = tl.load(shapes_ptr + 2 * l + 1)
            start = tl.load(starts_ptr + l)
            fh = H.to(tl.float32)
            fw = W.to(tl.float32)
            for k in tl.static_range(K):
                p = pq * (L * K) + l * K + k
                loc_x = tl.load(loc_ptr + 2 * p, mask=mask_q, other=0.0).to(tl.float32)
                loc_y = tl.load(loc_ptr + 2 * p + 1, mask=mask_q, other=0.0).to(tl.float32)
                attn = tl.load(attn_ptr + p, mask=mask_q, other=0.0).to(tl.float32)

                x = loc_x * fw - 0.5
                y = loc_y * fh - 0.5
                x0f = tl.math.floor(x)
                y0f = tl.math.floor(y)
                lx = x - x0f
                ly = y - y0f
                x0 = x0f.to(tl.int64)
                y0 = y0f.to(tl.int64)
                x1 = x0 + 1
                y1 = y0 + 1

                vx0 = mask_q & (x0 >= 0) & (x0 < W)
                vx1 = mask_q & (x1 >= 0) & (x1 < W)
                vy0 = (y0 >= 0) & (y0 < H)
                vy1 = (y1 >= 0) & (y1 < H)
                row0 = start + y0 * W
                row1 = row0 + W
                m00 = (vy0 & vx0)[:, None] & mask_d[None, :]
                m01 = (vy0 & vx1)[:, None] & mask_d[None, :]
                m10 = (vy1 & vx0)[:, None] & mask_d[None, :]
                m11 = (vy1 & vx1)[:, None] & mask_d[None, :]
                off00 = ((row0 + x0) * (M * D))[:, None]
                off01 = ((row0 + x1) * (M * D))[:, None]
                off10 = ((row1 + x0) * (M * D))[:, None]
                off11 = ((row1 + x1) * (M * D))[:, None]

                v00 = tl.load(val_base + off00, mask=m00, other=0.0).to(tl.float32)
                v01 = tl.load(val_base + off01, mask=m01, other=0.0).to(tl.float32)
                v10 = tl.load(val_base + off10, mask=m10, other=0.0).to(tl.float32)
                v11 = tl.load(val_base + off11, mask=m11, other=0.0).to(tl.float32)

                w00 = (1.0 - lx) * (1.0 - ly)
                w01 = lx * (1.0 - ly)
                w10 = (1.0 - lx) * ly
                w11 = lx * ly

                # ================= YOUR CODE =================
                # 1. sampled = Σ corners*weights;  ga = tl.sum(g*sampled, axis=1)
                #    tl.store(grad_attn_ptr + p, ga.to(
                #        grad_attn_ptr.dtype.element_ty), mask=mask_q)
                # 2. dx, dy (corner differences); gx = attn * fw * tl.sum(g*dx, 1)
                #    gy = attn * fh * ...; store to grad_loc_ptr + 2*p (+1)
                # 3. wg = g * attn[:, None]; four
                #    tl.atomic_add(gval_base + offXX,
                #        (wg * wXX[:, None]).to(grad_value_ptr.dtype.element_ty),
                #        mask=mXX, sem="relaxed")
                pass
                # =============================================

    def msda_backward(value, spatial_shapes, level_start_index,
                      sampling_locations, attention_weights, grad_out,
                      fp32_grad_accum=False, BLOCK_Q=16, num_warps=4):
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
        value = value.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        grad_out = grad_out.reshape(B, Q, M, D).contiguous()

        _acc_dtype = _torch.float32 if fp32_grad_accum else value.dtype
        grad_value = _torch.zeros_like(value, dtype=_acc_dtype)
        grad_loc = _torch.empty_like(sampling_locations)
        grad_attn = _torch.empty_like(attention_weights)

        _grid = (triton.cdiv(Q, BLOCK_Q), B * M)
        msda_backward_kernel[_grid](
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
            grad_out, grad_value, grad_loc, grad_attn,
            Q, S, M=M, D=D, L=L, K=K,
            BLOCK_Q=BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
            num_warps=num_warps,
        )
        if grad_value.dtype != value.dtype:
            grad_value = grad_value.to(value.dtype)
        return grad_value, grad_loc, grad_attn
    return (msda_backward,)


@app.cell
def _(mo):
    mo.accordion({
        "Hint — region 1 and 2 are Lab 7 lines with tl. spelling": mo.md(
            "```python\nsampled = (v00 * w00[:, None] + v01 * w01[:, None]\n"
            "           + v10 * w10[:, None] + v11 * w11[:, None])\n"
            "ga = tl.sum(g * sampled, axis=1)\n"
            "dx = (v01 - v00) * (1.0 - ly)[:, None] + (v11 - v10) * ly[:, None]\n"
            "dy = (v10 - v00) * (1.0 - lx)[:, None] + (v11 - v01) * lx[:, None]\n"
            "gx = attn * fw * tl.sum(g * dx, axis=1)\n"
            "gy = attn * fh * tl.sum(g * dy, axis=1)\n```"
        ),
        "Hint — one atomic, fully spelled": mo.md(
            "```python\nwg = g * attn[:, None]\n"
            "gdt = grad_value_ptr.dtype.element_ty\n"
            "tl.atomic_add(gval_base + off00, (wg * w00[:, None]).to(gdt),\n"
            "              mask=m00, sem=\"relaxed\")\n```\n"
            "A masked-off lane's atomic simply doesn't happen — an out-of-bounds "
            "corner contributed zero forward, so it receives zero backward."
        ),
    })
    return


@app.cell
def _(checks, msda_backward, msda_backward_torch, setup, torch):
    _cases = ([dict(B=1, Q=4, M=2, D=4, shapes=[(5, 7), (3, 4)], K=2,
                    loc_lo=-0.3, loc_hi=1.3)]
              if setup.INTERPRETED else None)

    def _matches_lab7():
        for _case in _cases or checks.DEFAULT_CASES:
            _v, _s, _st, _l, _a = checks.make_inputs(**_case, dtype=torch.float32)
            _g = torch.randn(_v.shape[0], _l.shape[1], _v.shape[2] * _v.shape[3])
            _gv, _gl, _ga = msda_backward(_v, _s, _st, _l, _a, _g, BLOCK_Q=4)
            assert not torch.equal(_gl, torch.zeros_like(_gl)), (
                "kernel wrote nothing yet"
            )
            _gv_t, _gl_t, _ga_t = msda_backward_torch(_v, _s, _st, _l, _a, _g)
            torch.testing.assert_close(_gv, _gv_t, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(_gl, _gl_t, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(_ga, _ga_t, rtol=1e-4, atol=1e-4)

    def _matches_autograd():
        checks.assert_grads_match(
            lambda v, s, st, l, a, g: msda_backward(v, s, st, l, a, g, BLOCK_Q=4),
            cases=_cases, dtype=torch.float32,
        )

    checks.run_checks({
        "matches YOUR Lab 7 torch backward": _matches_lab7,
        "matches autograd on the reference": _matches_autograd,
    })
    return


@app.cell
def _(checks, mo, msda_backward, setup, torch):
    if setup.HAS_CUDA:
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=2, Q=300, M=8, D=32, shapes=[(50, 84), (25, 42)], K=4,
            dtype=torch.float32, device="cuda",
        )
        _g = torch.randn(2, 300, 8 * 32, device="cuda")
        _g1 = msda_backward(_v, _s, _st, _l, _a, _g)[0]
        _g2 = msda_backward(_v, _s, _st, _l, _a, _g)[0]
        _diff = float((_g1 - _g2).abs().max())
        mo.md(
            f"**Nondeterminism, measured.** Two identical backward calls: "
            f"max |Δgrad_value| = `{_diff:.2e}`. Floating-point addition isn't "
            f"associative, atomics commit in scheduler order, so trailing bits "
            f"differ run to run. This is normal (PyTorch documents the same for "
            f"`index_add_` on CUDA, embedding backward, ...) — it's why the repo's "
            f"gradient tests use tolerances, and what "
            f"`torch.use_deterministic_algorithms(True)` would refuse to run."
        ).callout(kind="info")
    else:
        mo.md(
            "🟡 On the serial interpreter two runs are bit-identical. On a GPU, "
            "run-to-run max |Δgrad_value| is typically ~1e-7 (fp32): atomics "
            "commit in scheduler order and floating-point addition is not "
            "associative. The repo's tests use tolerances for exactly this reason."
        ).callout(kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### 🏁 Stage complete — save your work

    Copy `msda_backward_kernel` **and** `msda_backward` into
    **`labs/my/lab08.py`** (Lab 10 builds the autograd op from it). You now have
    a complete, correct forward + backward. But one dial is still set blindly:
    `grad_value`'s **dtype**. It looks like a detail. On an A100 with bf16 it is
    a **10× performance cliff**, and you can see the cliff *in the compiled
    assembly without owning either GPU*. Next: **Lab 9 — hardware detective**.
    """)
    return


if __name__ == "__main__":
    app.run()
