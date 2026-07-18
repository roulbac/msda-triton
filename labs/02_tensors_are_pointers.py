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

    from labs.common import checks
    from labs.common.loader import stage
    return checks, mo, stage, torch


@app.cell
def _(mo, stage):
    msda_naive, _naive_src = stage(1, "msda_naive_student")
    from labs.common.loader import stage_banner

    stage_banner(mo, {"msda_naive_student": _naive_src})
    return (msda_naive,)


@app.cell
def _(mo):
    mo.md(r"""
    # Lab 2 — Tensors are pointers

    A Triton kernel does not receive tensors. It receives **the memory address of
    element zero** — no shape, no strides, no `[b, q, m]` indexing, no `.view()`.
    Every access is `base_address + offset` where *you* compute the offset.

    This lab takes your Lab 1 implementation and strips away every indexing
    convenience, so that when you write the real kernel in Lab 5, the pointer
    arithmetic is already muscle memory. You'll also meet the first genuine
    production bug class of this repo: **integer overflow in flat indices**.

    ## Row-major layout in 60 seconds

    A contiguous tensor of shape $(A, B, C)$ is one flat array where element
    $(a, b, c)$ lives at flat position

    $$\text{flat} = (a \cdot B + b) \cdot C + c$$

    — the last axis varies fastest. Nesting deeper is mechanical: for
    `value (B, S, M, D)`, element $(b, s, m, d)$ is at
    $((b \cdot S + s) \cdot M + m) \cdot D + d$. That formula **is** the kernel's
    `val_base` pointer math in `src/msda_triton/kernels.py`; go look after this lab.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ✏️ Exercise 1 — warm-up: `flat_index`

    No tensors, just the formula.
    """)
    return


@app.cell
def _(checks):
    def flat_index(b, s, m, d, S, M, D):
        """Flat position of value[b, s, m, d] in a contiguous (B, S, M, D)."""
        # ================= YOUR CODE =================
        raise checks.NotDoneYet()
        # =============================================
    return (flat_index,)


@app.cell
def _(checks, flat_index, torch):
    def _matches_view():
        _v = torch.arange(2 * 30 * 3 * 4).reshape(2, 30, 3, 4)
        _flat = _v.reshape(-1)
        for _b, _s, _m, _d in [(0, 0, 0, 0), (1, 17, 2, 3), (0, 29, 1, 2)]:
            _i = flat_index(_b, _s, _m, _d, 30, 3, 4)
            assert _flat[_i] == _v[_b, _s, _m, _d], (
                f"flat_index({_b},{_s},{_m},{_d}) = {_i} points at "
                f"{int(_flat[_i])}, expected {int(_v[_b, _s, _m, _d])}"
            )

    checks.run_checks({"flat_index agrees with tensor indexing": _matches_view})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The gather trick: masks before you meet masks

    In Lab 5, out-of-bounds corners will be handled by *masked loads*: lanes whose
    corner is outside the image simply don't read, and receive `0.0` instead.
    Torch has no masked gather — indexing with an out-of-range index throws. The
    standard translation, which you'll use below and see again written as
    `tl.load(ptr, mask=valid, other=0.0)`:

    ```python
    idx = idx.clamp(0, flat.numel() - 1)     # make every index legal...
    vals = flat[idx] * valid                  # ...then zero the illegal lanes
    ```

    ### ✏️ Exercise 2 — `flat_msda`

    Reimplement MSDA under kernel rules:

    - `value`, `sampling_locations`, `attention_weights` may only be used as
      **flattened 1-D views** (`.reshape(-1)`); all indexing through offsets you
      compute. (`spatial_shapes` / `level_start_index` are tiny metadata — read
      them normally, like the kernel loads its scalars.)
    - Python loops only over `l` and `k`. Vectorize over `(B, Q, M)` with index
      tensors — this mirrors the kernel exactly: a GPU launches parallel programs
      over `(B, Q, M)` and loops over the small `L×K` inside.
    - Do the bilinear math on whole `(B, Q, M)` tensors at once.

    The key offsets (`d = torch.arange(D)`):

    ```text
    pq        = (b*Q + q)*M + m                      # into loc/attn layouts
    p         = pq*(L*K) + l*K + k                   # this sample's slot
    loc_x     = loc_flat[2*p],  loc_y = loc_flat[2*p + 1]
    corner    = ((b*S + start + y*W + x)*M + m)*D    # + d for the channel row
    ```
    """)
    return


@app.cell
def _(checks, torch):
    def flat_msda(value, spatial_shapes, level_start_index, sampling_locations,
                  attention_weights):
        """Same result as Lab 1, but only flat views + hand-computed offsets.

        Returns (B, Q, M*D)."""
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        _dev = value.device
        _acc_dtype = value.dtype if value.dtype == torch.float64 else torch.float32

        v = value.reshape(-1)
        loc = sampling_locations.reshape(-1).to(_acc_dtype)
        attn = attention_weights.reshape(-1).to(_acc_dtype)

        # Broadcastable index tensors over the parallel axes:
        b = torch.arange(B, dtype=torch.int64, device=_dev)[:, None, None]
        q = torch.arange(Q, dtype=torch.int64, device=_dev)[None, :, None]
        m = torch.arange(M, dtype=torch.int64, device=_dev)[None, None, :]
        d = torch.arange(D, dtype=torch.int64, device=_dev)

        out = torch.zeros(B, Q, M, D, dtype=_acc_dtype, device=_dev)
        # ================= YOUR CODE =================
        # for lvl in range(L):                       # H, W, start from metadata
        #     for k in range(K):
        #         p = ...                            # (B, Q, M) int64
        #         x = loc[2*p] * W - 0.5;  y = loc[2*p+1] * H - 0.5
        #         floor -> x0, y0, lx, ly; corner validities vx0/vx1/vy0/vy1
        #         for each corner: idx (B,Q,M,D) via the corner formula,
        #             clamp+gather, zero invalid lanes, out += attn*weight*vals
        raise checks.NotDoneYet()
        # =============================================
        return out.view(B, Q, M * D).to(value.dtype)
    return (flat_msda,)


@app.cell
def _(mo):
    mo.accordion({
        "Hint 1 — one corner, end to end": mo.md(
            "```python\nrow0 = start + y0 * W          # (B, Q, M)\n"
            "valid = (y0 >= 0) & (y0 < H) & (x0 >= 0) & (x0 < W)\n"
            "idx = (((b * S + row0 + x0) * M + m) * D)[..., None] + d\n"
            "vals = v[idx.clamp(0, v.numel() - 1)].to(out.dtype)\n"
            "vals = vals * valid[..., None]\n"
            "out += vals * (a * (1 - lx) * (1 - ly))[..., None]\n```"
        ),
        "Hint 2 — the other three corners": mo.md(
            "`x1 = x0 + 1`, `y1 = y0 + 1`, `row1 = row0 + W`; weights "
            "`lx*(1-ly)`, `(1-lx)*ly`, `lx*ly`; validities pair `vy0/vy1` with "
            "`vx0/vx1`. Consider a small helper closure taking "
            "`(row, xc, valid, weight)` — the reference solution has one."
        ),
    })
    return


@app.cell
def _(checks, flat_msda, msda_naive, torch):
    def _matches_reference():
        checks.assert_msda_matches(flat_msda, dtype=torch.float64, takes_starts=True)

    def _matches_your_lab1():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=2, Q=4, M=2, D=3, shapes=[(5, 6), (3, 3)], K=2,
            dtype=torch.float64, loc_lo=-0.2, loc_hi=1.2,
        )
        torch.testing.assert_close(
            flat_msda(_v, _s, _st, _l, _a), msda_naive(_v, _s, _l, _a)
        )

    checks.run_checks({
        "matches the repo reference": _matches_reference,
        "matches your Lab 1 implementation": _matches_your_lab1,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The overflow trap

    Flat indices *multiply* dimensions together, and the products get big faster
    than intuition suggests. A signed 32-bit integer tops out at
    $2^{31} - 1 = 2{,}147{,}483{,}647$.

    ### ✏️ Exercise 3 — when does MSDA's flat index overflow int32?

    The *encoder* preset processes every pyramid token as a query. For a
    1333×800-ish input the pyramid has $S \approx 138{,}000$ tokens at strides
    8/16/32/64; production models run larger. With `M=8` heads and `D=32`
    channels, the largest flat index into `value` is just under `B·S·M·D`.
    """)
    return


@app.cell
def _(checks):
    def max_flat_index(B, S, M, D):
        """Largest flat index into a contiguous (B, S, M, D): the LAST element."""
        # ================= YOUR CODE ================= (one line)
        raise checks.NotDoneYet()
        # =============================================

    def overflows_int32(B, S, M, D):
        """True if indexing value (B,S,M,D) needs 64-bit arithmetic."""
        # ================= YOUR CODE ================= (one line)
        raise checks.NotDoneYet()
        # =============================================
    return max_flat_index, overflows_int32


@app.cell
def _(checks, max_flat_index, overflows_int32):
    def _last_element():
        assert max_flat_index(2, 10, 3, 4) == 2 * 10 * 3 * 4 - 1

    def _decoder_fits_encoder_does_not():
        assert not overflows_int32(4, 138_000, 8, 32), "decoder-ish sizes fit"
        assert overflows_int32(2, 4_200_000, 8, 32), "a 4K-video pyramid does not"

    checks.run_checks({
        "max_flat_index is the last element": _last_element,
        "int32 verdicts (small fits, huge doesn't)": _decoder_fits_encoder_does_not,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    Work out `2 * 4_200_000 * 8 * 32` — comfortably past $2^{31}$. This is why
    `kernels.py` sprinkles `.to(tl.int64)` on every index that feeds the value
    pointer: **at toy scale an int32 kernel is bit-identical to an int64 one, and
    at production scale it silently reads garbage.** (The repo even benchmarks an
    int32 variant — `benchmarks/modal_variants.py` — because 64-bit index math
    costs a little; the shipped kernel chooses correctness.)

    ---
    ### 🏁 Stage complete — save your work

    Copy `flat_index` and `flat_msda` into **`labs/my/lab02.py`**. You now compute
    addresses exactly the way the kernel will. Next: **Lab 3 — the machine** —
    before writing GPU code, you'll build a model of what GPU *memory* does with
    your addresses, and use it to predict this repo's headline result.
    """)
    return


if __name__ == "__main__":
    app.run()
