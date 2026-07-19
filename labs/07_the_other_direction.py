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

    from labs.common import checks
    return checks, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 7 — The other direction: deriving the backward

    Your forward kernel can't train a model until gradients flow through it.
    PyTorch's autograd can't differentiate a Triton kernel — you left Python —
    so **you** are the autograd now: derive the three gradients on paper, then
    implement them in torch (where mistakes are cheap and comparable against real
    autograd). Lab 8 ports what you build here into a kernel.

    No GPU needed anywhere in this lab.

    ## Setting up the derivation

    Fix one query/head; write the forward for corner values $v_{00..11}$, corner
    weights $w_{00} = (1{-}l_x)(1{-}l_y)$ etc., and attention weight $A_{lk}$:

    $$
    \mathrm{out} = \sum_{l,k} A_{lk} \cdot \underbrace{\big( w_{00} v_{00} +
    w_{01} v_{01} + w_{10} v_{10} + w_{11} v_{11} \big)}_{\text{sampled}(l,k)}
    $$

    Autograd hands you $g = \partial \mathcal{L} / \partial\,\mathrm{out}$ (one
    $D$-vector per query/head) and wants gradients w.r.t. **attention weights**,
    **sampling locations**, and **values**. Derive each *before* opening its
    accordion — pen and paper, five minutes each.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "∂L/∂A — derive, then check yourself": mo.md(
            "The output is *linear* in $A_{lk}$, so\n\n"
            "$$\\frac{\\partial \\mathcal{L}}{\\partial A_{lk}} = g \\cdot "
            "\\mathrm{sampled}(l,k)$$\n\na dot product over channels. One scalar "
            "per sample point, written to its own slot — *no memory drama.*"
        ),
        "∂L/∂x — derive, then check yourself": mo.md(
            "Bilinear interpolation is piecewise-linear in $x$: its derivative "
            "is the **corner difference**, blended vertically:\n\n"
            "$$\\frac{\\partial\\,\\mathrm{sampled}}{\\partial x_{\\mathrm{im}}} = "
            "(v_{01} - v_{00})(1 - l_y) + (v_{11} - v_{10})\\, l_y$$\n\n"
            "then the chain rule through $x_{\\mathrm{im}} = x \\cdot W - 0.5$ "
            "multiplies by $W$:\n\n"
            "$$\\frac{\\partial \\mathcal{L}}{\\partial x} = A_{lk} \\cdot W \\cdot "
            "\\big(g \\cdot \\tfrac{\\partial\\,\\mathrm{sampled}}{\\partial "
            "x_{\\mathrm{im}}}\\big)$$\n\n(similarly $y$ with $H$ and the roles of "
            "$l_x, l_y$ swapped). **The $\\cdot W$ is the classic forgotten "
            "factor** — the checker below will catch you if you drop it."
        ),
        "∂L/∂v — derive, then check yourself": mo.md(
            "Corner $v_{00}$ entered with coefficient $A_{lk} w_{00}$, so it "
            "receives $A_{lk} w_{00}\\, g$. But here's the twist: a given "
            "*pixel* may be a corner for **many queries** — its total gradient "
            "is a sum over every sample that landed nearby:\n\n"
            "$$\\frac{\\partial \\mathcal{L}}{\\partial v_p} = \\sum_{\\text{all }"
            "(q,m,l,k)\\text{ touching } p} A\\, w\\, g$$\n\n"
            "The forward **gathered** (scattered reads → private sum). The "
            "backward **scatters** (private values → sums at scattered "
            "addresses). This asymmetry is generic to every sampling operator, "
            "and it's why backward passes of this family are slower than "
            "forwards in *every* implementation, mmcv's CUDA included."
        ),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ The exercise — `msda_backward_torch`

    One function, three fill-in regions (A, then locations, then the value
    scatter). The scaffold mirrors your Lab 2 `flat_msda`: vectorized over
    `(B, Q, M)`, loops over `(l, k)`, flat indices throughout — so Lab 8's port
    to Triton is a transliteration, not a redesign.

    For the scatter, torch gives you `grad_value.index_add_(0, idx, contrib)` on
    a flat buffer — the safe, serial cousin of the atomic adds Lab 8 needs
    (torch guarantees repeated indices accumulate; it's *how* GPUs guarantee
    that which Lab 8 is about).
    """)
    return


@app.cell
def _(checks, torch):
    def msda_backward_torch(value, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights, grad_out):
        """Returns (grad_value (B,S,M,D), grad_loc (B,Q,M,L,K,2),
        grad_attn (B,Q,M,L,K)), all float32. grad_out: (B, Q, M*D)."""
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        _dev = value.device

        v = value.reshape(-1).float()
        loc = sampling_locations.float()
        attn = attention_weights.float()
        g = grad_out.reshape(B, Q, M, D).float()

        grad_value = torch.zeros(B * S * M * D, dtype=torch.float32, device=_dev)
        grad_loc = torch.zeros(B, Q, M, L, K, 2, dtype=torch.float32, device=_dev)
        grad_attn = torch.zeros(B, Q, M, L, K, dtype=torch.float32, device=_dev)

        b = torch.arange(B, dtype=torch.int64, device=_dev)[:, None, None]
        m = torch.arange(M, dtype=torch.int64, device=_dev)[None, None, :]
        d = torch.arange(D, dtype=torch.int64, device=_dev)
        val_base = (b * S) * M + m         # add s*M, then *D + d for a channel

        for lvl in range(L):
            H = int(spatial_shapes[lvl, 0])
            W = int(spatial_shapes[lvl, 1])
            start = int(level_start_index[lvl])
            # ================= YOUR CODE =================
            # for k in range(K):
            #   1. x, y, a; floor; lx/ly; corner indices + validities (Lab 2)
            #   2. gather the four corner values v00..v11 (clamp+zero trick),
            #      shape (B, Q, M, D)
            #   3. grad_attn[..., lvl, k] = (g * sampled).sum(-1)
            #   4. dx = (v01-v00)*(1-ly)[...,None] + (v11-v10)*ly[...,None]
            #      dy = (v10-v00)*(1-lx)[...,None] + (v11-v01)*lx[...,None]
            #      grad_loc[..., lvl, k, 0] = a * W * (g*dx).sum(-1)   (y: H)
            #   5. wg = g * a[..., None]; for each corner:
            #      contrib = wg * weight[..., None] * valid[..., None]
            #      grad_value.index_add_(0, idx.clamp(0, v.numel()-1).reshape(-1),
            #                            contrib.reshape(-1))
            raise checks.NotDoneYet()
            # =============================================

        return grad_value.view(B, S, M, D), grad_loc, grad_attn
    return (msda_backward_torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint 1 — reuse your Lab 2 gather verbatim": mo.md(
            "```python\ndef gather(row, xc, valid):\n"
            "    idx = ((val_base + (row + xc) * M) * D)[..., None] + d\n"
            "    return v[idx.clamp(0, v.numel() - 1)] * valid[..., None], idx\n```\n"
            "Returning `idx` too means the scatter below reuses it."
        ),
        "Hint 2 — the three formulas in code": mo.md(
            "```python\nsampled = (v00*w00[...,None] + v01*w01[...,None]\n"
            "           + v10*w10[...,None] + v11*w11[...,None])\n"
            "grad_attn[:, :, :, lvl, k] = (g * sampled).sum(-1)\n"
            "dx = (v01 - v00)*(1-ly)[...,None] + (v11 - v10)*ly[...,None]\n"
            "dy = (v10 - v00)*(1-lx)[...,None] + (v11 - v01)*lx[...,None]\n"
            "grad_loc[:, :, :, lvl, k, 0] = a * W * (g * dx).sum(-1)\n"
            "grad_loc[:, :, :, lvl, k, 1] = a * H * (g * dy).sum(-1)\n```"
        ),
        "Hint 3 — the scatter": mo.md(
            "```python\nwg = g * a[..., None]\nfor idx, valid, w in [\n"
            "    (idx00, valid00, w00), (idx01, valid01, w01),\n"
            "    (idx10, valid10, w10), (idx11, valid11, w11)]:\n"
            "    contrib = wg * w[..., None] * valid[..., None]\n"
            "    grad_value.index_add_(0, idx.clamp(0, v.numel()-1).reshape(-1),\n"
            "                          contrib.reshape(-1))\n```\n"
            "Invalid corners scatter a *zero* to a clamped (legal) address — "
            "harmless, and exactly what a masked atomic will do in Lab 8."
        ),
    })
    return


@app.cell(hide_code=True)
def _(checks, msda_backward_torch, torch):
    def _matches_autograd():
        checks.assert_grads_match(msda_backward_torch, dtype=torch.float32)

    def _oob_gradients_dont_leak():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=6, M=2, D=4, shapes=[(5, 7)], K=2,
            dtype=torch.float32, loc_lo=-0.5, loc_hi=1.5,
        )
        _g = torch.randn(1, 6, 2 * 4)
        _gv, _gl, _ga = msda_backward_torch(_v, _s, _st, _l, _a, _g)
        _gv_ref, _gl_ref, _ga_ref = checks.reference_grads(_v, _s, _l, _a, _g)
        torch.testing.assert_close(_gv, _gv_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(_gl, _gl_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(_ga, _ga_ref, rtol=1e-4, atol=1e-4)

    checks.run_checks({
        "all three gradients match autograd on the reference": _matches_autograd,
        "out-of-bounds samples: correct (zero) gradient flow":
            _oob_gradients_dont_leak,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice what the checker is doing, because it's a technique to steal: the
    reference forward is pure PyTorch, so **autograd differentiates it for
    free** — your hand-derived formulas are being graded against machine-derived
    truth. A dropped $W$ factor, a swapped corner, a sign error: all caught
    without anyone writing expected values by hand.

    ## The last doubt: is autograd itself right?

    Healthy paranoia. The independent oracle is the **finite difference**: nudge
    one input scalar by $\pm\epsilon$ and difference the outputs. Below we pick
    one sampling-location coordinate and compare all three answers — yours,
    autograd's, and arithmetic's:
    """)
    return


@app.cell(hide_code=True)
def _(checks, mo, msda_backward_torch, torch):
    try:
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=3, M=1, D=2, shapes=[(4, 5)], K=1, dtype=torch.float64
        )
        _g = torch.ones(1, 3, 2, dtype=torch.float64)
        _eps = 1e-6

        _gl_mine = msda_backward_torch(_v, _s, _st, _l, _a, _g)[1][0, 1, 0, 0, 0, 0]

        _lp, _lm = _l.clone(), _l.clone()
        _lp[0, 1, 0, 0, 0, 0] += _eps
        _lm[0, 1, 0, 0, 0, 0] -= _eps
        _fd = (checks.msda_reference(_v, _s, _lp, _a).sum()
               - checks.msda_reference(_v, _s, _lm, _a).sum()) / (2 * _eps)

        _gl_auto = checks.reference_grads(_v, _s, _l, _a, _g)[1][0, 1, 0, 0, 0, 0]

        mo.md(
            f"∂L/∂x for one sample point — three independent answers:\n\n"
            f"| method | value |\n|---|---|\n"
            f"| your formula | `{float(_gl_mine):+.8f}` |\n"
            f"| autograd | `{float(_gl_auto):+.8f}` |\n"
            f"| finite difference | `{float(_fd):+.8f}` |\n\n"
            f"(`torch.autograd.gradcheck` automates exactly this comparison over "
            f"every input element — worth knowing when you write your next "
            f"custom op.)"
        )
    except (checks.NotDoneYet, NotImplementedError):
        mo.md("*(finish the exercise to run the three-way comparison)*").callout(
            kind="neutral"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 🏁 Stage complete — save your work

    Copy `msda_backward_torch` into **`labs/my/lab07.py`** — Lab 8's kernel is
    checked against it, and Lab 10 wires it all into autograd. One loose end
    remains, and it's a big one: `index_add_` ran your scatter *serially*. On a
    GPU, thousands of programs will scatter **concurrently into the same
    buffer**. What could possibly go wrong? Next: **Lab 8 — concurrent writes**.
    """)
    return


if __name__ == "__main__":
    app.run()
