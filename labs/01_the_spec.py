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
    return checks, mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Lab 1 — The spec: MSDA in plain Python

    Welcome to the bottom of the ladder. Ten labs from now you'll have rebuilt this
    repo's optimized GPU operator from scratch; today you build the thing every
    later stage is tested against: **multi-scale deformable attention (MSDA), by
    definition, in loops you can verify with your eyes.**

    In *From NAND to Tetris* terms, this lab is the chip specification. Everything
    that follows — flat indexing, memory simulators, Triton kernels, atomics — must
    reproduce *exactly* what you write today, only faster.

    **What MSDA is, in one paragraph.** Detectors like Deformable DETR run
    attention over image feature pyramids with ~100,000 positions. Comparing every
    query against every position is hopeless, so deformable attention lets each
    query *predict* a handful of interesting locations — `K` points per feature
    level per head — and reads only those, blending each read from the 4 nearest
    pixels (**bilinear interpolation**) and combining reads with learned
    **attention weights**. No query–key dot products at all.

    *(Want the full story with pictures? Read
    [course chapter 1](https://roulbac.github.io/msda-triton/course/01-deformable-attention/)
    — this lab is its hands-on twin.)*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Bilinear interpolation: reading between pixels

    The network's sampling locations are *continuous* — $(2.7, 1.3)$ — but images
    only have values at integer pixels. Bilinear interpolation blends the 4
    surrounding pixels by proximity. For a point $(x, y)$ with $x_0 = \lfloor x
    \rfloor$, $y_0 = \lfloor y \rfloor$ and fractional parts $l_x = x - x_0$,
    $l_y = y - y_0$:

    $$
    \text{sample} =
    (1{-}l_x)(1{-}l_y)\,v_{00} + l_x(1{-}l_y)\,v_{01} +
    (1{-}l_x)\,l_y\,v_{10} + l_x l_y\,v_{11}
    $$

    where $v_{00}$ is the pixel at $(y_0, x_0)$, $v_{01}$ at $(y_0, x_0{+}1)$, etc.
    The four weights sum to 1. Drag the point below and watch them:
    """)
    return


@app.cell
def _(mo):
    slider_x = mo.ui.slider(0.0, 4.0, step=0.05, value=1.7, label="x")
    slider_y = mo.ui.slider(0.0, 3.0, step=0.05, value=1.3, label="y")
    mo.hstack([slider_x, slider_y])
    return slider_x, slider_y


@app.cell
def _(mo, slider_x, slider_y):
    import math as _math

    _x, _y = slider_x.value, slider_y.value
    _x0, _y0 = _math.floor(_x), _math.floor(_y)
    _lx, _ly = _x - _x0, _y - _y0
    _w = {
        f"v(y={_y0}, x={_x0})": (1 - _lx) * (1 - _ly),
        f"v(y={_y0}, x={_x0 + 1})": _lx * (1 - _ly),
        f"v(y={_y0 + 1}, x={_x0})": (1 - _lx) * _ly,
        f"v(y={_y0 + 1}, x={_x0 + 1})": _lx * _ly,
    }
    _rows = "\n".join(f"| `{k}` | {v:.3f} |" for k, v in _w.items())
    mo.md(
        f"Sampling at **({_x:.2f}, {_y:.2f})** → corner weights "
        f"(sum = {sum(_w.values()):.3f}):\n\n| corner | weight |\n|---|---|\n{_rows}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two conventions every implementation in this repo agrees on — get either wrong
    and *nothing* downstream will match:

    1. **Coordinate mapping.** Locations arrive normalized to $[0,1]$; they map to
       pixel space as $x_\text{im} = x \cdot W - 0.5$ (and $y \cdot H - 0.5$).
       The $-0.5$ places values at pixel *centers* — identical to PyTorch's
       `grid_sample(align_corners=False)`.
    2. **Zeros padding.** A corner that falls outside the image contributes zero
       (it is *not* clamped to the border). The sample fades smoothly to zero as
       the point leaves the image.

    ### ✏️ Exercise 1 — `bilinear_sample`

    Sample an `(H, W, D)` image at continuous **pixel** coordinates `(x, y)`
    (the `·W − 0.5` mapping is your caller's job, exercise 2). Return a `(D,)`
    vector; out-of-bounds corners contribute zero.
    """)
    return


@app.cell
def _(checks):
    def bilinear_sample(img, x, y):
        """img: (H, W, D) tensor; x, y: floats in pixel coords. -> (D,)"""
        # ================= YOUR CODE =================
        # 1. x0, y0 = floor(x), floor(y);  lx, ly = fractional parts
        # 2. for each of the 4 corners: if inside the image, add weight * pixel
        raise checks.NotDoneYet()
        # =============================================
    return (bilinear_sample,)


@app.cell
def _(mo):
    mo.accordion({
        "Hint 1 — the skeleton": mo.md(
            "```python\nH, W, _ = img.shape\nx0, y0 = math.floor(x), math.floor(y)\n"
            "lx, ly = x - x0, y - y0\nout = img.new_zeros(img.shape[-1])\n```"
            "\nThen loop over the four `(yy, xx, weight)` corner triples."
        ),
        "Hint 2 — the corner table": mo.md(
            "```python\ncorners = (\n    (y0,     x0,     (1-lx)*(1-ly)),\n"
            "    (y0,     x0+1,  lx*(1-ly)),\n    (y0+1,  x0,     (1-lx)*ly),\n"
            "    (y0+1,  x0+1,  lx*ly),\n)\nfor yy, xx, w in corners:\n"
            "    if 0 <= xx < W and 0 <= yy < H:\n        out += w * img[yy, xx]\n```"
        ),
        "Stuck?": mo.md(
            "The reference solution is `labs/solutions/lab01.py` — but every later "
            "lab leans on your understanding of exactly this function, so fight "
            "for it first."
        ),
    })
    return


@app.cell
def _(bilinear_sample, checks, torch):
    def _center_hits_pixel():
        _img = torch.arange(12., dtype=torch.float64).reshape(3, 4, 1)
        # exactly on pixel (1, 2) -> its value, weight 1
        torch.testing.assert_close(bilinear_sample(_img, 2.0, 1.0), _img[1, 2])

    def _midpoint_blends_equally():
        _img = torch.zeros(2, 2, 1, dtype=torch.float64)
        _img[0, 0], _img[0, 1], _img[1, 0], _img[1, 1] = 0.0, 1.0, 2.0, 3.0
        torch.testing.assert_close(
            bilinear_sample(_img, 0.5, 0.5), torch.tensor([1.5], dtype=torch.float64)
        )

    def _outside_fades_to_zero():
        _img = torch.ones(3, 3, 2, dtype=torch.float64)
        torch.testing.assert_close(
            bilinear_sample(_img, -0.5, 1.0),
            torch.full((2,), 0.5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            bilinear_sample(_img, -1.0, 1.0), torch.zeros(2, dtype=torch.float64)
        )

    checks.run_checks({
        "on-pixel sample returns that pixel": _center_hits_pixel,
        "midpoint blends 4 corners equally": _midpoint_blends_equally,
        "out-of-bounds corners contribute zero": _outside_fades_to_zero,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The full operator

    Now assemble MSDA. Here is the equation (Deformable DETR, Eq. 1), and — more
    usefully — the exact tensor contract this repo uses everywhere:

    $$
    \mathrm{MSDA}(q) = \sum_{m=1}^{M} \Big[ \sum_{l=1}^{L} \sum_{k=1}^{K}
    A_{mlqk} \cdot x_l ( \phi_l(\hat{p}_q) + \Delta p_{mlqk} ) \Big]
    $$

    | tensor | shape | meaning |
    |---|---|---|
    | `value` | `(B, S, M, D)` | the feature pyramid, all `L` levels flattened along `S = Σ H_l·W_l`, split into `M` heads × `D` channels |
    | `spatial_shapes` | `(L, 2)` | `(H_l, W_l)` of each level |
    | `sampling_locations` | `(B, Q, M, L, K, 2)` | normalized `(x, y)` in `[0,1]` — offsets already added |
    | `attention_weights` | `(B, Q, M, L, K)` | softmax-normalized over `(L, K)` |
    | output | `(B, Q, M·D)` | per query: heads concatenated |

    Note what the flattened `S` axis means: level `l`'s image starts at offset
    `start_l = Σ_{i<l} H_i·W_i` and pixel `(y, x)` of that level lives at
    `value[b, start_l + y·W_l + x, m]`. You'll compute `start_l` yourself.

    ### ✏️ Exercise 2 — `msda_naive_student`

    Five nested loops (`b, q, m, level, k`); inside: map the normalized location
    to pixel coords, call **your** `bilinear_sample` on the level's image, weight
    by attention, accumulate. Slow is fine. *Obvious* is the goal.
    """)
    return


@app.cell
def _(bilinear_sample, checks):
    def msda_naive_student(value, spatial_shapes, sampling_locations,
                           attention_weights):
        """Returns (B, Q, M*D). Use bilinear_sample for each read."""
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        _ = bilinear_sample  # you'll want this
        # ================= YOUR CODE =================
        # 1. shapes = [(int(h), int(w)) for h, w in spatial_shapes]
        #    starts  = running sum of h*w (level offsets into the S axis)
        # 2. out = value.new_zeros(B, Q, M, D); five loops; inside:
        #      img = value[b, start:start+h*w, m].view(h, w, D)
        #      x = float(loc[b,q,m,l,k,0]) * w - 0.5   (same for y with h)
        #      out[b,q,m] += attn[b,q,m,l,k] * bilinear_sample(img, x, y)
        # 3. return out.view(B, Q, M * D)
        raise checks.NotDoneYet()
        # =============================================
    return (msda_naive_student,)


@app.cell
def _(mo):
    mo.accordion({
        "Hint — level offsets": mo.md(
            "```python\nshapes = [(int(h), int(w)) for h, w in spatial_shapes]\n"
            "starts = [0]\nfor h, w in shapes[:-1]:\n"
            "    starts.append(starts[-1] + h * w)\n```"
        ),
        "Hint — why `.view(h, w, D)` works": mo.md(
            "`value[b, start:start+h*w, m]` is `(h*w, D)`, and the level was "
            "flattened row-major — so reshaping to `(h, w, D)` recovers the image. "
            "Lab 2 makes you do this arithmetic *without* `.view`."
        ),
    })
    return


@app.cell
def _(checks, msda_naive_student, torch):
    def _matches_reference():
        checks.assert_msda_matches(
            lambda v, s, l, a: msda_naive_student(v, s, l, a),
            dtype=torch.float64,
        )

    def _weights_scale_output():
        _v, _s, _st, _l, _a = checks.make_inputs(
            B=1, Q=3, M=2, D=4, shapes=[(5, 7)], K=2, dtype=torch.float64
        )
        _o1 = msda_naive_student(_v, _s, _l, _a)
        _o2 = msda_naive_student(_v, _s, _l, 2 * _a)
        torch.testing.assert_close(2 * _o1, _o2)

    checks.run_checks({
        "matches the repo reference (incl. out-of-bounds)": _matches_reference,
        "output is linear in the attention weights": _weights_scale_output,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    The first check is the important one: it compares you against
    `tests/reference_impls.py::msda_reference` — the `grid_sample`-based
    implementation the *repo's own test suite* trusts. From now on, **your**
    function is ground truth for everything you build.

    ## Why this operator needs the next nine labs

    Count what your loops do for the standard decoder config (`B=4, Q=300, M=8,
    D=32, L=4, K=4`): 153,600 sample points × 4 corner reads × 32 channels ≈ 20M
    multiply-adds — *trivial* arithmetic for a GPU that does trillions per second.
    But each read lands wherever the *network* pointed, scattered across a ~50 MB
    pyramid. This operator is **memory-bound and data-dependent**: the entire
    game is how bytes move, which is exactly what labs 2 and 3 are about.

    ---
    ### 🏁 Stage complete — save your work

    Copy your two functions into **`labs/my/lab01.py`** (create the file):
    later labs import your versions from there, falling back to the reference
    solution if the file is missing. Next: **Lab 2 — tensors are pointers**,
    where `.view(h, w, D)` and fancy indexing get taken away from you, exactly
    like Triton will take them.
    """)
    return


if __name__ == "__main__":
    app.run()
