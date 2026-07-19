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
    import torch.nn.functional as F

    from labs.common import checks

    return F, checks, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 0 — The idea: deformable attention in pure PyTorch

    The ten labs after this one rebuild this repo's GPU kernel from nothing. This
    lab is the prequel that answers the question the kernel takes for granted:
    **what *is* multi-scale deformable attention, and why does it exist?**

    You'll build three operators, each a small mutation of the previous one, in
    plain PyTorch — `nn.Linear`, tensor indexing, autograd. No Triton, no GPU:

    1. **Deformable attention** — attention that *predicts where to look* instead
       of comparing its query against every position;
    2. **Deformable cross-attention** — the same idea when the queries are a
       handful of learned object queries instead of pixels (the Deformable DETR
       decoder);
    3. **Multi-scale deformable attention (MSDA)** — the same idea across a
       feature pyramid: the exact operator this repo ships as a Triton kernel.

    **Assumed:** standard attention ($\mathrm{softmax}(QK^\top/\sqrt{d})\,V$) and
    convolution. **Not assumed:** any prior contact with anything deformable.

    Every exercise cell is followed by a checker cell — edit your code and it
    re-runs instantly (✅ passed / ❌ wrong / 🚧 not attempted) — plus collapsible
    hints that escalate from a nudge to near-code. Fight before peeking. The
    final checks grade your MSDA against the same reference implementation this
    repo's own pytest suite trusts.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Shared SVG drawing helpers for this lab's illustrations.
    PAL = dict(
        ink="#334155", faint="#cbd5e1", grid="#94a3b8",
        blue="#2563eb", lightblue="#dbeafe", orange="#ea580c",
        lightorange="#ffedd5", card="#ffffff", edge="#e2e8f0",
    )

    def svg_wrap(inner, w, h):
        """Wrap SVG elements in a light 'card' so it reads on any theme."""
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'font-family="system-ui, sans-serif" font-size="11">'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="8" '
            f'fill="{PAL["card"]}" stroke="{PAL["edge"]}"/>' + inner + "</svg>"
        )

    def svg_grid(x0, y0, cols, rows, cell, color, width=0.7):
        parts = []
        for c in range(cols + 1):
            x = x0 + c * cell
            parts.append(
                f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + rows * cell}" '
                f'stroke="{color}" stroke-width="{width}"/>'
            )
        for r in range(rows + 1):
            y = y0 + r * cell
            parts.append(
                f'<line x1="{x0}" y1="{y}" x2="{x0 + cols * cell}" y2="{y}" '
                f'stroke="{color}" stroke-width="{width}"/>'
            )
        return "".join(parts)

    def svg_dot(x, y, r, color, opacity=1.0):
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                f'fill="{color}" fill-opacity="{opacity}"/>')

    def svg_rect(x, y, w, h, fill, opacity=1.0, rx=0):
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                f'height="{h:.1f}" rx="{rx}" fill="{fill}" '
                f'fill-opacity="{opacity}"/>')

    def svg_text(x, y, s, color, size=11, anchor="middle", bold=False):
        weight = ' font-weight="600"' if bold else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" '
                f'font-size="{size}" text-anchor="{anchor}"{weight}>{s}</text>')

    def svg_arrow(x1, y1, x2, y2, color, width=1.5, opacity=1.0, trim=0.0):
        """Arrow from (x1, y1) to (x2, y2); `trim` stops it short of the tip
        (e.g. a dot's radius) so the head stays visible next to markers."""
        import math

        ang = math.atan2(y2 - y1, x2 - x1)
        if trim:
            t = min(trim, max(math.hypot(x2 - x1, y2 - y1) - 5, 0))
            x2, y2 = x2 - t * math.cos(ang), y2 - t * math.sin(ang)
        hx, hy = x2 - 6 * math.cos(ang), y2 - 6 * math.sin(ang)
        lt = (hx - 3.4 * math.sin(ang), hy + 3.4 * math.cos(ang))
        rt = (hx + 3.4 * math.sin(ang), hy - 3.4 * math.cos(ang))
        return (
            f'<g stroke="{color}" fill="{color}" opacity="{opacity}">'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" '
            f'stroke-width="{width}"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {lt[0]:.1f},{lt[1]:.1f} '
            f'{rt[0]:.1f},{rt[1]:.1f}" stroke="none"/></g>'
        )

    return PAL, svg_arrow, svg_dot, svg_grid, svg_rect, svg_text, svg_wrap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Why "deformable"?

    Put standard attention on an image and watch the cost. A detection backbone
    at stride 8 turns an $800 \times 800$ image into a $100 \times 100$ feature
    map: $S = 10{,}000$ positions. Self-attention compares every query with every
    key — $O(Q \cdot S)$ dot products and just as many softmax weights. At
    $Q = S$ that is $10^8$ query–key pairs *per head, per layer, per image* — and
    attention maps on images are overwhelmingly near-zero: a pixel on a car cares
    about the car, not about every patch of sky. The original DETR pays this
    price and famously needs ~500 epochs, much of it spent just *learning where
    to put* its attention.

    Convolution has the opposite character. A $3 \times 3$ kernel reads 9
    neighbors — cheap, local, and **rigid**: the sampling grid is identical at
    every location, for every image, forever. Content can modulate *how much*
    each tap contributes (the kernel weights are learned), but never *where* the
    taps are.

    Deformable attention is the hybrid (lineage: Deformable ConvNets 2017 →
    Deformable DETR 2020). Each query reads only $K \approx 4$ points, like a
    tiny convolution — but the sampling *locations* and their *weights* are both
    **predicted from the query's own feature vector** by linear layers, like
    attention at its most content-dependent:
    """)
    return


@app.cell(hide_code=True)
def _(PAL, mo, svg_arrow, svg_dot, svg_grid, svg_rect, svg_text, svg_wrap):
    _cell, _cols, _rows, _oy = 16, 9, 7, 36
    _titles = ["convolution", "attention", "deformable attention"]
    _subs = ["fixed 3×3 grid", "reads all S positions", "K predicted points"]
    _parts = []
    for _p, _ox in enumerate((24, 216, 408)):
        _qc, _qr = 4, 3
        _qx = _ox + (_qc + 0.5) * _cell
        _qy = _oy + (_qr + 0.5) * _cell
        if _p == 0:
            for _r in range(_qr - 1, _qr + 2):
                for _c in range(_qc - 1, _qc + 2):
                    _parts.append(svg_rect(_ox + _c * _cell, _oy + _r * _cell,
                                           _cell, _cell, PAL["lightblue"]))
        if _p == 1:
            for _r in range(_rows):
                for _c in range(_cols):
                    _a = 0.04 + 0.30 * (((_c * 7 + _r * 13) % 10) / 10)
                    _parts.append(svg_rect(_ox + _c * _cell, _oy + _r * _cell,
                                           _cell, _cell, PAL["blue"],
                                           opacity=_a))
        _parts.append(svg_grid(_ox, _oy, _cols, _rows, _cell, PAL["faint"]))
        if _p == 2:
            for _dx, _dy, _w in ((2.6, -1.8, 0.45), (-2.4, 1.6, 0.25),
                                 (3.3, 2.1, 0.20), (-1.0, -2.6, 0.10)):
                _sx, _sy = _qx + _dx * _cell, _qy + _dy * _cell
                _parts.append(svg_arrow(_qx, _qy, _sx, _sy, PAL["orange"],
                                        trim=3 + 8 * _w + 1.5))
                _parts.append(svg_dot(_sx, _sy, 3 + 8 * _w, PAL["orange"], 0.85))
        _parts.append(svg_dot(_qx, _qy, 4.5, PAL["blue"]))
        _cxm = _ox + _cols * _cell / 2
        _parts.append(svg_text(_cxm, _oy - 14, _titles[_p], PAL["ink"], bold=True))
        _parts.append(svg_text(_cxm, _oy + _rows * _cell + 18, _subs[_p],
                               PAL["grid"]))
    mo.Html(svg_wrap("".join(_parts), 576, 192))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *One query (blue). Convolution reads a fixed grid; attention reads
    everything; deformable attention reads a few points it chose itself, each
    with a predicted weight (dot size).*

    | | where it reads | how weights arise | cost per query |
    |---|---|---|---|
    | convolution | fixed grid around the position | learned constants, same everywhere | $k^2$ |
    | attention | every position | $\mathrm{softmax}(q \cdot k)$ — *computed* by comparison | $S$ |
    | deformable attention | $K$ predicted points | softmax of a *predicted* logit — no comparison | $K$ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The sentence to internalize:** in deformable attention there are **no
    query–key dot products.** A linear layer maps the query vector directly to
    $K$ sampling offsets and $K$ weight logits. Both *where* and *how much* are
    guesses by the network, trained end-to-end — and the only reason gradients
    can train the *where* is the subject of the next section.
    """).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The foundation: a differentiable read

    An offset comes out of `nn.Linear`, so it is a continuous value — a query
    might ask for the feature at $(2.37,\ 1.58)$. Two problems: the image only
    has values at integer pixels, and — worse — training the offset predictor
    requires $\partial\,\text{output} / \partial\,\text{location}$ to exist. If
    sampling meant `img[round(y), round(x)]`, that derivative would be zero
    almost everywhere and the network could never learn *where* to look.

    **Bilinear interpolation** solves both. For a point $(x, y)$ with
    $x_0 = \lfloor x \rfloor$, $y_0 = \lfloor y \rfloor$ and fractional parts
    $l_x = x - x_0$, $l_y = y - y_0$, blend the 4 surrounding pixels by
    proximity:

    $$
    \text{sample} =
    (1{-}l_x)(1{-}l_y)\,v_{00} + l_x(1{-}l_y)\,v_{01} +
    (1{-}l_x)\,l_y\,v_{10} + l_x l_y\,v_{11}
    $$

    where $v_{00}$ is the pixel at $(y_0, x_0)$, $v_{01}$ at $(y_0, x_0{+}1)$,
    etc. The four weights sum to 1, and each is piecewise-linear in $x$ and $y$
    — so the sample varies *continuously* as the point moves, and autograd can
    push gradients into the coordinates themselves. Drag the point (past the
    border, too):
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bilin_x = mo.ui.slider(-0.75, 4.75, step=0.05, value=1.7, label="x")
    bilin_y = mo.ui.slider(-0.75, 3.75, step=0.05, value=1.3, label="y")
    mo.hstack([bilin_x, bilin_y], justify="start")
    return bilin_x, bilin_y


@app.cell(hide_code=True)
def _(PAL, bilin_x, bilin_y, mo, svg_dot, svg_rect, svg_text, svg_wrap):
    import math as _math

    _W, _H, _cell, _ox, _oy = 5, 4, 44, 56, 52
    _cw, _ch = 2 * _ox + (_W - 1) * _cell, 2 * _oy + (_H - 1) * _cell
    _x, _y = bilin_x.value, bilin_y.value
    _x0, _y0 = _math.floor(_x), _math.floor(_y)
    _lx, _ly = _x - _x0, _y - _y0
    _corners = [
        (_x0, _y0, (1 - _lx) * (1 - _ly), "v00"),
        (_x0 + 1, _y0, _lx * (1 - _ly), "v01"),
        (_x0, _y0 + 1, (1 - _lx) * _ly, "v10"),
        (_x0 + 1, _y0 + 1, _lx * _ly, "v11"),
    ]
    # shade the unit cell the point lives in — its 4 corners do the blending
    _parts = [svg_rect(_ox + _x0 * _cell, _oy + _y0 * _cell, _cell, _cell,
                       PAL["lightblue"], opacity=0.55)]
    for _r in range(_H):
        for _c in range(_W):
            _parts.append(svg_dot(_ox + _c * _cell, _oy + _r * _cell, 2.2,
                                  PAL["faint"]))
    # image border: pixel edges run half a cell beyond the centers
    _parts.append(
        f'<rect x="{_ox - _cell / 2}" y="{_oy - _cell / 2}" '
        f'width="{_W * _cell}" height="{_H * _cell}" fill="none" '
        f'stroke="{PAL["grid"]}" stroke-dasharray="3 3"/>'
    )
    _px, _py = _ox + _x * _cell, _oy + _y * _cell
    for _i, (_cx, _cy, _w, _n) in enumerate(_corners):
        _sx, _sy = _ox + _cx * _cell, _oy + _cy * _cell
        _inside = 0 <= _cx < _W and 0 <= _cy < _H
        _parts.append(
            f'<line x1="{_px:.1f}" y1="{_py:.1f}" x2="{_sx}" y2="{_sy}" '
            f'stroke="{PAL["faint"]}" stroke-dasharray="2 3"/>'
        )
        _parts.append(svg_dot(_sx, _sy, 3 + 14 * _w,
                              PAL["orange"] if _inside else PAL["faint"], 0.75))
        # weight label diagonally outward from the cell (off the connector
        # lines), flipped back inward when it would leave the card
        _right, _bottom = _i % 2 == 1, _i // 2 == 1
        _tx, _anchor = (_sx + 7, "start") if _right else (_sx - 7, "end")
        if _tx < 32:
            _tx, _anchor = _sx + 7, "start"
        elif _tx > _cw - 32:
            _tx, _anchor = _sx - 7, "end"
        _ty = _sy + 16 if _bottom else _sy - 9
        if _ty > _ch - 6:
            _ty = _sy - 9
        elif _ty < 14:
            _ty = _sy + 16
        _parts.append(svg_text(_tx, _ty, f"{_w:.2f}", PAL["ink"], size=10,
                               anchor=_anchor))
    _parts.append(svg_dot(_px, _py, 4, PAL["blue"]))
    _lx_lbl, _la = ((_px - 9, "end") if _px > _cw - 92
                    else (_px + 9, "start"))
    _ly_lbl = _py - 10 if _py > _ch - 24 else _py + 15
    _parts.append(svg_text(_lx_lbl, _ly_lbl, f"({_x:.2f}, {_y:.2f})",
                           PAL["blue"], size=10, anchor=_la, bold=True))
    _in_sum = sum(_w for _cx, _cy, _w, _n in _corners
                  if 0 <= _cx < _W and 0 <= _cy < _H)
    _rows = "\n".join(
        f"| `{_n}` ({_cx}, {_cy}) | {_w:.3f} | "
        + ("✓" if 0 <= _cx < _W and 0 <= _cy < _H else "✗ → reads 0")
        + " |"
        for _cx, _cy, _w, _n in _corners
    )
    _tbl = mo.md(
        f"$l_x = {_lx:.2f}$, $l_y = {_ly:.2f}$\n\n"
        f"| corner (x, y) | weight | in image |\n|---|---|---|\n{_rows}\n"
        f"| **Σ in-image** | **{_in_sum:.3f}** | |"
    )
    mo.hstack([mo.Html(svg_wrap("".join(_parts), _cw, _ch)), _tbl],
              justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two conventions, shared by this repo, by `F.grid_sample(align_corners=False)`
    and by every later lab — get either wrong and nothing will ever match:

    1. **Pixel centers at integers.** Pixel (row $i$, col $j$) has its value *at*
       the continuous point $(x{=}j,\ y{=}i)$. (Normalized $[0,1]$ coordinates
       enter in section 4 — for now everything is in pixels.)
    2. **Zeros padding.** A corner outside the image contributes zero — it is
       *not* clamped to the border. The sample fades smoothly to zero as the
       point leaves the image (watch the dashed border above), so a predicted
       offset that wanders off the map is harmless, and gradients can pull it
       back.

    ### ✏️ Exercise 1 — `bilinear_sample` via `grid_sample`

    Sample an `(H, W, D)` image at `points` — an `(N, 2)` tensor of continuous
    `(x, y)` **pixel** coordinates. Return `(N, D)`. The whole point of this
    exercise is to reach for the native primitive: implement it with a single
    `F.grid_sample` call. Rules:

    - **No Python loop over N**, no hand-rolled corner arithmetic — one
      `F.grid_sample` call does it. (The math above is the *why*; `grid_sample`
      is the *how*, and it's what every PyTorch fallback in this family uses.)
    - **Stay differentiable w.r.t. `points`**: `grid_sample` is differentiable
      w.r.t. the grid, so just build the grid from `points` with tensor ops —
      don't route coordinates through `int()` or `.item()`. A checker verifies
      gradients actually reach `points`.
    - Out-of-bounds corners contribute zero — `padding_mode="zeros"` gives you
      that for free.
    """)
    return


@app.cell
def _(checks):
    def bilinear_sample(img, points):
        """img: (H, W, D); points: (N, 2) of (x, y) pixel coords. -> (N, D)"""
        import torch.nn.functional as F
        H, W, D = img.shape
        # ================= YOUR CODE =================
        # One F.grid_sample call does all of it. Steps:
        # 1. pixel coords -> grid_sample's [-1, 1] (align_corners=False):
        #    grid_x = 2 * (points[:, 0] + 0.5) / W - 1   (same for y with H)
        # 2. stack to (x, y) and shape for a 4-D input (1, D, H, W):
        #    grid = grid[:, None, None, :][None]      # (1, N, 1, 2)
        # 3. v = img.permute(2, 0, 1)[None]        # (1, D, H, W)
        # 4. sampled = F.grid_sample(v, grid, mode="bilinear",
        #    padding_mode="zeros", align_corners=False)
        # 5. return sampled[0, :, :, 0].T   # (N, D)
        raise checks.NotDoneYet()
        # =============================================
    return (bilinear_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint 1 — pixel → grid coords": mo.md(
            "`F.grid_sample` samples a 4-D input `(N, C, H, W)` at a grid "
            "`(N, H_out, W_out, 2)` in `[-1, 1]`. With one point per row, "
            "`H_out = W_out = 1`. The `align_corners=False` map is "
            "`x_im = ((g + 1) * W - 1) / 2 = x * W - 0.5`, so invert it:\n"
            "```python\nscale = points.new_tensor([W, H])\n"
            "grid = 2 * (points + 0.5) / scale - 1   # (N, 2), (x, y)\n"
            "grid = grid[:, None, None, :][None]      # (1, N, 1, 2)\n```\n"
            "(the leading `[None]` matches the image's batch of 1 below)."
        ),
        "Hint 2 — the call": mo.md(
            "`img` is `(H, W, D)`; `grid_sample` wants `(N, C, H, W)`:\n"
            "```python\nv = img.permute(2, 0, 1)[None]        # (1, D, H, W)\n"
            "sampled = F.grid_sample(\n"
            "    v, grid, mode=\"bilinear\", padding_mode=\"zeros\",\n"
            "    align_corners=False,\n"
            ")                                  # (1, D, N, 1)\n"
            "return sampled[0, :, :, 0].T            # (N, D)\n```\n"
            "`padding_mode=\"zeros\"` gives zeros-padding for free (corners "
            "outside the image contribute zero), and `align_corners=False` "
            "places pixel centers at integers — both conventions the repo "
            "uses everywhere. `grid_sample` is differentiable w.r.t. the "
            "grid, so gradients reach `points` with no extra work."
        ),
        "Stuck?": mo.md(
            "The reference solution is `labs/solutions/lab00.py` — but read "
            "the `grid_sample` docs first; this function is the load-bearing "
            "wall of everything below (and of labs 1–10), so fight for it first."
        ),
    })
    return


@app.cell(hide_code=True)
def _(F, bilinear_sample, checks, torch):
    def _pix_to_grid(pts, H, W):
        # pixel coords -> grid_sample(align_corners=False) coords
        gx = 2 * (pts[:, 0] + 0.5) / W - 1
        gy = 2 * (pts[:, 1] + 0.5) / H - 1
        return torch.stack([gx, gy], dim=-1)

    def _matches_grid_sample():
        _g = torch.Generator().manual_seed(0)
        _img = torch.randn(5, 7, 3, generator=_g, dtype=torch.float64)
        _pts = torch.rand(40, 2, generator=_g, dtype=torch.float64)
        _pts = _pts * torch.tensor([10.0, 8.0], dtype=torch.float64) - 1.5
        _ref = F.grid_sample(
            _img.permute(2, 0, 1)[None],
            _pix_to_grid(_pts, 5, 7)[None, :, None],
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0, :, :, 0].T
        torch.testing.assert_close(bilinear_sample(_img, _pts), _ref)

    def _integer_coords_exact():
        _img = torch.arange(24.).reshape(4, 6, 1)
        _pts = torch.tensor([[2.0, 1.0], [0.0, 0.0], [5.0, 3.0]])
        _expected = torch.stack([_img[1, 2], _img[0, 0], _img[3, 5]])
        torch.testing.assert_close(bilinear_sample(_img, _pts), _expected)

    def _grad_flows_to_points():
        _g = torch.Generator().manual_seed(1)
        _img = torch.randn(4, 5, 2, generator=_g)
        _pts = (torch.rand(6, 2, generator=_g) * 3).requires_grad_(True)
        bilinear_sample(_img, _pts).sum().backward()
        assert _pts.grad is not None, "no gradient reached `points`"
        assert torch.isfinite(_pts.grad).all(), "non-finite gradient on points"
        assert _pts.grad.abs().sum() > 0, (
            "gradient on `points` is all zeros — did coordinates go through "
            "int()/.item()/round()?"
        )

    checks.run_checks({
        "matches grid_sample (incl. out-of-bounds points)": _matches_grid_sample,
        "on-pixel sample returns that pixel exactly": _integer_coords_exact,
        "gradients flow to the sampling coordinates": _grad_flows_to_points,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That third check is the one that makes deformable attention *trainable*:
    the sample's derivative w.r.t. the coordinate is exactly the difference of
    neighboring pixels (a local image gradient), so "move a little to the right"
    has a well-defined effect on the loss. This gradient path is why the offset
    predictors you're about to build can learn at all.

    ## 3. Deformable attention

    Now the operator around the read. Take one feature map $x$ of shape
    $(B, H, W, C)$ and treat *every position* as a query — this is
    self-attention, deformable style. For the query at location $p_q$ with
    feature vector $z_q$ (Deformable DETR, eq. 2):

    $$
    \mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \Big[
    \sum_{k=1}^{K} A_{mqk} \cdot W'_m\, x\big(p_q + \Delta p_{mqk}\big) \Big]
    $$

    Read it inside-out:

    - $W'_m x$ — a linear layer projects the map to per-head **values**
      ($M$ heads $\times$ $D = C/M$ channels): `value_proj`;
    - $\Delta p_{mqk} = \mathrm{Linear}(z_q)$ — each head predicts $K$
      **offsets** in pixels, relative to $p_q$: `offset_proj`, output size
      $M \cdot K \cdot 2$;
    - $A_{mqk} = \mathrm{softmax}_k(\mathrm{Linear}(z_q))$ — each head predicts
      $K$ **weights**, normalized over its own $K$ points: `weight_proj`,
      output size $M \cdot K$;
    - $x(\,\cdot\,)$ — your `bilinear_sample`, because $p_q + \Delta p$ is
      fractional;
    - $W_m$ — concatenate heads, project out: `out_proj`.

    Same skeleton as the attention you know — values, weights, weighted sum,
    output projection — but the sum runs over $K$ sampled points instead of $S$
    keys, and $A$ comes from the query alone. Here is what one *untrained* head
    looks like; re-roll the initialization a few times:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    head_seed = mo.ui.slider(0, 9, step=1, value=3, label="re-roll the head (seed)")
    head_seed
    return (head_seed,)


@app.cell(hide_code=True)
def _(
    PAL,
    head_seed,
    mo,
    svg_arrow,
    svg_dot,
    svg_grid,
    svg_rect,
    svg_text,
    svg_wrap,
    torch,
):
    _cell, _cols, _rows, _ox, _oy = 18, 10, 8, 26, 34
    _g = torch.Generator().manual_seed(int(head_seed.value))
    _offsets = torch.randn(4, 2, generator=_g) * 1.9
    _w = torch.randn(4, generator=_g).softmax(dim=0)
    # A self-attention query belongs to an actual pixel. Pixel (col=4, row=3)
    # is drawn at the center of its grid cell, not at an arbitrary fractional
    # position.
    _qc, _qr = 4, 3
    _qx, _qy = _ox + (_qc + 0.5) * _cell, _oy + (_qr + 0.5) * _cell
    _parts = [
        svg_rect(
            _ox + _qc * _cell, _oy + _qr * _cell,
            _cell, _cell, PAL["lightblue"],
        ),
        svg_grid(_ox, _oy, _cols, _rows, _cell, PAL["faint"]),
    ]
    import math as _math

    for _k in range(4):
        _dx, _dy = float(_offsets[_k, 0]), float(_offsets[_k, 1])
        _a = float(_w[_k])
        _r = 3 + 11 * _a
        _sx = _qx + _dx * _cell
        _sy = _qy + _dy * _cell
        _parts.append(svg_arrow(_qx, _qy, _sx, _sy, PAL["orange"],
                                trim=_r + 1.5))
        _parts.append(svg_dot(_sx, _sy, _r, PAL["orange"], 0.85))
        # tiny k index just past each dot (along the arrow direction) so it
        # can be matched to the legend at a glance
        _ang = _math.atan2(_sy - _qy, _sx - _qx)
        _parts.append(svg_text(_sx + (_r + 7) * _math.cos(_ang),
                               _sy + (_r + 7) * _math.sin(_ang) + 3,
                               str(_k), PAL["ink"], size=8))
        # Keep full labels in a stable legend rather than letting them collide
        # with sampling points as the seed changes.
        _parts.append(svg_dot(226, 58 + 25 * _k, 3 + 7 * _a,
                              PAL["orange"], 0.85))
        _parts.append(svg_text(
            238, 61 + 25 * _k,
            f"k={_k}: Δ=({_dx:+.1f}, {_dy:+.1f}), A={_a:.2f}",
            PAL["ink"], size=9, anchor="start",
        ))
    _parts.append(svg_dot(_qx, _qy, 4.5, PAL["blue"]))
    _parts.append(svg_text(_qx - 8, _qy - 8, "p_q", PAL["blue"], anchor="end",
                           bold=True))
    _parts.append(svg_text(_ox + _cols * _cell / 2, _oy - 14,
                           "query pixel (4, 3) + predicted offsets",
                           PAL["ink"], bold=True))
    _parts.append(svg_text(226, 35, "one untrained head", PAL["ink"],
                           anchor="start", bold=True))
    _parts.append(svg_text(
        226, 174, "dot size = attention weight", PAL["grid"],
        size=9, anchor="start",
    ))
    mo.Html(svg_wrap("".join(_parts), 410, _oy + _rows * _cell + 22))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every re-roll is a different randomly-initialized head. Training reshapes
    these patterns: in trained detectors, heads specialize — some hug object
    extremities, some reach out for context.

    ### ✏️ Exercise 2 — `deform_attend`: the core

    Just the inner bracket — one head, one image, no learned parts:
    `value (H, W, D)`, `points (Q, K, 2)` (reference + offset already summed,
    pixel coords), `weights (Q, K)` → `(Q, D)`. Two or three lines on top of
    your `bilinear_sample`.
    """)
    return


@app.cell
def _(bilinear_sample, checks):
    def deform_attend(value, points, weights):
        """out[q] = sum_k weights[q, k] * bilinear_sample(value, points[q, k]).

        value: (H, W, D); points: (Q, K, 2) pixel coords; weights: (Q, K).
        Returns (Q, D).
        """
        _ = bilinear_sample  # you'll want this
        # ================= YOUR CODE =================
        raise checks.NotDoneYet()
        # =============================================
    return (deform_attend,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint": mo.md(
            "`bilinear_sample` wants `(N, 2)` points: reshape `(Q, K, 2)` to "
            "`(Q*K, 2)`, sample, reshape the result back to `(Q, K, D)`, then "
            "a weighted sum over the K axis (`unsqueeze` the weights)."
        ),
    })
    return


@app.cell(hide_code=True)
def _(F, checks, deform_attend, torch):
    def _pix_to_grid(pts, H, W):
        gx = 2 * (pts[:, 0] + 0.5) / W - 1
        gy = 2 * (pts[:, 1] + 0.5) / H - 1
        return torch.stack([gx, gy], dim=-1)

    def _matches_grid_sample():
        _g = torch.Generator().manual_seed(2)
        _value = torch.randn(6, 5, 4, generator=_g, dtype=torch.float64)
        _pts = torch.rand(3, 2, 2, generator=_g, dtype=torch.float64)
        _pts = _pts * torch.tensor([7.0, 8.0], dtype=torch.float64) - 1.0
        _w = torch.rand(3, 2, generator=_g, dtype=torch.float64).softmax(dim=-1)
        _smp = F.grid_sample(
            _value.permute(2, 0, 1)[None],
            _pix_to_grid(_pts.reshape(6, 2), 6, 5)[None, :, None],
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0, :, :, 0].T.reshape(3, 2, 4)
        _expected = (_w.unsqueeze(-1) * _smp).sum(dim=1)
        torch.testing.assert_close(deform_attend(_value, _pts, _w), _expected)

    def _linear_in_weights():
        _g = torch.Generator().manual_seed(3)
        _value = torch.randn(4, 4, 3, generator=_g, dtype=torch.float64)
        _pts = torch.rand(5, 3, 2, generator=_g, dtype=torch.float64) * 3
        _w = torch.rand(5, 3, generator=_g, dtype=torch.float64)
        torch.testing.assert_close(
            deform_attend(_value, _pts, 2 * _w),
            2 * deform_attend(_value, _pts, _w),
        )

    checks.run_checks({
        "weighted samples match grid_sample ground truth": _matches_grid_sample,
        "output is linear in the attention weights": _linear_in_weights,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ✏️ Exercise 3 — the layer

    Now wrap the core in the learned machinery:

    ```python
    DeformableAttention(embed_dim, num_heads, num_points)    # C, M, K
    forward(x: (B, H, W, C)) -> (B, H, W, C)
    ```

    Requirements (the checkers depend on the exact names):

    - four `nn.Linear` submodules — `value_proj` (C→C), `offset_proj`
      (C→M·K·2), `weight_proj` (C→M·K), `out_proj` (C→C);
    - reference points: each position's own location, $(x{=}j,\ y{=}i)$;
    - weights softmaxed over $K$ within each head;
    - a Python loop over `(b, m)` calling your `deform_attend` is perfectly
      fine — the checkers use toy sizes. (Fully vectorizing is a nice stretch
      goal, not a requirement.)

    Before you code, work out what the **collapse** check below asserts and
    why it must hold: zero out `offset_proj` *and* `weight_proj` (weights and
    biases). Every offset becomes $(0,0)$, so all $K$ samples land exactly on
    the query's own pixel — where bilinear interpolation returns the pixel
    exactly. All logits become $0$, so the softmax is uniform, and the average
    of $K$ identical samples is the sample. The whole layer must therefore
    collapse to `out_proj(value_proj(x))`. If yours disagrees, one of your
    conventions is off.
    """)
    return


@app.cell
def _(checks, deform_attend, torch):
    class DeformableAttention(torch.nn.Module):
        """Single-scale deformable self-attention. See the contract above."""

        def __init__(self, embed_dim, num_heads, num_points):
            super().__init__()
            # ================= YOUR CODE =================
            # the four projections (exact names above), plus bookkeeping
            raise checks.NotDoneYet()
            # =============================================

        def forward(self, x):
            _ = deform_attend  # you'll want this
            # ================= YOUR CODE =================
            # 1. project values; predict offsets and weights (softmax over K)
            # 2. reference grid: position (i, j) sits at (x=j, y=i)
            # 3. per (b, m): deform_attend, then reassemble heads, out_proj
            raise checks.NotDoneYet()
            # =============================================
    return (DeformableAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint 1 — the reference grid": mo.md(
            "```python\nys, xs = torch.meshgrid(\n"
            "    torch.arange(H, dtype=x.dtype, device=x.device),\n"
            "    torch.arange(W, dtype=x.dtype, device=x.device),\n"
            "    indexing=\"ij\",\n)\n"
            "ref = torch.stack([xs, ys], dim=-1).reshape(H * W, 2)\n```\n"
            "Note the order: the last axis is `(x, y)`, i.e. `(col, row)`."
        ),
        "Hint 2 — shapes to aim for": mo.md(
            "`value → (B, H, W, M, D)`; `offsets → (B, H·W, M, K, 2)`; "
            "`weights → (B, H·W, M, K)` after `softmax(dim=-1)`; sampling "
            "points `= ref[None, :, None, None, :] + offsets`. Then per "
            "`(b, m)`: `deform_attend(value[b, :, :, m], points[b, :, m], "
            "weights[b, :, m])` gives `(H·W, D)`; stack, reassemble to "
            "`(B, H, W, C)`, `out_proj`."
        ),
        "Stuck?": mo.md("`labs/solutions/lab00.py` — but the next two "
                        "exercises reuse this exact structure, so understand "
                        "every line you take."),
    })
    return


@app.cell(hide_code=True)
def _(DeformableAttention, checks, torch):
    def _make():
        torch.manual_seed(0)
        return DeformableAttention(8, 2, 3)

    def _zero_predictors(layer):
        with torch.no_grad():
            layer.offset_proj.weight.zero_()
            layer.offset_proj.bias.zero_()
            layer.weight_proj.weight.zero_()
            layer.weight_proj.bias.zero_()

    def _shape():
        _layer = _make()
        assert _layer(torch.randn(2, 5, 6, 8)).shape == (2, 5, 6, 8)

    def _collapse():
        _layer = _make()
        _zero_predictors(_layer)
        _x = torch.randn(2, 5, 6, 8)
        torch.testing.assert_close(
            _layer(_x), _layer.out_proj(_layer.value_proj(_x)),
            rtol=1e-5, atol=1e-6,
        )

    def _offsets_learn():
        _layer = _make()
        _x = torch.randn(1, 4, 5, 8, requires_grad=True)
        _layer(_x).sum().backward()
        _g = _layer.offset_proj.weight.grad
        assert _g is not None and _g.abs().sum() > 0, (
            "no gradient reached offset_proj — the layer can't learn where "
            "to look"
        )
        assert _x.grad is not None and _x.grad.abs().sum() > 0

    checks.run_checks({
        "output shape is (B, H, W, C)": _shape,
        "zeroed predictors collapse the layer to out∘value": _collapse,
        "gradients reach the offset predictor": _offsets_learn,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Deformable cross-attention

    In a DETR-style decoder the queries are not pixels. They are $Q \approx 300$
    learned **object queries** — embeddings, each responsible for finding at
    most one object. They live in no particular place on the image, so the
    layer becomes *cross*-attention: queries on one side, the encoder's feature
    map on the other. Exactly two things change from exercise 3:

    1. **Each query carries a reference point** $p_q \in [0,1]^2$ — an explicit
       guess at "where my object is", supplied *to* the layer. (In Deformable
       DETR it is predicted from the query embedding by a small head, then
       refined layer by layer into the box center; the attention layer itself
       just receives it.)
    2. **Offsets and weights are predicted from the query embedding** $z_q$,
       not from map features. The map only supplies values.
    """)
    return


@app.cell(hide_code=True)
def _(PAL, mo, svg_arrow, svg_dot, svg_grid, svg_rect, svg_text, svg_wrap):
    _parts = []
    _qx, _qw, _qh = 28, 104, 26
    _qys = (52, 100, 148)
    _parts.append(svg_text(_qx + _qw / 2, 34, "object queries", PAL["ink"],
                           bold=True))
    for _i, _qy in enumerate(_qys):
        _hl = _i == 1
        _parts.append(svg_rect(_qx, _qy, _qw, _qh,
                               PAL["lightblue"] if _hl else "#f1f5f9", rx=6))
        _parts.append(
            f'<rect x="{_qx}" y="{_qy}" width="{_qw}" height="{_qh}" rx="6" '
            f'fill="none" stroke="{PAL["blue"] if _hl else PAL["faint"]}"/>'
        )
        _parts.append(svg_text(_qx + _qw / 2, _qy + 17, f"z{_i}", PAL["ink"]))
    _ox, _oy, _cell, _cols, _rows = 300, 44, 14, 16, 11
    _parts.append(svg_text(_ox + _cols * _cell / 2, 34, "encoder feature map",
                           PAL["ink"], bold=True))
    _parts.append(svg_grid(_ox, _oy, _cols, _rows, _cell, PAL["faint"]))
    for _i, (_rx, _ry) in enumerate(((3.0, 2.2), (8.6, 5.8), (13.2, 9.0))):
        _hl = _i == 1
        _px, _py = _ox + _rx * _cell, _oy + _ry * _cell
        _parts.append(svg_arrow(_qx + _qw, _qys[_i] + _qh / 2, _px, _py,
                                PAL["blue"], opacity=1.0 if _hl else 0.35,
                                trim=(4 if _hl else 3) + 2))
        _parts.append(svg_dot(_px, _py, 4 if _hl else 3, PAL["blue"],
                              1.0 if _hl else 0.5))
        if _hl:
            _parts.append(svg_text(_px + 8, _py - 8, "reference point",
                                   PAL["blue"], size=10, anchor="start",
                                   bold=True))
            for _dx, _dy, _w in ((1.8, -1.5, 0.4), (-1.4, 1.2, 0.3),
                                 (2.3, 1.7, 0.2), (-2.0, -1.0, 0.1)):
                _sx, _sy = _px + _dx * _cell, _py + _dy * _cell
                _parts.append(svg_arrow(_px, _py, _sx, _sy, PAL["orange"],
                                        width=1.2, trim=2.5 + 7 * _w + 1.5))
                _parts.append(svg_dot(_sx, _sy, 2.5 + 7 * _w, PAL["orange"],
                                      0.85))
    mo.Html(svg_wrap("".join(_parts), 560, 220))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Normalized coordinates.** Reference points arrive normalized to $[0, 1]$
    — "63% across, 41% down" — instead of pixels, which makes them
    resolution-independent (remember this: it is what will make *multi-scale*
    possible in the next section). The mapping back to pixels, used by this
    repo and by `grid_sample(align_corners=False)`:

    $$x_\text{pix} = x \cdot W - 0.5, \qquad y_\text{pix} = y \cdot H - 0.5$$

    Why $-0.5$: normalized coordinates measure the image *edge to edge*, and
    pixel $j$'s center sits $(j + 0.5)/W$ of the way across — so $x{=}0$ is
    half a pixel left of pixel 0's center and $x{=}1$ half a pixel right of the
    last pixel's center. Burn this into memory; it recurs in every later lab.

    ### ✏️ Exercise 4 — `DeformableCrossAttention`

    ```python
    DeformableCrossAttention(embed_dim, num_heads, num_points)
    forward(query: (B, Q, C), reference_points: (B, Q, 2) in [0, 1],
            value: (B, H, W, C)) -> (B, Q, C)
    ```

    Same four submodule names as exercise 3. Map the reference points to pixel
    coordinates, add the predicted offsets (in pixels), and everything after
    that you already built. One new demand from the checkers: **gradients must
    reach `reference_points`** — in a real detector that gradient path is what
    trains the box-refinement heads.
    """)
    return


@app.cell
def _(checks, deform_attend, torch):
    class DeformableCrossAttention(torch.nn.Module):
        """Object queries sample a feature map around their reference points.

        See the contract above; predictions come from `query`, values from
        `value`, reference points are normalized to [0, 1].
        """

        def __init__(self, embed_dim, num_heads, num_points):
            super().__init__()
            # ================= YOUR CODE =================
            raise checks.NotDoneYet()
            # =============================================

        def forward(self, query, reference_points, value):
            _ = deform_attend  # you'll want this
            # ================= YOUR CODE =================
            raise checks.NotDoneYet()
            # =============================================
    return (DeformableCrossAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint — what actually changes from exercise 3": mo.md(
            "Three edits: `offset_proj`/`weight_proj` are applied to `query` "
            "instead of `x`; the reference grid is replaced by\n"
            "```python\nref_pix = reference_points * query.new_tensor([W, H]) - 0.5\n```\n"
            "(note the `(x, y)` ↔ `(W, H)` pairing); and the output has Q "
            "positions instead of H·W. The `(b, m)` loop over `deform_attend` "
            "is identical."
        ),
        "Stuck?": mo.md("`labs/solutions/lab00.py`."),
    })
    return


@app.cell(hide_code=True)
def _(DeformableCrossAttention, F, checks, torch):
    def _make():
        torch.manual_seed(0)
        return DeformableCrossAttention(8, 2, 3)

    def _zero_predictors(layer):
        with torch.no_grad():
            layer.offset_proj.weight.zero_()
            layer.offset_proj.bias.zero_()
            layer.weight_proj.weight.zero_()
            layer.weight_proj.bias.zero_()

    def _shape():
        _layer = _make()
        _out = _layer(torch.randn(2, 5, 8), torch.rand(2, 5, 2),
                      torch.randn(2, 6, 7, 8))
        assert _out.shape == (2, 5, 8)

    def _collapse_to_lookup():
        _layer = _make()
        _zero_predictors(_layer)
        _query = torch.randn(2, 5, 8)
        _ref = torch.rand(2, 5, 2)
        _value = torch.randn(2, 6, 7, 8)
        _out = _layer(_query, _ref, _value)
        # with zeroed predictors, each query just reads value_proj(value) at
        # its reference point (x*W - 0.5 in pixels == grid coord 2x - 1)
        _vv = _layer.value_proj(_value).permute(0, 3, 1, 2)
        _smp = F.grid_sample(
            _vv, (2 * _ref - 1)[:, :, None, :],
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[..., 0]
        _expected = _layer.out_proj(_smp.transpose(1, 2))
        torch.testing.assert_close(_out, _expected, rtol=1e-5, atol=1e-6)

    def _ref_points_differentiable():
        _layer = _make()
        _ref = torch.rand(1, 4, 2, requires_grad=True)
        _layer(torch.randn(1, 4, 8), _ref, torch.randn(1, 6, 7, 8)).sum().backward()
        assert _ref.grad is not None and _ref.grad.abs().sum() > 0, (
            "no gradient reached reference_points"
        )
        _g = _layer.offset_proj.weight.grad
        assert _g is not None and _g.abs().sum() > 0

    checks.run_checks({
        "output shape is (B, Q, C)": _shape,
        "zeroed predictors collapse to a lookup at the reference point":
            _collapse_to_lookup,
        "gradients reach reference_points and offset_proj":
            _ref_points_differentiable,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Multi-scale: the real operator

    One feature map cannot serve detection. Small objects dissolve at stride
    32; at stride 8 a large object doesn't fit in any local neighborhood. So
    backbones emit a **pyramid** — $L$ maps of the same scene at different
    resolutions (FPN-style, typically strides 8/16/32/64) — and a query should
    sample *all* of them: fine levels for detail, coarse levels for context.

    Two mechanical problems, and their solutions:

    **One point, $L$ resolutions?** Already solved. A *normalized* reference
    point means the same place on every level; each level maps it to its own
    pixels with $x \cdot W_l - 0.5$.

    **One tensor, $L$ shapes?** You cannot stack a $100{\times}100$ map and a
    $50{\times}50$ map into one tensor along $H, W$ — so flatten each level
    row-major and concatenate along a single axis $S = \sum_l H_l W_l$. The
    price: shape metadata now travels separately (`spatial_shapes`), and level
    $l$ starts at $\text{start}_l = \sum_{i<l} H_i W_i$, so its pixel $(y, x)$
    lives at `value[b, start_l + y * W_l + x, m]`. Move the point and watch
    both mappings at once:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pyr_x = mo.ui.slider(0.0, 1.0, step=0.01, value=0.63, label="x (normalized)")
    pyr_y = mo.ui.slider(0.0, 1.0, step=0.01, value=0.41, label="y (normalized)")
    mo.hstack([pyr_x, pyr_y], justify="start")
    return pyr_x, pyr_y


@app.cell(hide_code=True)
def _(PAL, mo, pyr_x, pyr_y, svg_dot, svg_grid, svg_rect, svg_text, svg_wrap):
    import math as _math

    _levels = [(12, 16), (6, 8), (3, 4)]  # (H_l, W_l)
    _pw, _ph = 160, 120
    _oxs, _oy = (30, 225, 420), 40
    _xn, _yn = pyr_x.value, pyr_y.value
    _parts, _ticks, _start = [], [], 0
    for _lvl, (_h, _w) in enumerate(_levels):
        _ox = _oxs[_lvl]
        _cellsz = _pw // _w
        _xp, _yp = _xn * _w - 0.5, _yn * _h - 0.5
        _x0, _y0 = _math.floor(_xp), _math.floor(_yp)
        _lx, _ly = _xp - _x0, _yp - _y0
        _corners = (
            (_x0,     _y0,     (1 - _lx) * (1 - _ly)),
            (_x0 + 1, _y0,     _lx * (1 - _ly)),
            (_x0,     _y0 + 1, (1 - _lx) * _ly),
            (_x0 + 1, _y0 + 1, _lx * _ly),
        )
        _indices = []
        _valid_weight = 0.0
        for _cx, _cy, _weight in _corners:
            if _weight > 1e-9 and 0 <= _cx < _w and 0 <= _cy < _h:
                _flat = _start + _cy * _w + _cx
                _indices.append(_flat)
                _valid_weight += _weight
                _ticks.append((_flat, _weight))
                _parts.append(svg_rect(
                    _ox + _cx * _cellsz, _oy + _cy * _cellsz,
                    _cellsz, _cellsz, PAL["lightorange"],
                    opacity=0.25 + 0.75 * _weight,
                ))
        _parts.append(svg_grid(_ox, _oy, _w, _h, _cellsz, PAL["faint"]))
        _parts.append(svg_dot(_ox + _xn * _pw, _oy + _yn * _ph, 4, PAL["blue"]))
        _parts.append(svg_text(_ox + _pw / 2, _oy - 16,
                               f"level {_lvl} — {_h}×{_w}", PAL["ink"],
                               bold=True))
        _parts.append(svg_text(
            _ox + _pw / 2, _oy + _ph + 16,
            f"(x, y) = ({_xp:.2f}, {_yp:.2f}) px",
            PAL["grid"], size=10,
        ))
        _parts.append(svg_text(
            _ox + _pw / 2, _oy + _ph + 30,
            f"S: {', '.join(map(str, _indices))} · Σw={_valid_weight:.2f}",
            PAL["ink"], size=9,
        ))
        _start += _h * _w
    _S = _start
    _bx, _bw, _by, _bh = 30, 550, 222, 20
    _x = _bx
    for _lvl, (_h, _w) in enumerate(_levels):
        _seg = _bw * _h * _w / _S
        _parts.append(svg_rect(_x, _by, _seg, _bh,
                               ("#eff6ff", "#e0f2fe", "#f0fdf4")[_lvl]))
        _parts.append(
            f'<rect x="{_x:.1f}" y="{_by}" width="{_seg:.1f}" height="{_bh}" '
            f'fill="none" stroke="{PAL["grid"]}" stroke-width="0.8"/>'
        )
        if _seg > 60:
            _parts.append(svg_text(_x + _seg / 2, _by + 14, f"level {_lvl}",
                                   PAL["ink"], size=10))
        _level_start = sum(h * w for h, w in _levels[:_lvl])
        _parts.append(svg_text(_x + 2, _by + _bh + 13, f"{_level_start}",
                               PAL["grid"], size=9, anchor="start"))
        _x += _seg
    for _flat, _weight in _ticks:
        # Draw the marker at the center of the flattened element, with opacity
        # and width encoding its bilinear contribution.
        _tickx = _bx + _bw * (_flat + 0.5) / _S
        _parts.append(
            f'<line x1="{_tickx:.1f}" y1="{_by - 5}" '
            f'x2="{_tickx:.1f}" y2="{_by + _bh}" '
            f'stroke="{PAL["orange"]}" stroke-width="{1 + 3 * _weight:.1f}" '
            f'opacity="{0.3 + 0.7 * _weight:.2f}"/>'
        )
    _parts.append(svg_text(_bx + _bw, _by + _bh + 13, f"S = {_S}",
                           PAL["grid"], size=9, anchor="end"))
    _parts.append(svg_text(
        _bx, _by - 9,
        "flattened S axis (orange ticks = valid bilinear corners; strength = weight)",
        PAL["ink"], size=10, anchor="start",
    ))
    mo.Html(svg_wrap("".join(_parts), 610, 278))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The operator in full (Deformable DETR, eq. 3 — the equation this entire
    repo is about):

    $$
    \mathrm{MSDA}(q) = \sum_{m=1}^{M} \Big[ \sum_{l=1}^{L} \sum_{k=1}^{K}
    A_{mlqk} \cdot x_l \big( \phi_l(\hat{p}_q) + \Delta p_{mlqk} \big) \Big]
    $$

    and its tensor contract, used by this repo everywhere:

    | tensor | shape | meaning |
    |---|---|---|
    | `value` | `(B, S, M, D)` | the feature pyramid, all `L` levels flattened along `S = Σ H_l·W_l`, split into `M` heads × `D` channels |
    | `spatial_shapes` | `(L, 2)` | `(H_l, W_l)` of each level |
    | `sampling_locations` | `(B, Q, M, L, K, 2)` | normalized `(x, y)` in `[0, 1]` — **offsets already added** |
    | `attention_weights` | `(B, Q, M, L, K)` | softmax-normalized over `(L, K)` |
    | output | `(B, Q, M·D)` | per query: heads concatenated |

    Two details worth a pause:

    - the softmax runs over **all $L \cdot K$ points jointly** — a query can
      shift its whole attention budget onto one scale;
    - the operator receives `sampling_locations` with offsets *already added*:
      splitting "reference + offset" is the layer's business (the capstone),
      not the operator's. This exact function signature is what labs 1–10 turn
      into a GPU kernel.

    ### ✏️ Exercise 5 — `msda`

    Implement the contract with the native primitive: one `F.grid_sample`
    call per level — the standard PyTorch fallback (same math as mmcv's
    `multi_scale_deformable_attn_pytorch`, and exactly what
    `tests/reference_impls.py::msda_reference` does). Split `value` by level
    along the flat `S` axis, build the grid as `2 * sampling_locations - 1`, and
    weight the per-level samples by `attention_weights`. Loops over `level`
    are fine; everything else is one tensor op per level.
    """)
    return


@app.cell
def _(checks, deform_attend):
    def msda(value, spatial_shapes, sampling_locations, attention_weights):
        """Multi-scale deformable attention. Returns (B, Q, M * D)."""
        import torch.nn.functional as F
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = sampling_locations.shape
        _ = deform_attend  # you may reuse this, or go straight to grid_sample
        # ================= YOUR CODE =================
        # The standard PyTorch fallback: one F.grid_sample call per level.
        # 1. shapes = [(int(h), int(w)) for h, w in spatial_shapes];
        #    value_list = value.split([h * w for h, w in shapes], dim=1)
        # 2. grids = 2 * sampling_locations - 1   # [0,1] -> [-1,1]
        # 3. per level lvl: v = value_list[lvl].flatten(2).transpose(1, 2)
        #       .reshape(B * M, D, h, w)
        #    g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # (B*M, Q, K, 2)
        #    sampled.append(F.grid_sample(v, g, mode="bilinear",
        #                               padding_mode="zeros", align_corners=False))
        # 4. stack levels -> (B*M, D, Q, L*K); weight by attn; sum;
        #    reshape to (B, Q, M * D)
        raise checks.NotDoneYet()
        # =============================================
    return (msda,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint 1 — split the S axis by level": mo.md(
            "```python\nshapes = [(int(h), int(w)) for h, w in spatial_shapes]\n"
            "value_list = value.split([h * w for h, w in shapes], dim=1)\n"
            "```\n"
            "`value_list[lvl]` is `(B, H_l*W_l, M, D)` — each level already "
            "lives in its own slice of the flat `S` axis."
        ),
        "Hint 2 — one grid_sample per level": mo.md(
            "Locations are normalized `[0, 1]`; `grid_sample` wants "
            "`[-1, 1]`, so `grids = 2 * sampling_locations - 1`. Then per level:\n"
            "```python\nv = value_list[lvl].flatten(2).transpose(1, 2)\\\n"
            "      .reshape(B * M, D, h, w)\n"
            "g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # (B*M, Q, K, 2)\n"
            "sampled = F.grid_sample(v, g, mode=\"bilinear\",\n"
            "                       padding_mode=\"zeros\", align_corners=False)\n"
            "```\n"
            "Stack the per-level `(B*M, D, Q, K)` results over levels → "
            "`(B*M, D, Q, L*K)`, weight by `attention_weights` reshaped to "
            "`(B*M, 1, L*K)`, sum over the last axis, and reshape to "
            "`(B, Q, M * D)`. This is exactly "
            "`tests/reference_impls.py::msda_reference`."
        ),
        "Stuck?": mo.md("`labs/solutions/lab00.py`."),
    })
    return


@app.cell(hide_code=True)
def _(checks, msda, torch):
    def _matches_reference():
        checks.assert_msda_matches(
            lambda v, s, l, a: msda(v, s, l, a), dtype=torch.float64
        )
        checks.assert_msda_matches(
            lambda v, s, l, a: msda(v, s, l, a), dtype=torch.float32
        )

    def _differentiable_everywhere():
        _v, _s, _st, _loc, _attn = checks.make_inputs(
            B=1, Q=3, M=2, D=4, shapes=[(5, 7), (3, 4)], K=2
        )
        _v.requires_grad_(True)
        _loc.requires_grad_(True)
        _attn.requires_grad_(True)
        msda(_v, _s, _loc, _attn).sum().backward()
        for _t, _name in ((_v, "value"), (_loc, "locations"), (_attn, "weights")):
            assert _t.grad is not None and _t.grad.abs().sum() > 0, (
                f"no gradient reached {_name}"
            )
            assert torch.isfinite(_t.grad).all()

    checks.run_checks({
        "matches the repo reference (incl. out-of-bounds)": _matches_reference,
        "differentiable w.r.t. value, locations, and weights":
            _differentiable_everywhere,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That first check compares you against
    `tests/reference_impls.py::msda_reference` — the `grid_sample`-based
    implementation the *repo's own pytest suite* trusts, at the suite's own
    tolerances, including cases that sample outside the image. Pass it and your
    function is, functionally, this repo's operator.

    ## 6. Capstone: the full Deformable DETR layer

    Everything at once — object queries, reference points, a flattened pyramid.
    This is the layer that sits inside every Deformable-DETR-family decoder
    (DINO, Grounding DINO, and friends):

    ```python
    MSDACrossAttention(embed_dim, num_heads, num_levels, num_points)
    forward(query: (B, Q, C), reference_points: (B, Q, 2) in [0, 1],
            value: (B, S, C), spatial_shapes: (L, 2)) -> (B, Q, C)
    ```

    - same four submodule names; `offset_proj` now outputs $M{\cdot}L{\cdot}K{\cdot}2$
      and `weight_proj` $M{\cdot}L{\cdot}K$;
    - offsets are predicted in *each level's pixel units*, so divide by
      $(W_l, H_l)$ to make them normalized before adding:
      $\text{loc}_{mlk} = \hat p_q + \Delta p_{mlk} / (W_l, H_l)$;
    - weights: **one** softmax over the flattened $L{\cdot}K$ axis;
    - the core is your `msda`.

    (The real implementation differs only in engineering: careful init —
    offset biases arranged in a spread-out star so training starts stable —
    and padding masks. The math is what you are writing.)

    ### ✏️ Exercise 6 — `MSDACrossAttention`
    """)
    return


@app.cell
def _(checks, msda, torch):
    class MSDACrossAttention(torch.nn.Module):
        """Multi-scale deformable cross-attention (Deformable DETR decoder).

        See the contract above; the sampling core is your `msda`.
        """

        def __init__(self, embed_dim, num_heads, num_levels, num_points):
            super().__init__()
            # ================= YOUR CODE =================
            raise checks.NotDoneYet()
            # =============================================

        def forward(self, query, reference_points, value, spatial_shapes):
            _ = msda  # you'll want this
            # ================= YOUR CODE =================
            raise checks.NotDoneYet()
            # =============================================
    return (MSDACrossAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint — the two lines that are new": mo.md(
            "```python\nscale = spatial_shapes.flip(-1).to(query.dtype)  # (L, 2) = (W_l, H_l)\n"
            "locs = (reference_points[:, :, None, None, None, :]\n"
            "        + offsets / scale[None, None, None, :, None, :])\n```\n"
            "and the joint softmax: reshape the weight logits to "
            "`(B, Q, M, L*K)`, `softmax(dim=-1)`, reshape back to "
            "`(B, Q, M, L, K)`. Everything else is exercise 4's wiring with "
            "`msda` in the middle."
        ),
        "Stuck?": mo.md("`labs/solutions/lab00.py`."),
    })
    return


@app.cell(hide_code=True)
def _(MSDACrossAttention, checks, torch):
    def _make():
        torch.manual_seed(0)
        return MSDACrossAttention(8, 2, 2, 3)

    _shapes = torch.tensor([[5, 7], [3, 4]])  # S = 35 + 12 = 47

    def _zero_predictors(layer):
        with torch.no_grad():
            layer.offset_proj.weight.zero_()
            layer.offset_proj.bias.zero_()
            layer.weight_proj.weight.zero_()
            layer.weight_proj.bias.zero_()

    def _shape():
        _layer = _make()
        _out = _layer(torch.randn(2, 5, 8), torch.rand(2, 5, 2),
                      torch.randn(2, 47, 8), _shapes)
        assert _out.shape == (2, 5, 8)

    def _collapse():
        _layer = _make()
        _zero_predictors(_layer)
        _query = torch.randn(2, 5, 8)
        _ref = torch.rand(2, 5, 2)
        _value = torch.randn(2, 47, 8)
        _out = _layer(_query, _ref, _value, _shapes)
        # zeroed predictors: every sample sits at the reference point on every
        # level, uniformly weighted 1/(L*K)
        _v = _layer.value_proj(_value).reshape(2, 47, 2, 4)
        _locs = _ref[:, :, None, None, None, :].expand(2, 5, 2, 2, 3, 2)
        _w = torch.full((2, 5, 2, 2, 3), 1 / 6)
        _expected = _layer.out_proj(
            checks.msda_reference(_v, _shapes, _locs, _w)
        )
        torch.testing.assert_close(_out, _expected, rtol=1e-5, atol=1e-6)

    def _offsets_learn():
        _layer = _make()
        _out = _layer(torch.randn(1, 4, 8), torch.rand(1, 4, 2),
                      torch.randn(1, 47, 8), _shapes)
        _out.sum().backward()
        _g = _layer.offset_proj.weight.grad
        assert _g is not None and _g.abs().sum() > 0

    checks.run_checks({
        "output shape is (B, Q, C)": _shape,
        "zeroed predictors collapse to uniform reads at the reference":
            _collapse,
        "gradients reach the offset predictor": _offsets_learn,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏁 What you built — and where the ladder goes

    | artifact | what it is |
    |---|---|
    | `bilinear_sample` | the differentiable read — the reason "where to look" is trainable at all |
    | `deform_attend` | the core: weighted bilinear reads at predicted points |
    | `DeformableAttention` | the single-scale self-attention layer (predictions from map features) |
    | `DeformableCrossAttention` | object queries + reference points (predictions from query embeddings) |
    | `msda` | **this repo's exact operator contract**, verified against its test suite's ground truth |
    | `MSDACrossAttention` | the Deformable DETR decoder layer, end to end |

    Notice where the *parameters* live: four `nn.Linear`s per layer — nothing
    exotic. All of the operator's difficulty is inside `msda`'s data-dependent
    gather, which is exactly why it gets a hand-written kernel. Count what it
    does at the standard decoder config (`B=4, Q=300, M=8, D=32, L=4, K=4`):
    ~150k sampling points × 4 corner reads × 32 channels ≈ 20M multiply-adds —
    trivial arithmetic — but every read lands wherever the *network* pointed,
    scattered across a ~50 MB pyramid. **Memory-bound and data-dependent.**
    PyTorch has no fused primitive for it; making it fast is labs 1–10.

    ---
    ### Next

    **Lab 1 — the spec** (`01_the_spec.py`) restates `msda` as loops so simple
    they can't be wrong: that version becomes the ground truth every kernel
    stage answers to, all the way down to Triton. If you want, copy your
    functions into **`labs/my/lab00.py`** — nothing imports them, but after
    lab 10 it is deeply satisfying to drop the Triton operator *you built*
    into today's `MSDACrossAttention` and watch the same layer run on your own
    kernel.

    **Reading:** [Deformable DETR](https://arxiv.org/abs/2010.04159) (Zhu et
    al., ICLR 2021) — eqs. 1–3 are this lab;
    [Deformable ConvNets](https://arxiv.org/abs/1703.06211) for the lineage;
    [course chapter 1](https://roulbac.github.io/msda-triton/course/01-deformable-attention/)
    for the expository version with this repo's conventions.
    """)
    return


if __name__ == "__main__":
    app.run()
