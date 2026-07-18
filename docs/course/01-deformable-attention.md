# 1. Deformable attention, from scratch

!!! abstract "What you'll learn"
    By the end of this chapter you'll know what multi-scale deformable attention (MSDA) computes, why it exists, and how bilinear interpolation works — which is 90% of the math in this entire repo.

## The problem: attention over images is quadratic

You already know standard attention from transformers: every query attends to every key,

$$
\mathrm{Attn}(q) = \sum_{j} \mathrm{softmax}_j\!\left(\frac{q \cdot k_j}{\sqrt{d}}\right) v_j .
$$

The cost is proportional to (number of queries) × (number of keys). For text, both are sequence length — a few thousand tokens, fine.

Now try it on an image. Object detectors like DETR run a transformer over the feature maps produced by a CNN backbone. A typical detection input is 800×1333 pixels, and the backbone emits a **feature pyramid**: the same image at several resolutions (strides 8, 16, 32, 64), which detectors need because objects come in all sizes. Flattened into tokens, that pyramid contains

$$
S = \sum_{l=1}^{L} H_l \times W_l \approx 100{,}000 \text{ tokens.}
$$

Dense attention with 100k keys per query is $10^{10}$ query–key pairs per layer. This is why the original DETR was restricted to a single low-resolution feature map, trained slowly, and struggled with small objects.

## The idea: attend to a few *learned* locations

**Deformable attention** (Zhu et al., 2021 — the "Deformable DETR" paper) makes a radical simplification: instead of comparing each query against every key, let each query directly *predict* a handful of locations worth looking at, and only read those.

Concretely, for each query:

1. The query has a **reference point** $\hat{p}_q$ — a 2D anchor (normalized to $[0,1]^2$) saying roughly where in the image it's looking.
2. A small linear layer on the query features predicts $K$ **sampling offsets** $\Delta p$ per head and per pyramid level — "look a bit up-left of the anchor, a bit right, …". These are learned, and different for every query at every layer.
3. Another linear layer predicts **attention weights** $A$ — one scalar per sampling point, softmax-normalized so they sum to 1.
4. The output is just the weighted sum of the feature vectors read at those locations.

No query–key dot products at all. The "attention" weights come from the query alone, and each query touches a *fixed* number of locations: with $M=8$ heads, $L=4$ levels, $K=4$ points, that's $M \times L \times K = 128$ reads per query, instead of 100,000 comparisons. The cost is now *linear* in the number of queries and independent of image size.

## The equation, term by term

Here is the full operator, as in Eq. 1 of the Deformable DETR paper:

$$
\mathrm{MSDA}(q) \;=\; \sum_{m=1}^{M} W_m \underbrace{\left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot x_l\!\big(\phi_l(\hat{p}_q) + \Delta p_{mlqk}\big) \right]}_{\text{what this repo implements}}
$$

Reading it inside-out:

| Symbol | Meaning |
|---|---|
| $x_l$ | Feature map at pyramid level $l$ (shape $H_l \times W_l$, with $D$ channels per head) |
| $\hat{p}_q$ | Reference point of query $q$, normalized to $[0,1]^2$ |
| $\phi_l(\cdot)$ | Rescales the normalized point into level $l$'s coordinate frame |
| $\Delta p_{mlqk}$ | Learned offset: where head $m$ of query $q$ looks at level $l$, point $k$ |
| $x_l(\cdot)$ | *Read the feature map at that (fractional!) location* — this is bilinear interpolation, below |
| $A_{mlqk}$ | Learned attention weight for that sample; softmax-normalized over all $(l,k)$ |
| $W_m$ | The usual per-head output projection |

The bracketed term — sample $L \times K$ locations, weight them, sum them, per head — is the entire job of this repo's kernels. Everything outside the bracket ($W_m$, and the linear layers that *produce* $\Delta p$ and $A$) is ordinary dense matrix multiplication that PyTorch already does optimally.

In code, the kernel's inputs are exactly the tensors in that bracket:

```python
value               # (B, S, M, D)          the flattened feature pyramid, per head
spatial_shapes      # (L, 2)                (H_l, W_l) for each level
sampling_locations  # (B, Q, M, L, K, 2)    = φ_l(p̂_q) + Δp, already summed, in [0,1]
attention_weights   # (B, Q, M, L, K)       softmax-normalized over (L, K)
# output:           # (B, Q, M*D)
```

Note that the *model* keeps the reference points and offsets separate (offsets get a smaller learning rate, etc.), but by the time the operator is called they've been added together into `sampling_locations`.

## Bilinear interpolation: reading between the pixels

There's one wrinkle: $\phi_l(\hat p_q) + \Delta p$ is a *continuous* coordinate like $(37.3, 12.8)$ — the network learns offsets as real numbers, and gradients need to flow through them. But a feature map only has values at integer pixel positions. **Bilinear interpolation** bridges the gap: read the four surrounding pixels and blend them by proximity.

For a sample point at $(x, y)$, let $x_0 = \lfloor x \rfloor$, $y_0 = \lfloor y \rfloor$, and let $l_x = x - x_0$, $l_y = y - y_0$ be the fractional parts. Then

$$
x_l(x, y) \;=\;
\underbrace{(1-l_x)(1-l_y)}_{w_{00}} v_{00} +
\underbrace{l_x(1-l_y)}_{w_{01}} v_{01} +
\underbrace{(1-l_x)\,l_y}_{w_{10}} v_{10} +
\underbrace{l_x\,l_y}_{w_{11}} v_{11}
$$

where $v_{00}$ is the pixel at $(y_0, x_0)$, $v_{01}$ at $(y_0, x_0{+}1)$, and so on. The four weights are each in $[0,1]$ and always sum to 1: if the point sits exactly on a pixel, that pixel gets weight 1; if it sits in the middle of four pixels, each gets ¼.

Two conventions matter, because every implementation must agree on them exactly:

**Coordinate mapping.** Sampling locations arrive normalized to $[0,1]$. This repo (like the reference CUDA kernel, and like `F.grid_sample(align_corners=False)`) maps them to pixel space as

$$
x_{\text{im}} = x \cdot W_l - 0.5, \qquad y_{\text{im}} = y \cdot H_l - 0.5 .
$$

The $-0.5$ treats pixels as unit squares with values at their *centers*: normalized coordinate $x = \frac{0.5}{W_l}$ lands exactly on the center of the first pixel.

**Out-of-bounds handling.** A learned offset can point outside the image. Any of the four corner pixels that falls outside contributes **zero** (`padding_mode="zeros"`). The sample fades smoothly to zero as the point leaves the image — important, because a hard cutoff would break gradients.

## The simplest possible implementation

Everything above fits in a page of plain Python. This is `msda_naive` from [`tests/reference_impls.py`](https://github.com/roulbac/msda-triton/blob/main/tests/reference_impls.py) — unbearably slow, but its correctness is checkable by eye, which makes it the root of this repo's testing strategy (chapter 7):

```python
for b in range(B):
  for q in range(Q):
    for m in range(M):                       # heads
      for lvl in range(L):                   # pyramid levels
        h, w = shapes[lvl]
        for k in range(K):                   # sampling points
          x = float(sampling_locations[b, q, m, lvl, k, 0]) * w - 0.5
          y = float(sampling_locations[b, q, m, lvl, k, 1]) * h - 0.5
          a = attention_weights[b, q, m, lvl, k]
          x0, y0 = math.floor(x), math.floor(y)
          lx, ly = x - x0, y - y0
          corners = (
              (y0,     x0,     (1 - lx) * (1 - ly)),
              (y0,     x0 + 1, lx       * (1 - ly)),
              (y0 + 1, x0,     (1 - lx) * ly),
              (y0 + 1, x0 + 1, lx       * ly),
          )
          for yy, xx, wgt in corners:
              if 0 <= xx < w and 0 <= yy < h:          # zeros padding
                  out[b, q, m] += a * wgt * value[b, starts[lvl] + yy * w + xx, m]
```

Note the innermost line: `value[b, starts[lvl] + yy*w + xx, m]` is a $D$-dimensional vector. Each sample point reads **4 corner vectors × D channels**, multiplies by a scalar, and accumulates. For the standard decoder configuration ($B{=}4$, $Q{=}300$, $M{=}8$, $D{=}32$, $L{=}4$, $K{=}4$) that's about 20 million multiply-adds — trivial arithmetic. The challenge is *where* those reads go.

## Why this operator is hard to make fast

Compare with matrix multiplication, the operator GPUs are built for. In a matmul, every memory access is knowable at compile time; you can tile the problem so each loaded value is reused hundreds of times from fast on-chip memory, and modern GPUs have dedicated hardware (tensor cores) for exactly this.

MSDA has none of that structure:

- **The read locations are data-dependent.** They're network outputs, unknown until runtime, different for every query. No static tiling can guarantee reuse.
- **Reads are scattered.** Query 17 might sample the top-left of level 0 while query 18 samples the bottom-right of level 3 — addresses all over a ~50 MB buffer.
- **Arithmetic is trivial.** Four multiply-adds per loaded value. The GPU's floating-point units are almost idle; all the time goes into waiting for memory.

This is a **memory-bound, gather/scatter** operator. Making it fast is entirely about orchestrating memory traffic — which is why the next chapter is about how GPU memory actually works.

!!! summary "Key takeaways"
    - MSDA replaces dense attention with $M \times L \times K$ (typically 128) *learned* bilinear reads per query — cost linear in queries, independent of image size.
    - This repo implements the sampling/aggregation core; the surrounding projections are ordinary matmuls.
    - Bilinear interpolation = blend 4 neighboring pixels with weights $(1{-}l_x)(1{-}l_y)$, etc.; convention here is `x·W − 0.5` with zeros padding.
    - The operator is memory-bound with data-dependent, scattered accesses — the opposite of a matmul, and the reason it needs a carefully designed kernel.

[Next: How GPUs actually run code →](02-gpu-programming-model.md)
