"""Triton kernels for multi-scale deformable attention (MSDA).

The kernel design follows "Efficient Multi-Scale Deformable Attention on
GPUs" (TMLR submission): a *query-block tiling* in which each program
instance owns a contiguous block of queries for one (batch, head) pair.
Sampling parameters are loaded once per query block (one scalar per query,
coalesced over the query dimension) and each bilinear-corner gather is a wide
load over the channel dimension, so every DRAM transaction moves a full
head-channel row. This deliberately trades SM occupancy for effective
bandwidth, the profitable direction for this memory-bound operator
(paper Section 3.2: 17% occupancy / 36% of peak bandwidth beats 85%
occupancy / 5.1%).

Sampling locations and attention weights are read through disjoint pointers,
eliminating the concatenated (B, Q, L, K, 3) parameter buffer that the
reference implementation materializes (paper Section 2.3).

The backward kernel recomputes the bilinear corners and scatters grad_value
with relaxed-ordering atomic adds (`sem="relaxed"`). Relaxed ordering is safe
because gradient accumulation is commutative and the kernel-end barrier
provides the required happens-before edge; it also lets the memory subsystem
coalesce atomics aggressively and, on SM 9.0+, lowers to the native
hardware-reduction instruction (paper Sections 3.3 and 4.2). The dtype of the
grad_value buffer is chosen by the caller (ops.py): on GPUs without native
half-precision atomics (SM < 9.0) an FP32 scratch buffer avoids the
compare-and-swap emulation that collapses BF16 atomic throughput by ~10x
(paper Table 2).

Bilinear-sampling convention (identical to the reference CUDA kernel and to
``F.grid_sample(align_corners=False, padding_mode="zeros")``):

    x_im = x * W_l - 0.5,   y_im = y * H_l - 0.5

with out-of-bounds corners contributing zero.

All flat indices into value/grad_value and the sampling parameters are
computed in 64-bit arithmetic: at encoder scale (S in the millions) the flat
element index B*S*M*D exceeds 2^31.
"""

import triton
import triton.language as tl

# The backward launch is fixed rather than autotuned: autotuning a kernel
# that accumulates into its output via atomics requires resetting the output
# between timing runs. The config was picked by a launch-config sweep on
# L40S/A100/H100 (benchmarks/modal_variants.py::bwd_sweep): BLOCK_Q=16 with
# 4 warps gives each thread a 4-element slice of the (16, 32) tile, which
# spreads the atomic stream across the most warps. It is uniformly fastest
# at decoder scale (up to 1.75x over (32, 4) for fp16 and 2.1x for bf16
# native atomics on L40S; ~5% on H100) and within noise of every other
# config where the kernel is atomic-pipeline-bound (A100 fp32, encoder
# scale), consistent with paper Section 3.3.
BWD_BLOCK_Q = 16
BWD_NUM_WARPS = 4


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": bq}, num_warps=nw)
        for bq in (16, 32, 64)
        for nw in (2, 4)
    ],
    key=["Q", "M", "D", "L", "K"],
)
@triton.jit
def _msda_forward_kernel(
    value_ptr,  # (B, S, M, D) contiguous
    shapes_ptr,  # (L, 2) int64: (H_l, W_l)
    starts_ptr,  # (L,) int64: offset of level l in the S axis
    loc_ptr,  # (B, Q, M, L, K, 2) normalized [0, 1] (x, y)
    attn_ptr,  # (B, Q, M, L, K)
    out_ptr,  # (B, Q, M, D)
    Q,
    S,
    M: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bm = tl.program_id(1)
    b = (pid_bm // M).to(tl.int64)
    m = pid_bm % M

    offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

    # Flat (b, q, m) index shared by the sampling-location and
    # attention-weight layouts (disjoint pointers, no concatenated buffer).
    pq = (b * Q + offs_q) * M + m
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

            v00 = tl.load(
                val_base + ((row0 + x0) * (M * D))[:, None],
                mask=(vy0 & vx0)[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v01 = tl.load(
                val_base + ((row0 + x1) * (M * D))[:, None],
                mask=(vy0 & vx1)[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v10 = tl.load(
                val_base + ((row1 + x0) * (M * D))[:, None],
                mask=(vy1 & vx0)[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v11 = tl.load(
                val_base + ((row1 + x1) * (M * D))[:, None],
                mask=(vy1 & vx1)[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += (
                v00 * (attn * (1.0 - lx) * (1.0 - ly))[:, None]
                + v01 * (attn * lx * (1.0 - ly))[:, None]
                + v10 * (attn * (1.0 - lx) * ly)[:, None]
                + v11 * (attn * lx * ly)[:, None]
            )

    out_ptrs = out_ptr + pq[:, None] * D + offs_d[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_q[:, None] & mask_d[None, :],
    )


@triton.jit
def _msda_backward_kernel(
    value_ptr,  # (B, S, M, D) contiguous
    shapes_ptr,  # (L, 2) int64
    starts_ptr,  # (L,) int64
    loc_ptr,  # (B, Q, M, L, K, 2)
    attn_ptr,  # (B, Q, M, L, K)
    grad_out_ptr,  # (B, Q, M, D) incoming gradient
    grad_value_ptr,  # (B, S, M, D) zero-initialized; fp32 or value dtype
    grad_loc_ptr,  # (B, Q, M, L, K, 2)
    grad_attn_ptr,  # (B, Q, M, L, K)
    Q,
    S,
    M: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
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

    g = tl.load(
        grad_out_ptr + pq[:, None] * D + offs_d[None, :], mask=mask_qd, other=0.0
    ).to(tl.float32)

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

            # d(out)/d(attn): the bilinearly sampled value itself.
            sampled = (
                v00 * w00[:, None]
                + v01 * w01[:, None]
                + v10 * w10[:, None]
                + v11 * w11[:, None]
            )
            ga = tl.sum(g * sampled, axis=1)

            # Analytic bilinear derivative w.r.t. the image coordinates,
            # chained through x_im = loc_x * W - 0.5 (hence the fw/fh factor).
            dx = (v01 - v00) * (1.0 - ly)[:, None] + (v11 - v10) * ly[:, None]
            dy = (v10 - v00) * (1.0 - lx)[:, None] + (v11 - v01) * lx[:, None]
            gx = attn * fw * tl.sum(g * dx, axis=1)
            gy = attn * fh * tl.sum(g * dy, axis=1)

            ldt = grad_loc_ptr.dtype.element_ty
            tl.store(grad_loc_ptr + 2 * p, gx.to(ldt), mask=mask_q)
            tl.store(grad_loc_ptr + 2 * p + 1, gy.to(ldt), mask=mask_q)
            tl.store(
                grad_attn_ptr + p, ga.to(grad_attn_ptr.dtype.element_ty), mask=mask_q
            )

            # Scattered gradient accumulation into grad_value: four relaxed
            # atomic adds per sample point (paper Figure 8).
            wg = g * attn[:, None]
            gdt = grad_value_ptr.dtype.element_ty
            tl.atomic_add(
                gval_base + off00, (wg * w00[:, None]).to(gdt), mask=m00, sem="relaxed"
            )
            tl.atomic_add(
                gval_base + off01, (wg * w01[:, None]).to(gdt), mask=m01, sem="relaxed"
            )
            tl.atomic_add(
                gval_base + off10, (wg * w10[:, None]).to(gdt), mask=m10, sem="relaxed"
            )
            tl.atomic_add(
                gval_base + off11, (wg * w11[:, None]).to(gdt), mask=m11, sem="relaxed"
            )
