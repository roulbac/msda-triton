"""Lab 5 solution: the complete MSDA forward kernel (query-block tiling).

Functionally identical to src/msda_triton/kernels.py::_msda_forward_kernel,
minus autotuning (BLOCK_Q is an explicit argument here; Lab 6 adds the
autotuner). Every design decision is explained in the notebook.
"""

from labs.common import setup  # noqa: F401  (must precede triton import)

import torch
import triton
import triton.language as tl


@triton.jit
def msda_forward_kernel(
    value_ptr,   # (B, S, M, D) contiguous
    shapes_ptr,  # (L, 2) int64: (H_l, W_l)
    starts_ptr,  # (L,) int64: offset of level l in the S axis
    loc_ptr,     # (B, Q, M, L, K, 2) normalized [0, 1] (x, y)
    attn_ptr,    # (B, Q, M, L, K)
    out_ptr,     # (B, Q, M, D)
    Q, S,
    M: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bm = tl.program_id(1)
    b = (pid_bm // M).to(tl.int64)
    m = pid_bm % M

    offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

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

            v00 = tl.load(val_base + ((row0 + x0) * (M * D))[:, None],
                          mask=(vy0 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            v01 = tl.load(val_base + ((row0 + x1) * (M * D))[:, None],
                          mask=(vy0 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            v10 = tl.load(val_base + ((row1 + x0) * (M * D))[:, None],
                          mask=(vy1 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            v11 = tl.load(val_base + ((row1 + x1) * (M * D))[:, None],
                          mask=(vy1 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            acc += (
                v00 * (attn * (1.0 - lx) * (1.0 - ly))[:, None]
                + v01 * (attn * lx * (1.0 - ly))[:, None]
                + v10 * (attn * (1.0 - lx) * ly)[:, None]
                + v11 * (attn * lx * ly)[:, None]
            )

    out_ptrs = out_ptr + pq[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=mask_q[:, None] & mask_d[None, :])


def msda_forward(value, spatial_shapes, level_start_index, sampling_locations,
                 attention_weights, BLOCK_Q=32):
    """Launch wrapper. Same contract as the repo operator; returns (B,Q,M*D)."""
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    spatial_shapes = spatial_shapes.to(value.device, torch.int64).contiguous()
    if level_start_index is None:
        hw = spatial_shapes.prod(-1)
        level_start_index = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    level_start_index = level_start_index.to(value.device, torch.int64).contiguous()

    out = value.new_empty(B, Q, M, D)
    grid = (triton.cdiv(Q, BLOCK_Q), B * M)
    msda_forward_kernel[grid](
        value.contiguous(), spatial_shapes, level_start_index,
        sampling_locations.contiguous(), attention_weights.contiguous(), out,
        Q, S, M=M, D=D, L=L, K=K,
        BLOCK_Q=BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
    )
    return out.view(B, Q, M * D)
