"""Lab 4 solution: the ladder of first Triton kernels.

Each kernel comes with a tiny launch wrapper so checkers (and you) can call
it like a normal function. All of them run under TRITON_INTERPRET=1.
"""

from labs.common import setup  # noqa: F401  (must precede triton import)

import torch
import triton
import triton.language as tl


# --- rung 1: vector add ----------------------------------------------------


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y, BLOCK=256):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out


# --- rung 2: 2-D tile copy (masks on both axes) -----------------------------


@triton.jit
def copy2d_kernel(x_ptr, out_ptr, R, C, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r < R)[:, None] & (offs_c < C)[None, :]
    ptrs = offs_r[:, None] * C + offs_c[None, :]  # row-major flat offsets
    tile = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + ptrs, tile, mask=mask)


def copy2d(x, BLOCK_R=16, BLOCK_C=16):
    R, C = x.shape
    out = torch.empty_like(x)
    grid = (triton.cdiv(R, BLOCK_R), triton.cdiv(C, BLOCK_C))
    copy2d_kernel[grid](x, out, R, C, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C)
    return out


# --- rung 3: data-dependent gather ------------------------------------------


@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    val = tl.load(src_ptr + idx, mask=mask, other=0.0)  # addresses from data!
    tl.store(out_ptr + offs, val, mask=mask)


def gather(src, idx, BLOCK=256):
    out = torch.empty(idx.shape, dtype=src.dtype, device=src.device)
    n = idx.numel()
    gather_kernel[(triton.cdiv(n, BLOCK),)](src, idx, out, n, BLOCK=BLOCK)
    return out


# --- rung 4: bilinear gather, one level / one point per query ---------------
# A miniature of the real forward: value is one (H*W, D) image, each query
# has one normalized location; output is the bilinearly sampled (Q, D).


@triton.jit
def bilinear_kernel(
    value_ptr,  # (H*W, D) contiguous
    loc_ptr,    # (Q, 2) normalized (x, y)
    out_ptr,    # (Q, D)
    Q, H, W,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

    x = tl.load(loc_ptr + 2 * offs_q, mask=mask_q, other=0.0).to(tl.float32) * W - 0.5
    y = tl.load(loc_ptr + 2 * offs_q + 1, mask=mask_q, other=0.0).to(tl.float32) * H - 0.5
    x0f = tl.math.floor(x)
    y0f = tl.math.floor(y)
    lx = x - x0f
    ly = y - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    vx0 = mask_q & (x0 >= 0) & (x0 < W)
    vx1 = mask_q & (x1 >= 0) & (x1 < W)
    vy0 = (y0 >= 0) & (y0 < H)
    vy1 = (y1 >= 0) & (y1 < H)

    base = value_ptr + offs_d[None, :]
    v00 = tl.load(base + ((y0 * W + x0) * D)[:, None],
                  mask=(vy0 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v01 = tl.load(base + ((y0 * W + x1) * D)[:, None],
                  mask=(vy0 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v10 = tl.load(base + ((y1 * W + x0) * D)[:, None],
                  mask=(vy1 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v11 = tl.load(base + ((y1 * W + x1) * D)[:, None],
                  mask=(vy1 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    out = (
        v00 * ((1.0 - lx) * (1.0 - ly))[:, None]
        + v01 * (lx * (1.0 - ly))[:, None]
        + v10 * ((1.0 - lx) * ly)[:, None]
        + v11 * (lx * ly)[:, None]
    )
    out_ptrs = out_ptr + offs_q[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty),
             mask=mask_q[:, None] & mask_d[None, :])


def bilinear(value_hw_d, loc, BLOCK_Q=32):
    """value_hw_d: (H, W, D); loc: (Q, 2) normalized. Returns (Q, D)."""
    H, W, D = value_hw_d.shape
    Q = loc.shape[0]
    out = value_hw_d.new_empty(Q, D)
    bilinear_kernel[(triton.cdiv(Q, BLOCK_Q),)](
        value_hw_d.reshape(H * W, D).contiguous(), loc.contiguous(), out,
        Q, H, W, D=D, BLOCK_Q=BLOCK_Q, BLOCK_D=triton.next_power_of_2(D),
    )
    return out
