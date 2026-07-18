"""Lab 8 solution: the race demo and the full Triton backward kernel.

The backward kernel is functionally identical to
src/msda_triton/kernels.py::_msda_backward_kernel (BLOCK_Q explicit here).
The histogram kernels exist to make the data race *observable*: the racy
version loses updates on a real GPU, the atomic version never does.
"""

from labs.common import setup  # noqa: F401  (must precede triton import)

import torch
import triton
import triton.language as tl


# --- the race, made visible --------------------------------------------------


@triton.jit
def histogram_racy_kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out[idx[i]] += 1 via separate load and store: NOT atomic. Two programs
    hitting the same bin can both read c, both write c+1 — one increment lost."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    cur = tl.load(out_ptr + idx, mask=mask, other=0.0)   # read...
    tl.store(out_ptr + idx, cur + 1.0, mask=mask)        # ...modify...write: RACY
    # (Also racy *within* a program: lanes of one block sharing a bin collide.)


@triton.jit
def histogram_atomic_kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """The fix: an indivisible read-modify-write per lane."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    tl.atomic_add(out_ptr + idx, 1.0, mask=mask, sem="relaxed")


def histogram(idx, n_bins, atomic, BLOCK=256):
    out = torch.zeros(n_bins, dtype=torch.float32, device=idx.device)
    n = idx.numel()
    kern = histogram_atomic_kernel if atomic else histogram_racy_kernel
    kern[(triton.cdiv(n, BLOCK),)](idx, out, n, BLOCK=BLOCK)
    return out


# --- the full backward kernel ------------------------------------------------


@triton.jit
def msda_backward_kernel(
    value_ptr,       # (B, S, M, D) contiguous
    shapes_ptr,      # (L, 2) int64
    starts_ptr,      # (L,) int64
    loc_ptr,         # (B, Q, M, L, K, 2)
    attn_ptr,        # (B, Q, M, L, K)
    grad_out_ptr,    # (B, Q, M, D) incoming gradient
    grad_value_ptr,  # (B, S, M, D) zero-initialized; fp32 or value dtype
    grad_loc_ptr,    # (B, Q, M, L, K, 2)
    grad_attn_ptr,   # (B, Q, M, L, K)
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

            # d(out)/d(attn): the bilinearly sampled value itself.
            sampled = (v00 * w00[:, None] + v01 * w01[:, None]
                       + v10 * w10[:, None] + v11 * w11[:, None])
            ga = tl.sum(g * sampled, axis=1)

            # Analytic bilinear derivative, chained through x_im = x*W - 0.5.
            dx = (v01 - v00) * (1.0 - ly)[:, None] + (v11 - v10) * ly[:, None]
            dy = (v10 - v00) * (1.0 - lx)[:, None] + (v11 - v01) * lx[:, None]
            gx = attn * fw * tl.sum(g * dx, axis=1)
            gy = attn * fh * tl.sum(g * dy, axis=1)

            ldt = grad_loc_ptr.dtype.element_ty
            tl.store(grad_loc_ptr + 2 * p, gx.to(ldt), mask=mask_q)
            tl.store(grad_loc_ptr + 2 * p + 1, gy.to(ldt), mask=mask_q)
            tl.store(grad_attn_ptr + p, ga.to(grad_attn_ptr.dtype.element_ty),
                     mask=mask_q)

            # The scatter: four relaxed atomic adds per sample point. Relaxed
            # is safe because addition commutes and nobody reads grad_value
            # until the kernel-end barrier.
            wg = g * attn[:, None]
            gdt = grad_value_ptr.dtype.element_ty
            tl.atomic_add(gval_base + off00, (wg * w00[:, None]).to(gdt),
                          mask=m00, sem="relaxed")
            tl.atomic_add(gval_base + off01, (wg * w01[:, None]).to(gdt),
                          mask=m01, sem="relaxed")
            tl.atomic_add(gval_base + off10, (wg * w10[:, None]).to(gdt),
                          mask=m10, sem="relaxed")
            tl.atomic_add(gval_base + off11, (wg * w11[:, None]).to(gdt),
                          mask=m11, sem="relaxed")


def msda_backward(value, spatial_shapes, level_start_index, sampling_locations,
                  attention_weights, grad_out, fp32_grad_accum=False,
                  BLOCK_Q=16, num_warps=4):
    """Launch wrapper. Returns (grad_value, grad_loc, grad_attn)."""
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    spatial_shapes = spatial_shapes.to(value.device, torch.int64).contiguous()
    if level_start_index is None:
        hw = spatial_shapes.prod(-1)
        level_start_index = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    level_start_index = level_start_index.to(value.device, torch.int64).contiguous()

    value = value.contiguous()
    sampling_locations = sampling_locations.contiguous()
    attention_weights = attention_weights.contiguous()
    grad_out = grad_out.reshape(B, Q, M, D).contiguous()

    acc_dtype = torch.float32 if fp32_grad_accum else value.dtype
    grad_value = torch.zeros_like(value, dtype=acc_dtype)
    grad_loc = torch.empty_like(sampling_locations)
    grad_attn = torch.empty_like(attention_weights)

    grid = (triton.cdiv(Q, BLOCK_Q), B * M)
    msda_backward_kernel[grid](
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
