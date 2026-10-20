"""Metal (Apple GPU) port of the MSDA kernels, launched through PyTorch MPS.

Triton has no Apple GPU backend, so this module re-implements kernels.py in
Metal Shading Language and compiles it at runtime with
``torch.mps.compile_shader``. The math, the bilinear convention and the
memory layout are identical to the Triton kernels (see kernels.py).

Mapping. Each simdgroup (or a ``LANES``-lane slice of one, when D < 32) owns
one flat (b, q, m) query and its lanes stride over the head channels, so every
bilinear-corner gather is a coalesced row of D contiguous elements — the same
wide channel loads as the Triton query-block kernel. The per-sample scalars
(location, weight) are read by every lane at the same address, which the
hardware broadcasts. Only the compile-time shape constants (M, D, L, K and the
dtype) are baked into the source, one compiled library per combination.

Backward. ``grad_value`` is scattered with relaxed ``atomic_float`` adds into
an FP32 scratch buffer, always: Metal has no native half-precision atomics, so
this is the paper's FP32-accumulator path unconditionally. ``grad_loc`` and
``grad_attn`` need a sum over channels, done with a butterfly shuffle across
the lanes that share a query.

All flat indices are 64-bit (``long``). The dispatch grid is 32-bit, so very
large problems are split into chunks of queries, passed as a base offset.
"""

from __future__ import annotations

import functools

import torch

_MSL_DTYPE = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}

# Largest thread grid per dispatch; thread_position_in_grid is a uint32.
_MAX_THREADS = 1 << 31

_COMMON = """
#include <metal_stdlib>
using namespace metal;

typedef {t} T;
#define MSDA_M {m}
#define MSDA_D {d}
#define MSDA_L {l}
#define MSDA_K {k}
#define LANES {lanes}
#define CPT {cpt}

// prm = {{Q, S, total (B*Q*M), base}}: the flat (b, q, m) query handled by
// thread `tid` is base + tid / LANES.
"""

_FORWARD = """
kernel void msda_forward(
    device const T* value [[buffer(0)]],
    device const long* shapes [[buffer(1)]],
    device const long* starts [[buffer(2)]],
    device const T* loc [[buffer(3)]],
    device const T* attn [[buffer(4)]],
    device T* out [[buffer(5)]],
    device const long* prm [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const long Q = prm[0], S = prm[1], total = prm[2];
    const long pq = prm[3] + (long)(tid / LANES);
    if (pq >= total) return;
    const int lane = tid % LANES;
    const long b = pq / (Q * MSDA_M);
    const long m = pq % MSDA_M;
    const long vbase = (b * S * MSDA_M + m) * MSDA_D;

    float acc[CPT];
    for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;

    for (int l = 0; l < MSDA_L; ++l) {
        const long H = shapes[2 * l], W = shapes[2 * l + 1], start = starts[l];
        const float fh = (float)H, fw = (float)W;
        for (int k = 0; k < MSDA_K; ++k) {
            const long p = pq * (MSDA_L * MSDA_K) + l * MSDA_K + k;
            const float x = (float)loc[2 * p] * fw - 0.5f;
            const float y = (float)loc[2 * p + 1] * fh - 0.5f;
            const float a = (float)attn[p];
            const float x0f = floor(x), y0f = floor(y);
            const float lx = x - x0f, ly = y - y0f;
            // Clamp before the integer cast: anything past [-2, size + 1] has
            // all four corners out of bounds anyway.
            const long x0 = (long)clamp(x0f, -2.0f, fw + 1.0f);
            const long y0 = (long)clamp(y0f, -2.0f, fh + 1.0f);
            const long x1 = x0 + 1, y1 = y0 + 1;

            const bool vx0 = x0 >= 0 && x0 < W, vx1 = x1 >= 0 && x1 < W;
            const bool vy0 = y0 >= 0 && y0 < H, vy1 = y1 >= 0 && y1 < H;
            const float w00 = a * (1.0f - lx) * (1.0f - ly);
            const float w01 = a * lx * (1.0f - ly);
            const float w10 = a * (1.0f - lx) * ly;
            const float w11 = a * lx * ly;
            const long row0 = start + y0 * W, row1 = row0 + W;
            const long o00 = vbase + (row0 + x0) * (MSDA_M * MSDA_D);
            const long o01 = vbase + (row0 + x1) * (MSDA_M * MSDA_D);
            const long o10 = vbase + (row1 + x0) * (MSDA_M * MSDA_D);
            const long o11 = vbase + (row1 + x1) * (MSDA_M * MSDA_D);

            for (int c = 0; c < CPT; ++c) {
                const int d = lane + c * LANES;
                if (d >= MSDA_D) continue;
                float v = 0.0f;
                if (vy0 && vx0) v += w00 * (float)value[o00 + d];
                if (vy0 && vx1) v += w01 * (float)value[o01 + d];
                if (vy1 && vx0) v += w10 * (float)value[o10 + d];
                if (vy1 && vx1) v += w11 * (float)value[o11 + d];
                acc[c] += v;
            }
        }
    }

    for (int c = 0; c < CPT; ++c) {
        const int d = lane + c * LANES;
        if (d < MSDA_D) out[pq * MSDA_D + d] = (T)acc[c];
    }
}
"""

_BACKWARD = """
kernel void msda_backward(
    device const T* value [[buffer(0)]],
    device const long* shapes [[buffer(1)]],
    device const long* starts [[buffer(2)]],
    device const T* loc [[buffer(3)]],
    device const T* attn [[buffer(4)]],
    device const T* grad_out [[buffer(5)]],
    device atomic_float* grad_value [[buffer(6)]],
    device T* grad_loc [[buffer(7)]],
    device T* grad_attn [[buffer(8)]],
    device const long* prm [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    const long Q = prm[0], S = prm[1], total = prm[2];
    const long pq_raw = prm[3] + (long)(tid / LANES);
    // No early return: every lane of a simdgroup must reach the shuffles.
    // Out-of-range lanes read query 0 with g = 0 and never write.
    const bool valid = pq_raw < total;
    const long pq = valid ? pq_raw : 0;
    const int lane = tid % LANES;
    const long b = pq / (Q * MSDA_M);
    const long m = pq % MSDA_M;
    const long vbase = (b * S * MSDA_M + m) * MSDA_D;

    float g[CPT];
    for (int c = 0; c < CPT; ++c) {
        const int d = lane + c * LANES;
        g[c] = (valid && d < MSDA_D) ? (float)grad_out[pq * MSDA_D + d] : 0.0f;
    }

    for (int l = 0; l < MSDA_L; ++l) {
        const long H = shapes[2 * l], W = shapes[2 * l + 1], start = starts[l];
        const float fh = (float)H, fw = (float)W;
        for (int k = 0; k < MSDA_K; ++k) {
            const long p = pq * (MSDA_L * MSDA_K) + l * MSDA_K + k;
            const float x = (float)loc[2 * p] * fw - 0.5f;
            const float y = (float)loc[2 * p + 1] * fh - 0.5f;
            const float a = (float)attn[p];
            const float x0f = floor(x), y0f = floor(y);
            const float lx = x - x0f, ly = y - y0f;
            const long x0 = (long)clamp(x0f, -2.0f, fw + 1.0f);
            const long y0 = (long)clamp(y0f, -2.0f, fh + 1.0f);
            const long x1 = x0 + 1, y1 = y0 + 1;

            const bool vx0 = x0 >= 0 && x0 < W, vx1 = x1 >= 0 && x1 < W;
            const bool vy0 = y0 >= 0 && y0 < H, vy1 = y1 >= 0 && y1 < H;
            const bool ok00 = vy0 && vx0, ok01 = vy0 && vx1;
            const bool ok10 = vy1 && vx0, ok11 = vy1 && vx1;
            const float w00 = (1.0f - lx) * (1.0f - ly);
            const float w01 = lx * (1.0f - ly);
            const float w10 = (1.0f - lx) * ly;
            const float w11 = lx * ly;
            const long row0 = start + y0 * W, row1 = row0 + W;
            const long o00 = vbase + (row0 + x0) * (MSDA_M * MSDA_D);
            const long o01 = vbase + (row0 + x1) * (MSDA_M * MSDA_D);
            const long o10 = vbase + (row1 + x0) * (MSDA_M * MSDA_D);
            const long o11 = vbase + (row1 + x1) * (MSDA_M * MSDA_D);

            float sa = 0.0f, sx = 0.0f, sy = 0.0f;
            for (int c = 0; c < CPT; ++c) {
                const int d = lane + c * LANES;
                if (d >= MSDA_D) continue;
                const float v00 = ok00 ? (float)value[o00 + d] : 0.0f;
                const float v01 = ok01 ? (float)value[o01 + d] : 0.0f;
                const float v10 = ok10 ? (float)value[o10 + d] : 0.0f;
                const float v11 = ok11 ? (float)value[o11 + d] : 0.0f;

                // d(out)/d(attn) is the bilinearly sampled value itself;
                // d(out)/d(loc) is the analytic bilinear derivative.
                sa += g[c] * (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11);
                sx += g[c] * ((v01 - v00) * (1.0f - ly) + (v11 - v10) * ly);
                sy += g[c] * ((v10 - v00) * (1.0f - lx) + (v11 - v01) * lx);

                if (valid) {
                    const float wg = g[c] * a;
                    if (ok00) atomic_fetch_add_explicit(&grad_value[o00 + d], wg * w00, memory_order_relaxed);
                    if (ok01) atomic_fetch_add_explicit(&grad_value[o01 + d], wg * w01, memory_order_relaxed);
                    if (ok10) atomic_fetch_add_explicit(&grad_value[o10 + d], wg * w10, memory_order_relaxed);
                    if (ok11) atomic_fetch_add_explicit(&grad_value[o11 + d], wg * w11, memory_order_relaxed);
                }
            }

            // Sum over channels: butterfly across the LANES lanes of this query.
            for (int off = LANES / 2; off > 0; off >>= 1) {
                sa += simd_shuffle_xor(sa, (ushort)off);
                sx += simd_shuffle_xor(sx, (ushort)off);
                sy += simd_shuffle_xor(sy, (ushort)off);
            }
            if (valid && lane == 0) {
                grad_loc[2 * p] = (T)(a * fw * sx);
                grad_loc[2 * p + 1] = (T)(a * fh * sy);
                grad_attn[p] = (T)sa;
            }
        }
    }
}
"""


def _lanes_cpt(D: int) -> tuple[int, int]:
    """Lanes per query (a power of two <= 32, so queries tile a simdgroup and
    the xor-butterfly stays inside one query) and channels per lane."""
    lanes = min(32, 1 << (D - 1).bit_length())
    return lanes, -(-D // lanes)


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype, M: int, D: int, L: int, K: int):
    lanes, cpt = _lanes_cpt(D)
    header = _COMMON.format(t=_MSL_DTYPE[dtype], m=M, d=D, l=L, k=K, lanes=lanes, cpt=cpt)
    return torch.mps.compile_shader(header + _FORWARD + _BACKWARD)


@functools.lru_cache(maxsize=256)
def _params(Q: int, S: int, total: int, base: int) -> torch.Tensor:
    return torch.tensor([Q, S, total, base], dtype=torch.int64, device="mps")


def _launches(total: int, lanes: int):
    """Yield (base, threads) chunks of the flat query range, each within the
    32-bit grid limit and a whole number of simdgroups."""
    per = (_MAX_THREADS // lanes // 32) * 32
    for base in range(0, total, per):
        n = min(per, total - base)
        yield base, -(-n * lanes // 32) * 32


def forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    out = value.new_empty(B, Q, M, D)
    lib = _library(value.dtype, M, D, L, K)
    total = B * Q * M
    for base, threads in _launches(total, _lanes_cpt(D)[0]):
        lib.msda_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, out,
            _params(Q, S, total, base), threads=threads,
        )
    return out


def backward(value, spatial_shapes, level_start_index, sampling_locations,
             attention_weights, grad_out):
    """Returns (grad_value, grad_loc, grad_attn); grad_value in ``value.dtype``
    (accumulated in FP32 and cast once at the end)."""
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    grad_value = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
    grad_loc = torch.empty_like(sampling_locations)
    grad_attn = torch.empty_like(attention_weights)
    lib = _library(value.dtype, M, D, L, K)
    total = B * Q * M
    for base, threads in _launches(total, _lanes_cpt(D)[0]):
        lib.msda_backward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_out,
            grad_value, grad_loc, grad_attn,
            _params(Q, S, total, base), threads=threads,
        )
    return grad_value.to(value.dtype), grad_loc, grad_attn
