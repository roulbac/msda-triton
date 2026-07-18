# 4. The forward kernel, line by line

!!! abstract "What you'll learn"
    The complete forward kernel — [`_msda_forward_kernel` in `kernels.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/kernels.py) — from work decomposition to the final store, and *why* its tiling beats the "obvious" one by ~7× in effective bandwidth (the paper's central §3.2 finding).

## The design question: what does one program own?

The forward pass must compute, for every (batch $b$, query $q$, head $m$): sample $L \times K = 16$ locations bilinearly, weight, and sum — producing one $D$-dimensional output vector. With $B{=}4$, $Q{=}300$, $M{=}8$: 9,600 independent output vectors, each needing 64 corner reads of $D{=}32$ channels.

How do we split this across GPU programs? Two candidate decompositions:

**Point-parallel (the "obvious" one — maximize parallelism).** One thread per (query, head, level, point) — 153,600 tiny work items, sky-high occupancy. This is roughly what a naive port of the operator produces. The problem: each thread bilinearly samples *one* location, reading 4 corners × D channels *by itself*, and then all $L\times K$ threads for a query must *combine* their partial sums (via shared memory or atomics). Threads in a warp are working on *different sample points at unrelated addresses* — so nothing coalesces, every load wastes most of its 32-byte sector, and the combination step adds synchronization traffic. This achieved **85% occupancy but 5.1% of peak bandwidth** in the paper's measurement.

**Query-block tiling (what this repo does).** One program owns a *block of `BLOCK_Q` consecutive queries* for one $(b, m)$ pair, and computes their **entire** output — all levels, all points — privately in registers:

```python
grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_Q"]), B * M)
```

A 2D grid: axis 0 tiles the queries, axis 1 enumerates (batch, head) pairs. No cross-program communication, no combination step. Occupancy is *lower* (each program is fat: it holds a `(BLOCK_Q, BLOCK_D)` fp32 accumulator in registers), but every memory access is shaped well — as we'll now see. **17% occupancy, 36% of peak bandwidth: ~7× the effective throughput.**

The general lesson: for gather-heavy operators, pick the decomposition where *the unavoidable scattered access moves the largest useful contiguous chunk*, and keep reductions private to one program.

## Setup: who am I, and where is my data?

```python
pid_q = tl.program_id(0)
pid_bm = tl.program_id(1)
b = (pid_bm // M).to(tl.int64)
m = pid_bm % M

offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
offs_d = tl.arange(0, BLOCK_D)
mask_q = offs_q < Q          # last query block may be ragged
mask_d = offs_d < D          # BLOCK_D = next_power_of_2(D); D itself may not be
```

Two index vectors — queries owned (`offs_q`) and channels (`offs_d`) — plus their validity masks. `BLOCK_D` is `triton.next_power_of_2(D)` (set by the wrapper in `ops.py`), because `tl.arange` needs powers of two; for $D{=}32$ the channel mask is all-true, but the kernel also handles odd head dims like $D{=}48$ correctly via `mask_d`.

!!! warning "The `int64` casts are load-bearing"
    Flat indices are computed in 64-bit arithmetic (`.to(tl.int64)`) because at *encoder* scale $S$ reaches ~10⁶ tokens and the flat element index $b \cdot S \cdot M \cdot D$ overflows a signed 32-bit integer (2.1 billion). Silent index overflow is a classic custom-kernel bug: everything works at toy scale and reads garbage at production scale.

```python
pq = (b * Q + offs_q) * M + m
val_base = value_ptr + (b * S * M + m) * D + offs_d[None, :]
```

`pq` is the flat $(b, q, m)$ index — shared by the sampling-location and attention-weight layouts, since both are `(B, Q, M, L, K, …)`. `val_base` fixes the batch, head, and channel offsets into `value` once; only the pixel index will vary per load.

Note that locations and weights are read through **disjoint pointers** (`loc_ptr`, `attn_ptr`). The standard PyTorch reference materializes a concatenated `(B, Q, L, K, 3)` buffer holding (x, y, weight) triples — an extra copy of every sampling parameter, written and re-read through DRAM. Passing the two tensors separately deletes that traffic entirely (paper §2.3).

## The main loop: load parameters once, coalesced

```python
acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

for l in tl.static_range(L):
    H = tl.load(shapes_ptr + 2 * l)
    W = tl.load(shapes_ptr + 2 * l + 1)
    start = tl.load(starts_ptr + l)          # where level l begins in the S axis
    ...
    for k in tl.static_range(K):
        p = pq * (L * K) + l * K + k
        loc_x = tl.load(loc_ptr + 2 * p,     mask=mask_q, other=0.0).to(tl.float32)
        loc_y = tl.load(loc_ptr + 2 * p + 1, mask=mask_q, other=0.0).to(tl.float32)
        attn  = tl.load(attn_ptr + p,        mask=mask_q, other=0.0).to(tl.float32)
```

Both loops are `tl.static_range` — fully unrolled at compile time (chapter 3), so all 16 (level, point) iterations become straight-line code whose loads can overlap.

Look at the shape of the parameter loads: for fixed $(l, k)$, `p` varies only through `offs_q` — **`BLOCK_Q` consecutive queries read `BLOCK_Q` nearly-consecutive addresses** (stride $L \cdot K \cdot M$ elements, a constant). Each sampling parameter is loaded exactly once per query per kernel, in a warp-friendly pattern. Contrast with point-parallel tiling, where the per-query parameters are re-fetched by every one of the threads that shares the query.

## Bilinear sampling: the same math as the naive loop

```python
x = loc_x * fw - 0.5                 # normalized [0,1] -> pixel coordinates
y = loc_y * fh - 0.5
x0f = tl.math.floor(x); y0f = tl.math.floor(y)
lx = x - x0f;           ly = y - y0f          # fractional parts
x0 = x0f.to(tl.int64);  y0 = y0f.to(tl.int64)
x1 = x0 + 1;            y1 = y0 + 1
```

Line for line the math from chapter 1 — except every variable is a *vector over `BLOCK_Q` queries*. One program computes 16 (or 32, 64) bilinear setups simultaneously.

```python
vx0 = mask_q & (x0 >= 0) & (x0 < W)
vx1 = mask_q & (x1 >= 0) & (x1 < W)
vy0 = (y0 >= 0) & (y0 < H)
vy1 = (y1 >= 0) & (y1 < H)
```

Here is zeros-padding as masks: a corner outside the image is an invalid lane. The naive loop's `if 0 <= xx < w and 0 <= yy < h:` became four boolean vectors — no branches, just switched-off loads. (Because a sample point's four corners have only two distinct x's and two y's, four checks cover all four corners.)

## The corner gathers: scattered rows, wide loads

The heart of the kernel:

```python
row0 = start + y0 * W
row1 = row0 + W

v00 = tl.load(
    val_base + ((row0 + x0) * (M * D))[:, None],     # (BLOCK_Q, BLOCK_D) addresses
    mask=(vy0 & vx0)[:, None] & mask_d[None, :],
    other=0.0,                                        # out-of-bounds -> zero
).to(tl.float32)
# ... v01, v10, v11 likewise
```

Unpack the address: `val_base` already contains batch, head and channel; `(row0 + x0) * (M*D)` selects the pixel. The result is a 2D block of addresses — `BLOCK_Q` rows (one per query, each at some *unpredictable* pixel) × `BLOCK_D` columns (**consecutive channels**).

This is the payoff of the whole design. The row addresses are scattered — nothing can fix that, the network chose them — but each row is a **contiguous $D$-channel vector**: for $D{=}32$ at 2 bytes/element, a 64-byte contiguous run, i.e. two full 32-byte DRAM sectors with **zero wasted bytes**. The unavoidable scatter got amortized over the widest useful contiguous unit the problem offers. Point-parallel tiling makes the same number of scattered accesses but wastes most of every sector.

## Accumulate and store

```python
acc += (
    v00 * (attn * (1.0 - lx) * (1.0 - ly))[:, None]
  + v01 * (attn * lx * (1.0 - ly))[:, None]
  + v10 * (attn * (1.0 - lx) * ly)[:, None]
  + v11 * (attn * lx * ly)[:, None]
)
```

The four bilinear weights are fused with the attention weight (one scalar per query, broadcast over channels) and accumulated in fp32 registers. After all 16 unrolled iterations:

```python
out_ptrs = out_ptr + pq[:, None] * D + offs_d[None, :]
tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_q[:, None] & mask_d[None, :])
```

One perfectly coalesced tile store per program — consecutive queries × consecutive channels — downcast to the output dtype only at the very end.

## The wrapper: `ops.py`

The kernel trusts its layout contract; [`ops.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/ops.py) enforces it. `multi_scale_deformable_attention` validates shapes and dtypes with real error messages, computes `level_start_index` from `spatial_shapes` if the caller passed `None`, calls `.contiguous()` on every tensor (a no-op if already contiguous — this is how non-contiguous inputs "just work"), handles $B{=}0$/$Q{=}0$ edge cases without launching, and wraps the kernels in a `torch.autograd.Function` so the operator composes with autograd. That last part is chapter 5's job.

## Scorecard against the chapter-2 checklist

1. **Bytes that must move**: each corner read is inherent to the operator; parameters and outputs read/written once. ✅
2. **Bytes actually moved**: no concatenated parameter buffer (disjoint pointers); parameters loaded once per query, coalesced over queries; corner gathers waste no sector bytes (full channel rows); fp16/bf16 I/O halves traffic while fp32 lives in registers. ✅
3. **Memory system kept busy**: 16 unrolled iterations × 4 corner loads of wide tiles per program give plenty of requests in flight — even at 17% occupancy. ✅

!!! summary "Key takeaways"
    - Decomposition: one program = one (batch, head) × one block of queries, whole output computed privately in registers. No inter-program reduction.
    - Query-block tiling makes parameter loads coalesced over the query dim and corner gathers *wide* over the channel dim — the scatter is unavoidable, wasted bytes are not.
    - Low occupancy + wide loads beat high occupancy + narrow loads by ~7× effective bandwidth for this operator.
    - Everything ships as masks: ragged edges, non-power-of-2 head dims, and zeros-padding for out-of-bounds samples.
    - 64-bit flat indexing is mandatory at encoder scale; the Python wrapper owns validation and the layout contract.

[Next: The backward kernel →](05-backward-kernel.md)
