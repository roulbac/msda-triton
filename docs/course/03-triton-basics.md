# 3. Triton in one sitting

!!! abstract "What you'll learn"
    Triton's programming model — programs instead of threads, block-valued operations, pointer arithmetic, masks — plus the practical machinery (`constexpr`, autotuning, launch grids) that the kernels in this repo use. After this chapter you can read real Triton code fluently.

## What Triton is

[Triton](https://triton-lang.org) is a Python-embedded language and compiler for writing GPU kernels. You write a decorated Python function operating on *blocks* of values; Triton JIT-compiles it to PTX (NVIDIA's GPU assembly) the first time it's called. PyTorch's CUDA builds bundle Triton — it's what `torch.compile` uses to generate kernels — so this repo needs **no CUDA toolchain, no C++ build step**: `kernels.py` is just importable Python.

The pitch, relative to CUDA:

| | CUDA | Triton |
|---|---|---|
| You program… | each individual **thread** | a whole **program** (≈ thread block) at once |
| Values are… | scalars per thread | **tensors of block size** (like NumPy arrays) |
| Coalescing, vectorization, warp layout | your job, by hand | **the compiler's job** |
| Shared-memory management | explicit `__shared__` + barriers | automatic |
| Build | nvcc, C++ | `@triton.jit` on a Python function |

The core abstraction shift: in CUDA you'd write "thread `i` loads element `i`" and *hope* your indexing coalesces. In Triton you write "this program loads *this block of 32 elements*" as a single operation, and the compiler maps it onto warps and picks vectorized instructions. You still choose **what** each program loads — that's the tiling decision, and it stays yours — but the mechanical layer below it is handled.

## Your first kernel

Every Triton tutorial starts with vector addition, and for good reason — it exhibits the full anatomy in 15 lines:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr,          # (1) pointers, not tensors
               n_elements,
               BLOCK_SIZE: tl.constexpr):       # (2) compile-time constant
    pid = tl.program_id(axis=0)                 # (3) which program am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # (4) my slice
    mask = offsets < n_elements                 # (5) guard the ragged edge
    x = tl.load(x_ptr + offsets, mask=mask)     # (6) block load
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)   # (7) launch grid
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

The seven ideas, which cover essentially all of Triton:

**(1) Kernels receive raw pointers.** Passing a PyTorch tensor to a `@triton.jit` function hands over the address of its first element. There are no shapes or strides inside the kernel — *you* compute every address by arithmetic. This is why the MSDA kernels' comments document each argument's layout (`# (B, S, M, D) contiguous`): the layout is a contract, enforced by the calling wrapper (`ops.py` calls `.contiguous()` on everything), not by the type system.

**(2) `tl.constexpr` arguments are compile-time constants.** Triton compiles a *separate specialized binary* for each distinct value. That lets the compiler unroll loops and size register allocations exactly — and it's why block dimensions must be constexpr. Mostly free, with one caveat you'll see in chapter 6: each specialization is a JIT compile, and the *lookup* of the right specialization happens in Python on every call.

**(3)–(4) The grid of programs.** A kernel launch spawns many independent **program instances**, distinguished only by `tl.program_id(axis)`. Each computes from its ID which slice of the problem it owns: program 0 takes elements 0–1023, program 1 takes 1024–2047, … The `grid` lambda (7) says how many programs to launch — here $\lceil n / \mathrm{BLOCK\_SIZE} \rceil$; grids can be up to 3D (the MSDA kernels use 2D: query-blocks × (batch·head)).

**(5) Masks handle everything that doesn't divide evenly.** `tl.arange(0, BLOCK_SIZE)` requires a power-of-2 size, and `n` won't generally be a multiple of it. Out-of-bounds lanes are switched off with a boolean mask: masked loads return `other=` (e.g. `0.0`) instead of reading memory, masked stores write nothing. In the MSDA kernels masks do double duty — the same mechanism that guards the last ragged query block also implements *zeros-padding for out-of-bounds sampling locations*: a bilinear corner outside the image is just a masked-off lane.

**(6) Loads and stores are block-valued.** `tl.load(ptr + offsets)` with a block of 1024 offsets produces a 1024-element value in one operation. Arithmetic on blocks broadcasts like NumPy — including the indexing idiom you'll see constantly in this repo:

```python
offs_q = tl.arange(0, BLOCK_Q)          # (BLOCK_Q,)   queries
offs_d = tl.arange(0, BLOCK_D)          # (BLOCK_D,)   channels
ptrs = base + rows[:, None] * stride + offs_d[None, :]   # (BLOCK_Q, BLOCK_D)
tile = tl.load(ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
```

`[:, None]` / `[None, :]` build a 2D block of addresses from two 1D ranges — a whole (queries × channels) tile loaded in one statement. When the addresses in a row are contiguous (here: consecutive channels), the compiler emits wide vectorized loads — coalescing falls out of the access pattern you wrote.

## The pieces the MSDA kernels add

Four more constructs and you can read `kernels.py` end to end.

**`tl.static_range` — compile-time loop unrolling.** `for l in tl.static_range(L)` with `L: tl.constexpr` is fully unrolled at compile time. The MSDA kernels unroll over levels and sample points ($L \times K = 16$ iterations), so the compiled kernel is straight-line code with no loop overhead, and every trip's loads can be in flight simultaneously — more memory-level parallelism (checklist item 3 from chapter 2!).

**Accumulate in FP32 registers, whatever the I/O dtype.**

```python
acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
...                                   # loads .to(tl.float32), accumulate
tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=...)
```

Inputs may be fp16/bf16 (half the bytes → half the memory traffic → nearly 2× faster for a memory-bound kernel), but summing 128 terms in bf16 (~8 bits of mantissa) would destroy precision. The accumulator lives in fp32 *registers* — precision costs registers, not memory traffic. `out_ptr.dtype.element_ty` reads the output dtype off the pointer, so one kernel source serves all three dtypes.

**`tl.atomic_add(ptr, val, mask=..., sem="relaxed")`** — the block-valued atomic scatter used by the backward. Full story in chapter 5.

**Autotuning.** The best `BLOCK_Q` / warp count depends on the GPU and problem size, so the forward kernel declares candidates and lets Triton time them:

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_Q": bq}, num_warps=nw)
             for bq in (16, 32, 64) for nw in (2, 4)],
    key=["Q", "M", "D", "L", "K"],     # re-tune when these change
)
@triton.jit
def _msda_forward_kernel(...):
```

First call with a given `key`: Triton benchmarks all six configs and caches the winner. `num_warps` sets how many warps execute each program — how the block-valued operations spread across hardware threads.

!!! note "Why the backward kernel is *not* autotuned"
    Autotuning *times the kernel repeatedly*. The backward kernel **accumulates into `grad_value` with atomics** — running it twice doubles the answer. Correct autotuning would need the output reset between timings (Triton's `reset_to_zero` exists but adds complexity and tuning-time cost). This repo instead fixes the backward launch config (`BWD_BLOCK_Q = 16`, 4 warps), selected by an offline sweep on L40S/A100/H100. A useful pattern to remember: **kernels with side effects don't compose naively with autotuning.**

## What Triton costs you

Two honest limitations, both of which shape this repo:

- **Python launch overhead.** Every call goes through Triton's Python launcher — argument specialization, autotuner cache lookup — costing ~40µs versus ~20µs for a C++ op. Irrelevant when kernels are big or the GPU queue is full; dominant when timing a 10µs kernel in isolation. This single fact drives most of chapter 6.
- **You control less of the final instruction selection.** Triton decides how block operations lower to instructions. Usually it does well; occasionally it misses — e.g. the backward's atomic adds currently lower to scalar (per-element) atomics where a hand-written CUDA kernel (or a differently-shaped Triton listing) gets the 8-wide vectorized form. Known headroom, documented in the README's caveats.

!!! summary "Key takeaways"
    - Triton programs operate on *blocks* of values; the compiler handles warp mapping, coalescing, vectorization. Tiling strategy remains your job.
    - Kernels see raw pointers — layout is a contract with the caller; all addresses are computed by hand, and 2D tiles come from `[:, None]`-style broadcast.
    - Masks turn boundary conditions *and* out-of-bounds sampling into switched-off lanes.
    - `constexpr` + `static_range` = specialization and full unrolling; accumulate in fp32 registers regardless of I/O dtype.
    - Autotune pure kernels; side-effecting (atomic-accumulating) kernels need fixed or carefully-reset configs.

[Next: The forward kernel, line by line →](04-forward-kernel.md)
