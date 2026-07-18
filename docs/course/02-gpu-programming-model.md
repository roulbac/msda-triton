# 2. How GPUs actually run code

!!! abstract "What you'll learn"
    The minimum GPU architecture needed to reason about kernel performance: how work is organized (threads, warps, blocks, SMs), how memory is organized (registers → caches → DRAM), what "memory-bound" means, and why *occupancy* — the metric everyone reaches for first — can point you in exactly the wrong direction.

You don't need to have written CUDA to follow this chapter. You do need the mental model it builds: chapters 4–6 lean on every concept here.

## One chip, thousands of tiny workers

A CPU is a few very smart cores, each racing through one instruction stream as fast as possible. A GPU is the opposite bet: **tens of thousands of simple execution lanes** that are individually slow but collectively enormous — *if* you can keep them all fed.

The hardware hierarchy, bottom-up:

- A **thread** is one logical worker with its own registers.
- Threads are executed in groups of 32 called **warps**. A warp executes *one instruction at a time, across all 32 threads* (SIMT: single instruction, multiple threads). If threads in a warp want to do different things — say, take different branches — the warp runs each path serially with some lanes switched off. Warps are the true unit of execution.
- Warps are grouped into **thread blocks** (a software choice, e.g. 128 threads = 4 warps). A block is guaranteed to run entirely on one…
- **Streaming Multiprocessor (SM)** — the GPU's "core". An A100 has 108 SMs; an H100 has 132. Each SM has its own register file, its own slice of fast on-chip memory, and its own warp schedulers.
- A **kernel launch** creates a **grid** of thread blocks. The hardware distributes blocks across SMs, and each SM juggles many resident warps at once.

The juggling is the key trick. When a warp issues a memory read, the data may take **hundreds of clock cycles** to arrive. A CPU would try to hide that with caches and speculation. An SM just *switches to a different warp* — zero-cost, every cycle. As long as an SM has enough resident warps that *someone* is always ready to run, memory latency is invisible. This is called **latency hiding**, and it's the entire reason GPUs want thousands of threads.

## The memory hierarchy: the numbers that matter

Everything in kernel optimization comes down to this table (approximate A100 figures):

| Level | Size | Bandwidth / latency | Who sees it |
|---|---|---|---|
| **Registers** | 256 KB *per SM* | ~1 cycle | one thread |
| **Shared memory / L1** | ~192 KB per SM | ~20–30 cycles | one thread block |
| **L2 cache** | 40 MB | ~200 cycles | whole GPU |
| **HBM (DRAM, "global memory")** | 40–80 GB | ~2 TB/s, ~400–800 cycles | whole GPU |

Two facts to internalize:

**1. Compute is nearly free; memory is not.** An A100 can do ~19.5 TFLOP/s of FP32 math but move only ~2 TB/s from DRAM. That's a ratio of ~40 floating-point operations per 4-byte float loaded. Any kernel doing *fewer* than ~40 ops per loaded float cannot be limited by arithmetic — it is **memory-bound**, and its best possible runtime is simply *bytes moved ÷ bandwidth*. Recall from chapter 1: MSDA does ~4 multiply-adds per loaded value. It is memory-bound by an order of magnitude. Optimizing its arithmetic is pointless; the only lever is **moving fewer bytes, and moving them well**.

**2. DRAM moves fixed-size chunks, not individual floats.** When any thread touches global memory, the hardware fetches an entire **32-byte sector** (and cache lines of 128 bytes). This gives rise to the single most important pattern in GPU programming:

### Memory coalescing

When the 32 threads of a warp read 32 *consecutive* 4-byte floats, the hardware notices the addresses are contiguous and merges ("**coalesces**") them into a handful of full-width transactions: 128 useful bytes per 128 bytes fetched. Perfect efficiency.

When the same warp reads 32 floats *scattered* across memory, each read pulls its own 32-byte sector to deliver 4 useful bytes — **8× more traffic for the same data**. Your *effective* bandwidth is now a fraction of the 2 TB/s on the spec sheet.

> Rule of thumb: **adjacent threads should read adjacent addresses.** Whether they do is a function of how you assign work to threads — which is exactly what "tiling" decisions are about, and the subject of chapter 4.

MSDA's reads are scattered *by nature* (the network decides where to sample). You can't make them contiguous — but you *can* choose which axis of the problem each thread walks, so that each unavoidable scattered access at least moves a full, useful chunk. Hold that thought.

## Occupancy — and why it can lie to you

**Occupancy** is the ratio of warps resident on an SM to the hardware maximum (64 warps on A100). High occupancy means many warps to switch between, so better latency hiding — and most profiling tools will wag a finger at you when it's low.

But resident warps compete for a fixed register file. A kernel whose threads each use lots of registers (say, to hold a large accumulator tile) fits *fewer* warps per SM → lower occupancy. So there's a real trade-off:

- **More, smaller threads** → high occupancy, great latency hiding, but each thread holds little state, so data gets re-loaded from memory instead of reused from registers.
- **Fewer, fatter threads** → low occupancy, but each thread keeps its working set in registers and issues *wide, efficient* memory transactions.

Occupancy is a *means* (hide latency), not an *end* (bandwidth actually delivered). A memory-bound kernel doesn't need many warps if each in-flight warp keeps the memory system saturated with big, useful requests.

!!! warning "The central empirical finding behind this repo"
    The paper this repo reproduces measured exactly this trade-off for MSDA (its §3.2). The "obvious" parallelization — one thread per (query, sample point), maximizing thread count — achieved **85% occupancy but only 5.1% of peak bandwidth**: every thread performed tiny scattered reads that wasted almost every byte fetched. The tiling used here achieves **17% occupancy but 36% of peak bandwidth** — ~7× more useful bytes per second, from a design profilers would flag as "bad". Occupancy is a proxy, and for scatter/gather operators it's a *misleading* one.

## Atomic operations: when threads write to the same place

One more concept, essential for the backward pass (chapter 5).

Threads mostly write to disjoint locations. But sometimes many threads must *accumulate into* the same address — as in MSDA's backward, where every query that sampled near pixel $p$ must add its contribution to $p$'s gradient. A plain `read; add; write` from two threads simultaneously loses updates (both read 5, both write 6, one +1 vanishes).

An **atomic add** makes read-modify-write indivisible — the hardware serializes concurrent updates to the same address, so nothing is lost. Modern GPUs execute atomic adds *in the L2 cache / memory subsystem itself*, which is remarkably fast — but three costs remain:

1. **Contention**: updates to the same address serialize. (Fine here — MSDA's collisions are sparse and incidental.)
2. **Instruction support depends on dtype and GPU generation**: some type/hardware combinations have a native atomic-add instruction; others must be *emulated* with a compare-and-swap retry loop that is roughly 10× slower. This turns out to be a major plot point (chapter 5).
3. **Memory ordering**: by default, atomics also act as synchronization points with ordering guarantees you may not need — and relaxing them (`sem="relaxed"`) is free performance *when it's safe*. Also chapter 5.

## The performance checklist

For any memory-bound GPU kernel, ask, in order:

1. **How many bytes *must* move?** (The algorithmic minimum — inputs read once, outputs written once.)
2. **How many bytes *actually* move?** (Wasted sectors from uncoalesced access? Same data loaded repeatedly?)
3. **Is the memory system kept busy?** (Enough parallelism / requests in flight — the real thing occupancy is a proxy for.)

Runtime ≈ actual bytes ÷ effective bandwidth. Every design decision in chapters 4 and 5 is an answer to question 2 or 3.

!!! summary "Key takeaways"
    - GPUs hide memory latency by switching between many resident warps; a warp is 32 threads executing in lockstep.
    - Compute is ~40× cheaper than memory traffic; MSDA (~4 ops/float) is deeply memory-bound, so only bytes moved matter.
    - DRAM traffic happens in 32-byte sectors: contiguous ("coalesced") access is up to 8× more efficient than scattered access.
    - Occupancy is a proxy for latency hiding, not a goal — for this operator, the low-occupancy/wide-loads design wins by ~7×.
    - Atomic adds enable concurrent accumulation; their speed depends critically on dtype and GPU generation.

[Next: Triton in one sitting →](03-triton-basics.md)
