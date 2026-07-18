# Glossary

Every technical term used in the course, in plain language. Chapter links point to where the term is explained in context.

### Atomic operation
A read-modify-write (e.g. "add 3 to this address") the hardware guarantees is indivisible — concurrent updates from many threads serialize instead of overwriting each other. [Ch. 2](course/02-gpu-programming-model.md), [Ch. 5](course/05-backward-kernel.md)

### Autotuning
Automatically benchmarking several launch configurations (block sizes, warp counts) for a kernel and caching the fastest. Unsafe out-of-the-box for kernels that accumulate into their output. [Ch. 3](course/03-triton-basics.md)

### BF16 / bfloat16
16-bit float with fp32's exponent range but only ~8 bits of mantissa. Halves memory traffic vs fp32; its *atomic add* is only hardware-native from SM 9.0 (Hopper). [Ch. 5](course/05-backward-kernel.md)

### Bilinear interpolation
Reading a 2D grid at a fractional coordinate by blending the 4 surrounding pixels with weights that sum to 1. The core primitive of MSDA. [Ch. 1](course/01-deformable-attention.md)

### CAS (compare-and-swap)
Primitive atomic: "replace the value at this address with X, but only if it still equals Y." Used in a retry loop to *emulate* atomic operations the hardware lacks — ~10× slower than a native atomic add under load. [Ch. 5](course/05-backward-kernel.md)

### Coalescing
The hardware merging a warp's accesses to *consecutive addresses* into few full-width memory transactions. The difference between using ~100% and ~12% of the bytes you fetch. [Ch. 2](course/02-gpu-programming-model.md)

### `constexpr` (Triton)
A kernel argument fixed at compile time; Triton compiles a specialized binary per distinct value, enabling loop unrolling and exact register sizing. [Ch. 3](course/03-triton-basics.md)

### CUDA event
GPU-recorded timestamp inserted into the command stream; the standard kernel-timing tool — misleading when the queue is CPU-bound by dispatch. [Ch. 6](course/06-benchmarking.md)

### CUDA graph
A recorded sequence of kernel launches replayed with a single driver call, eliminating per-kernel host dispatch overhead. [Ch. 6](course/06-benchmarking.md)

### Feature pyramid
The same image's features at several resolutions (e.g. strides 8/16/32/64), letting a detector see both small and large objects. The "multi-scale" in MSDA. [Ch. 1](course/01-deformable-attention.md)

### Gather / scatter
Reading from (gather) or writing to (scatter) memory addresses that are data-dependent and irregular. MSDA gathers in the forward pass and scatters gradients in the backward. [Ch. 4](course/04-forward-kernel.md), [Ch. 5](course/05-backward-kernel.md)

### Grid (of programs)
The set of independent program instances created by one kernel launch, indexed by `tl.program_id`. [Ch. 3](course/03-triton-basics.md)

### Happens-before edge
A synchronization guarantee that all memory effects of one piece of work are visible before another begins. Kernel completion provides one — which is why relaxed atomics suffice inside the backward. [Ch. 5](course/05-backward-kernel.md)

### HBM / global memory
The GPU's main DRAM (tens of GB, ~2–3 TB/s, hundreds of cycles of latency). Where all tensors live; the bottleneck for memory-bound kernels. [Ch. 2](course/02-gpu-programming-model.md)

### Latency hiding
Keeping the GPU busy during slow memory accesses by switching among many resident warps instead of waiting. [Ch. 2](course/02-gpu-programming-model.md)

### Launch (dispatch) overhead
Host-side cost of getting a kernel running: Python/C++ wrapper, driver submission (~20–40µs). Dominates wall clock when kernels are smaller than it. [Ch. 6](course/06-benchmarking.md)

### Mask (Triton)
Boolean block passed to `tl.load`/`tl.store`/`tl.atomic_add` switching individual lanes off; handles ragged edges, non-power-of-2 dims, and out-of-bounds sampling. [Ch. 3](course/03-triton-basics.md)

### Memory-bound
Runtime limited by bytes moved, not arithmetic; best possible time = bytes ÷ bandwidth. MSDA does ~4 multiply-adds per loaded float — deeply memory-bound. [Ch. 2](course/02-gpu-programming-model.md)

### Memory ordering / `sem="relaxed"`
The guarantees an atomic gives about how *other* memory operations may be ordered around it. "Relaxed" = indivisibility only, no fencing — free performance whenever the accumulation is commutative and consumed only after the kernel ends. [Ch. 5](course/05-backward-kernel.md)

### MSDA (multi-scale deformable attention)
Attention where each query reads a fixed number of *learned* bilinear sample points from a feature pyramid instead of comparing against every key. Core operator of Deformable DETR / DINO / Mask2Former. [Ch. 1](course/01-deformable-attention.md)

### Occupancy
Resident warps per SM as a fraction of the hardware maximum. A proxy for latency hiding — famously misleading for gather/scatter kernels (this repo's design: 17% occupancy, 7× the bandwidth). [Ch. 2](course/02-gpu-programming-model.md)

### Program (Triton)
Triton's unit of work: one instance of the kernel body operating on whole blocks of values; corresponds to a CUDA thread block. [Ch. 3](course/03-triton-basics.md)

### PTX
NVIDIA's portable GPU assembly, what Triton compiles to. Readable with effort — and worth reading: it's where you see whether an atomic lowered to a native `red.add` or a CAS loop. [Ch. 5](course/05-backward-kernel.md), [Ch. 7](course/07-testing.md)

### Reference implementation
A slow, obviously-correct version of an operator used as testing ground truth; best arranged in a chain from eyeball-verifiable to fast. [Ch. 7](course/07-testing.md)

### SM (streaming multiprocessor)
The GPU's "core" (108 on A100): register file, shared memory, warp schedulers. Thread blocks are assigned to SMs. [Ch. 2](course/02-gpu-programming-model.md)

### SM version / compute capability
Hardware generation number (A100 = SM 8.0, L40S = 8.9, H100 = 9.0) determining which instructions exist — e.g. native bf16 atomic add requires ≥ 9.0. [Ch. 5](course/05-backward-kernel.md)

### Tiling
The choice of which slice of the problem each program owns. *The* performance decision for a kernel; this repo's query-block tiling vs the naive point-parallel tiling is a ~7× effective-bandwidth difference. [Ch. 4](course/04-forward-kernel.md)

### Warp
32 threads executing one instruction in lockstep (SIMT); the true unit of GPU execution and of memory-coalescing analysis. [Ch. 2](course/02-gpu-programming-model.md)
