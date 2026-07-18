# msda-triton: the course

**Fast Triton kernels for multi-scale deformable attention — and a complete course explaining how they work.**

This site is the companion documentation for [`roulbac/msda-triton`](https://github.com/roulbac/msda-triton), a pure-Python/Triton implementation of **multi-scale deformable attention (MSDA)**, the core operator of Deformable DETR, DINO and Mask2Former. The kernels beat the standard mmcv CUDA implementation by up to **4.5×** while requiring no CUDA toolchain at all.

If you just want to *use* the operator, the [README](https://github.com/roulbac/msda-triton#readme) has everything: install, quickstart, benchmark numbers, caveats.

This site is for something else: **understanding it**.

## Who this course is for

You are a deep learning engineer. You are comfortable with PyTorch, tensors, autograd, and training loops. But:

- You've never written a GPU kernel — Triton or CUDA.
- Terms like *warp*, *occupancy*, *memory coalescing* and *atomic add* are fuzzy or unknown.
- You may not know what deformable attention actually computes, beyond "it's in DETR-like models".

That's exactly the starting point this course assumes. By the end, you will be able to read every line of the two kernels in this repo and explain *why* each design decision was made — and you'll have the mental toolkit to write and benchmark your own Triton kernels.

## Why learn this through one operator?

Most GPU-programming tutorials teach you matrix multiplication. That's a great first kernel, but it teaches a narrow lesson: how to tile compute-bound problems.

MSDA is the opposite — and in some ways a *better* teacher:

- It is **memory-bound**: performance is entirely about how you move bytes, not how you schedule math. Most real-world custom ops look like this.
- Its memory accesses are **data-dependent** (the network *learns* where to read), so the classic tricks don't apply and you must reason about access patterns from first principles.
- Its backward pass requires **scattered atomic writes** — a topic most tutorials skip entirely, and the source of the most interesting hardware story in this repo (why BF16 gradients were ~10× slower on A100 than H100, and how to fix it).
- It's small enough to hold in your head: the forward kernel is ~100 lines.

## The course

Work through the chapters in order — each builds on the previous one.

| | Chapter | You'll learn |
|---|---|---|
| 1 | [Deformable attention, from scratch](course/01-deformable-attention.md) | Why dense attention fails on images; what MSDA computes; bilinear interpolation |
| 2 | [How GPUs actually run code](course/02-gpu-programming-model.md) | SMs, warps, the memory hierarchy, memory-bound vs compute-bound, occupancy |
| 3 | [Triton in one sitting](course/03-triton-basics.md) | The block-program model, pointers, masks, autotuning — your first kernel |
| 4 | [The forward kernel, line by line](course/04-forward-kernel.md) | Query-block tiling, and why *lower* occupancy made it 7× faster |
| 5 | [The backward kernel](course/05-backward-kernel.md) | Deriving three gradients by hand; atomics, memory ordering, and a precision switch |
| 6 | [Benchmarking without fooling yourself](course/06-benchmarking.md) | The two pitfalls that made a faster kernel look slower |
| 7 | [Testing numerical kernels](course/07-testing.md) | Reference chains, tolerances, nondeterminism, and GPU-less CI |
| 8 | [Exercises & further reading](course/08-exercises.md) | Hands-on modifications, and where to go next |

There's also a [glossary](glossary.md) of every technical term used in the course.

**Prefer building to reading?** The repo also ships [**ten hands-on labs**](labs.md) — marimo notebooks in the *From NAND to Tetris* spirit where you rebuild the optimized kernel yourself, stage by checked stage, no GPU required for the correctness track. Course and labs cross-reference each other chapter by chapter.

!!! tip "How to read this course"
    Chapters 1–3 are background: skip any you already know. Chapters 4–7 are the heart of the course — they walk through the *actual code in this repo*, so keep [`kernels.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/kernels.py) and [`ops.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/ops.py) open in another tab.

## The paper behind the design

The kernel design reproduces the findings of *“Efficient Multi-Scale Deformable Attention on GPUs”* (a TMLR submission): a diagnostic study on A100/H100 whose three load-bearing findings — query-block tiling, relaxed-ordering atomics, and an accumulator-precision switch — map one-to-one onto the code here. Chapters 4 and 5 explain each finding in context.
