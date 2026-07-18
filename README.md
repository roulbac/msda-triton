# msda-triton

**Fast Triton kernels for multi-scale deformable attention (MSDA)** — the core operator of Deformable DETR, DINO and Mask2Former.

[![CI](https://github.com/roulbac/msda-triton/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/roulbac/msda-triton/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-course%20%26%20guide-blue)](https://roulbac.github.io/msda-triton/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-2.9%2B-ee4c2c)](https://pytorch.org)

This repo reproduces the kernel design of the paper [*“Efficient Multi-Scale Deformable Attention on GPUs”*](https://openreview.net/forum?id=Q4jZ7zKNKx) (TMLR submission), and ships with a slow-but-simple reference implementation, a correctness test suite, and a [Modal](https://modal.com) benchmark harness that compares against the standard [mmcv](https://github.com/open-mmlab/mmcv) CUDA kernel.

> 📚 **New to Triton, GPUs, or MSDA?** There is a full companion course at
> **[roulbac.github.io/msda-triton](https://roulbac.github.io/msda-triton/)** that explains this
> entire repo from first principles — no GPU background assumed. It covers what deformable
> attention is, how GPUs execute code, how to write Triton kernels, and walks through both
> kernels in this repo line by line.

---

## Highlights

- ⚡ **Faster than the mmcv CUDA kernel** in nearly every measured configuration — up to **4.5×** on H100 (see [benchmarks](#results-decoder-preset-b4-q300-8001333-d256)).
- 🎯 **Drop-in replacement**: the signature mirrors mmcv's `MultiScaleDeformableAttnFunction`, so it slots into Deformable-DETR-family models directly.
- 🧮 **FP32, FP16 and BF16** support with FP32 internal accumulation, including a hardware-aware backward path for BF16 on pre-Hopper GPUs.
- ✅ **Fully differentiable** — all three analytic gradients (`grad_value`, `grad_sampling_locations`, `grad_attention_weights`) validated against a reference implementation.
- 🐍 **Pure Python/Triton** — no CUDA toolchain, no C++ extension build; the only dependency is a CUDA build of PyTorch.

## Installation

Requires Python ≥ 3.12 and a CUDA build of PyTorch ≥ 2.9 (which bundles Triton).

The project uses [uv](https://docs.astral.sh/uv/) (uv build backend, `uv.lock` pinning the full graph). `uv sync` installs the library plus the default dependency groups (`test`, `bench`, `docs`, `labs`):

```bash
uv sync                     # library + test/bench/docs/labs (default)
uv run pytest tests/ -v     # run any command inside the project env
```

It remains a standard PEP 517 package, so `pip install -e .` also works for a plain library install.

## Quickstart

```python
import torch
from msda_triton import multi_scale_deformable_attention

B, M, D, L, K, Q = 4, 8, 32, 4, 4, 300
shapes = [(100, 167), (50, 84), (25, 42), (13, 21)]        # (H_l, W_l)
S = sum(h * w for h, w in shapes)

value = torch.randn(B, S, M, D, device="cuda", dtype=torch.bfloat16)
spatial_shapes = torch.tensor(shapes, device="cuda")
sampling_locations = torch.rand(B, Q, M, L, K, 2, device="cuda", dtype=torch.bfloat16)
attention_weights = torch.rand(B, Q, M, L, K, device="cuda", dtype=torch.bfloat16)
attention_weights = attention_weights.flatten(3).softmax(-1).view(B, Q, M, L, K)

out = multi_scale_deformable_attention(
    value, spatial_shapes, None, sampling_locations, attention_weights
)   # (B, Q, M*D), fully differentiable
```

The signature mirrors mmcv's `MultiScaleDeformableAttnFunction` (`level_start_index` may be `None`; there is no `im2col_step`). FP32, FP16 and BF16 are supported; the kernel always accumulates in FP32 internally.

Slow reference implementations (`msda_reference`, grid_sample-based, and `msda_naive`, pure-Python loops) ship with the testing code in [`tests/reference_impls.py`](tests/reference_impls.py), not with the package.

## The operator

For query embeddings with reference points, MSDA replaces dense attention with a fixed number of bilinear lookups:

```math
\mathrm{MSDA}(q) = \sum_{m=1}^{M} W_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot x_l ( \phi_l(\hat{p}_q) + \Delta p_{mlqk} ) \right]
```

with $`M`$ heads, $`L`$ feature levels, $`K`$ sampling points, learned offsets $`\Delta p`$ and softmax-normalized weights $`A`$. This package implements the sampling/aggregation core (the bracketed term) — the linear projections around it are ordinary matmuls. Sampled positions are data-dependent, so the operator is dominated by scattered bilinear reads (forward) and scattered atomic gradient writes (backward), not by tileable matmuls.

*For a gentle, from-scratch explanation of this equation, see [chapter 1 of the course](https://roulbac.github.io/msda-triton/course/01-deformable-attention/).*

## Kernel design

The paper is a diagnostic study on A100/H100; its three load-bearing findings map directly onto this implementation:

1. **Query-block tiling, not point-parallel tiling** (paper §3.2). Each Triton program owns a contiguous block of queries for one `(batch, head)` pair, loads the `L·K` sampling parameters once per query (coalesced over the query dim), and gathers the four bilinear corners as wide loads over the channel dim, accumulating in FP32 registers. This runs at *lower* occupancy but ~7× higher effective bandwidth than the occupancy-maximizing tiling — occupancy is a misleading proxy for this operator. Sampling locations and attention weights are read through disjoint pointers, eliminating the reference implementation's concatenated `(B, Q, L, K, 3)` parameter buffer (§2.3).

2. **Relaxed-ordering atomics in the backward** (paper §3.3, Fig. 8). `grad_value` is scattered with `tl.atomic_add(..., sem="relaxed")` — safe because gradient accumulation is commutative and the kernel-end barrier provides the happens-before edge. On SM 9.0+ this lowers to the native hardware reduction instruction (`RED.ADD.BF16x8`).

3. **Accumulator-precision switch** (paper §3.3/§4.1). Pre-Hopper GPUs have no native **BF16** atomic add (`red.add.noftz.bf16` arrives with SM 9.0 / PTX ISA 7.8); the emulated compare-and-swap loop collapses BF16 backward bandwidth ~10×. For BF16 inputs on SM < 9.0, `grad_value` is therefore accumulated in an FP32 scratch buffer and downcast on return. FP16 is *not* routed through the scratch buffer: `red.add.noftz.f16x2` is hardware-native since SM 6.0, and native FP16 atomics measure ~2-4× faster than the FP32-accumulator path on Ampere/Ada. The default is auto-selected from `torch.cuda.get_device_capability()` and can be overridden with `fp32_grad_accum=True/False`.

Bilinear sampling uses the reference CUDA convention $`x_{\mathrm{im}} = x \cdot W_l - 0.5`$ (identical to `F.grid_sample(align_corners=False, padding_mode="zeros")`); out-of-bounds corners contribute zero, in both the forward and all gradients.

## Benchmarks

[`benchmarks/modal_benchmark.py`](benchmarks/modal_benchmark.py) provisions a GPU of your choice and reports, as a Rich console table, p50 forward/backward latency, peak memory, speedups vs mmcv, and max abs error vs the FP32 reference, for: the PyTorch reference, the same reference under `torch.compile`, mmcv's CUDA kernel, and the Triton kernel with both backward accumulator paths.

```bash
MSDA_GPU=H100 uv run modal run benchmarks/modal_benchmark.py                  # decoder preset
MSDA_GPU=RTX-PRO-6000 uv run modal run benchmarks/modal_benchmark.py           # RTX PRO 6000
MSDA_GPU=A100 uv run modal run benchmarks/modal_benchmark.py --preset encoder
uv run modal run benchmarks/modal_benchmark.py --preset encoder --resolution 1536x2048 --dtypes bf16
```

`MSDA_GPU` supports `L40S`, `A100`, `H100`, `H200`, and `RTX-PRO-6000` (default `A100`). Presets follow the paper's operating points: `decoder` (B=4, Q=300) and `encoder` (B=2, Q = all pyramid tokens) at 800×1333 by default, hidden dim 256 (8 heads × 32), L=4, K=4. mmcv ships no BF16 kernel, so its bf16 rows report the error.

### Results (decoder preset, B=4, Q=300, 800×1333, D=256)

Device kernel time (torch.profiler CUDA self-time per call, cross-checked against CUDA-graph replay), Triton speedup vs mmcv's CUDA kernel:

| GPU | fwd fp32 | fwd fp16 | fwd bf16 | bwd fp32 | bwd fp16 | bwd bf16 |
|---|---|---|---|---|---|---|
| L40S (SM 8.9) | 15µs, **1.42×** | 9µs, **3.27×** | 9µs, *n/a* | 324µs, 0.81× | 74µs, **2.86×** | 490µs, *n/a* |
| A100 (SM 8.0) | 35µs, **1.39×** | 17µs, **3.21×** | 18µs, *n/a* | 315µs, 0.82× | 132µs, **3.93×** | 411µs, *n/a* |
| H100 (SM 9.0) | 12µs, **2.12×** | 8µs, **4.25×** | 8µs, *n/a* | 92µs, **1.34×** | 56µs, **4.50×** | 56µs, *n/a* |
| RTX PRO 6000 (SM 12.0) | 13µs, **1.50×** | 8µs, **2.40×** | 8µs, *n/a* | 86µs, **1.21×** | 46µs, **4.03×** | 45µs, *n/a* |

*n/a*: mmcv ships no bf16 kernel; the bf16 backward row shows the auto-selected accumulator path — native atomics on SM ≥ 9.0, FP32 scratch buffer on pre-Hopper GPUs. mmcv still wins the fp32 backward on pre-Hopper GPUs, where its shared-memory pre-reduction (`col2im_...shm_blocksize_aware_reduce`) beats plain scattered atomics by ~20% — at fp16/bf16 that advantage is dwarfed by its slower half-precision atomic path.

> **⚠️ Measurement pitfalls.** Two pitfalls make these kernels easy to mis-benchmark, and both made this Triton kernel look *slower* than mmcv in earlier tables. At decoder scale the kernels are 10-30µs while Triton's Python dispatch is ~40µs/call vs ~20µs for mmcv's C++ op, so (1) synchronizing inside each timed iteration measures launch latency plus idle-clock execution (~4× inflated, ranking inverted), and (2) even back-to-back CUDA event pairs report max(dispatch, kernel) once the queue is CPU-bound. Hence the profiler-based device column plus the separate dispatch-inclusive `e2e` column in the benchmark output (see [Caveats](#caveats)). The [benchmarking chapter of the course](https://roulbac.github.io/msda-triton/course/06-benchmarking/) explains both pitfalls in depth.

<details>
<summary><b>How the benchmark image builds mmcv</b></summary>

The benchmark image runs on the project's own pinned torch (`torch>=2.9`, resolved to the latest release via `uv.lock`), not an old fixed version. OpenMMLab never published prebuilt mmcv wheels past torch2.4, so mmcv is instead built from source at image-build time, against that same torch, from a `nvidia/cuda-devel` base image (see the `mmcv` dependency group and `[tool.uv.sources]` / `[tool.uv.extra-build-variables]` / `[tool.uv.extra-build-dependencies]` in `pyproject.toml`, and `Image.uv_sync(...)` in the benchmark script). `TORCH_CUDA_ARCH_LIST` in `pyproject.toml` covers every supported GPU architecture, including SM 12.0 for `RTX-PRO-6000`. That build needs `nvcc`/`CUDA_HOME`, so the `mmcv` group is deliberately excluded from `uv sync`'s defaults — it's not needed to install or test the library itself.

</details>

## Correctness testing

The ground-truth chain is: hand-rolled loop implementation (`msda_naive`) validates the `grid_sample`-based reference on CPU; the FP32 reference then validates the Triton forward **and all three analytic gradients** (`grad_value`, `grad_sampling_locations`, `grad_attention_weights`) across dtypes, both grad-accumulator paths, out-of-bounds sampling locations, exact grid-corner locations, non-power-of-two head dims, and non-contiguous inputs.

```bash
uv run pytest tests/ -v                                    # on a CUDA machine
uv run modal run benchmarks/modal_benchmark.py --run-tests # or on a Modal GPU
```

CI (GitHub Actions) runs on every PR: the CPU-runnable subset of the suite (reference-vs-naive validation) plus an ahead-of-time compilation of every kernel variant for SM 8.0 and SM 9.0 ([`scripts/compile_kernels_check.py`](scripts/compile_kernels_check.py)), which type-checks the kernel bodies and reports their PTX atomic lowering without needing a GPU.

Note: because `grad_value` uses atomic accumulation, its backward is not bitwise deterministic run-to-run (tests use tolerances accordingly).

## Repository layout

```
src/msda_triton/kernels.py       # Triton forward/backward kernels (query-block tiling)
src/msda_triton/ops.py           # autograd wrapper, validation, accumulator switch
tests/reference_impls.py         # grid_sample reference + naive loop implementation
tests/test_msda.py               # correctness suite
benchmarks/modal_benchmark.py    # Modal benchmark + remote test runner
scripts/compile_kernels_check.py # GPU-less AOT compile check (used by CI)
docs/                            # companion course (GitHub Pages, mkdocs-material)
labs/                            # hands-on marimo labs: rebuild the kernel from scratch
.github/workflows/ci.yml         # CPU tests + kernel compile check on every PR
.github/workflows/docs.yml       # builds and deploys the docs site
```

## Caveats

- The Triton operator requires CUDA; on CPU use the reference implementation from `tests/reference_impls.py`.
- **Host-side dispatch is heavier than a C++ op.** Triton's Python launcher (JIT specialization + autotuner lookup) costs ~40µs per call vs ~20µs for mmcv's C++ extension. Device-side, the Triton forward beats mmcv at every decoder-scale operating point we measured, but at these kernel sizes (10-30µs) a drained CUDA queue makes the *wall-clock* per isolated call dispatch-bound. In real training the queue is full and this cost overlaps with GPU work; for latency-critical inference, the op is CUDA-graph capturable (zero Python overhead on replay). The benchmark reports device time and dispatch-inclusive e2e time as separate columns.
- `grad_value` accumulation is atomic and therefore nondeterministic at the bit level; with `fp32_grad_accum=True` it is order-independent up to FP32 rounding.
- FP16 sampling locations quantize coordinates to ~`W_l / 2048` pixels at large resolutions (inherent to the input dtype, shared by all implementations); prefer BF16/FP32 locations at encoder scale.
- The backward's grad_value atomics currently lower to scalar (per-element) relaxed atomics; the paper's 1D channel-tile listing gets Triton to emit the 8-wide vectorized form (`add.noftz.v8.bf16`) on SM 9.0. The hardware-vs-CAS lowering and the accumulator switch — the effects that dominate — are reproduced exactly (verified in PTX for SM 8.0/9.0); vectorizing the atomic is remaining headroom.

## Learn more

The companion site at **[roulbac.github.io/msda-triton](https://roulbac.github.io/msda-triton/)** is a self-contained course built around this repo:

| Chapter | What you'll learn |
|---|---|
| [1. Deformable attention, from scratch](https://roulbac.github.io/msda-triton/course/01-deformable-attention/) | Why dense attention fails on images, and what MSDA computes |
| [2. How GPUs actually run code](https://roulbac.github.io/msda-triton/course/02-gpu-programming-model/) | SMs, warps, memory hierarchy, and why bandwidth is everything here |
| [3. Triton in one sitting](https://roulbac.github.io/msda-triton/course/03-triton-basics/) | The block-program model, pointers, masks, autotuning |
| [4. The forward kernel](https://roulbac.github.io/msda-triton/course/04-forward-kernel/) | A line-by-line walkthrough of `_msda_forward_kernel` |
| [5. The backward kernel](https://roulbac.github.io/msda-triton/course/05-backward-kernel/) | Deriving the gradients by hand; atomics and the precision switch |
| [6. Benchmarking GPU code](https://roulbac.github.io/msda-triton/course/06-benchmarking/) | How to time kernels without fooling yourself |
| [7. Testing numerical kernels](https://roulbac.github.io/msda-triton/course/07-testing/) | Reference chains, tolerances, nondeterminism, GPU-less CI |
| [8. Exercises & further reading](https://roulbac.github.io/msda-triton/course/08-exercises/) | Hands-on modifications and where to go next |

### 🧪 Hands-on labs: rebuild the kernel yourself

For learning by *building*, [`labs/`](labs/README.md) contains a ten-part series of interactive [marimo](https://marimo.io) notebooks in the spirit of *From NAND to Tetris*: starting from bilinear interpolation in plain Python loops, you construct every layer — flat pointer arithmetic, a GPU memory-traffic simulator, your first Triton kernels, the full forward and backward kernels, the atomics/precision machinery — until **your own operator passes this repo's test suite**. Reference solutions ship as fallbacks, every stage is checked reactively against the repo's references, and the correctness track runs **without a GPU** via Triton's CPU interpreter.

```bash
uv sync --group labs
uv run marimo edit labs/01_the_spec.py
```

## References

- Zhu et al., [*Deformable DETR: Deformable Transformers for End-to-End Object Detection*](https://arxiv.org/abs/2010.04159), ICLR 2021 — the operator.
- [*Efficient Multi-Scale Deformable Attention on GPUs*](https://openreview.net/forum?id=Q4jZ7zKNKx), submitted to Transactions on Machine Learning Research (TMLR) — the kernel design reproduced here.
- [Triton documentation](https://triton-lang.org) — the compiler and language.
