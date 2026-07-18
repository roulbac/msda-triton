# 8. Exercises & further reading

The fastest way to make this material stick is to modify working code and watch what happens. All exercises use this repo; a GPU helps but several need none at all. Set up with:

```bash
git clone https://github.com/roulbac/msda-triton && cd msda-triton
uv sync
uv run pytest tests/ -v        # GPU-only tests skip themselves on CPU
```

No local GPU? The Modal harness gives you one per-second: `MSDA_GPU=A100 uv run modal run benchmarks/modal_benchmark.py`.

## Warm-ups (no GPU needed)

**1. Re-derive the gradients.** Close chapter 5 and derive $\partial \mathcal{L}/\partial A_{lk}$, $\partial \mathcal{L}/\partial x$, and the four corner contributions to $\partial \mathcal{L}/\partial v$ from the forward equation. Check yourself against the `ga`, `gx/gy`, and `wg * w00` lines of `_msda_backward_kernel`. If your $\partial/\partial x$ is missing a factor of $W_l$, re-read the chain rule through $x_{\mathrm{im}} = x W_l - 0.5$.

**2. Break the reference, watch the chain catch it.** In `tests/reference_impls.py`, change `msda_naive`'s `- 0.5` to `- 0.4` and run `uv run pytest tests/ -k naive`. Then revert, and instead flip a corner weight in the same function. The point: convince yourself the naive-vs-reference link actually pins the sampling convention.

**3. Read the PTX.** Run `uv run python scripts/compile_kernels_check.py` (CPU-only, ~a minute). Find the atomic instructions in its output for: bf16-native on SM 8.0 vs SM 9.0, and the fp32-accumulator variant on both. Match what you see to chapter 5's table.

**4. Predict before you measure.** Using chapter 2's bandwidth arithmetic: at the decoder preset ($B{=}4, Q{=}300, M{=}8, D{=}32, L{=}K{=}4$, fp16), estimate the forward's minimum bytes moved (corner reads + parameters + output) and divide by an H100's ~3 TB/s. Compare with the measured 8µs. How much headroom does the number suggest?

## Kernel modifications (GPU required)

**5. Sabotage the coalescing.** In `_msda_forward_kernel`, the corner gathers are wide over channels because `offs_d` is the fast axis. Transpose the tile: make each program load `(BLOCK_D, BLOCK_Q)` with queries fast, channels slow (you'll need to transpose the accumulator and store too). Benchmark before/after. You've just re-created the point-parallel pathology from chapter 4 in miniature.

**6. Kill the `static_range` unrolling.** Replace `tl.static_range(L)` / `(K)` with `tl.range(...)` (a real loop) and benchmark. How much did memory-level parallelism from unrolling buy?

**7. Sweep the backward launch config.** `BWD_BLOCK_Q`/`BWD_NUM_WARPS` are fixed at (16, 4) by an offline sweep. Reproduce it: try (16, 32, 64) × (2, 4, 8) on your GPU (`benchmarks/modal_variants.py::bwd_sweep` is the harness) . Does your GPU agree with the shipped choice? Why can't you just slap `@triton.autotune` on this kernel? (Chapter 3 has the answer; try to state it precisely.)

**8. Measure the two benchmarking pitfalls.** Write the *wrong* benchmark on purpose: time the forward with `torch.cuda.synchronize()` inside the loop, then with CUDA events on a drained queue, then compare against the harness's profiler numbers. Reproduce the ~4× inflation from chapter 6 on your own hardware.

**9. Force the wrong atomic path.** On an A100 (or any pre-Hopper GPU), run the bf16 backward with `fp32_grad_accum=False` explicitly and compare against the default. You're measuring the CAS-emulation penalty directly. On an H100, verify the two paths are close instead.

## Projects

**10. Vectorize the backward atomics.** The README's last caveat is an open invitation: the backward's `grad_value` atomics currently lower to scalar per-element atomics, while the paper's 1D channel-tile formulation coaxes Triton into the 8-wide `red.add.noftz.v8.bf16` form on SM 9.0. Restructure the backward's scatter (one flat channel-tile axis instead of a 2D `(q, d)` tile) to get the vectorized lowering, verify with `compile_kernels_check.py`, and benchmark. This is real, unclaimed headroom in the repo.

**11. Port the recipe to a sibling operator.** ROI-Align, `grid_sample` itself, or point-cloud feature gathering all share MSDA's anatomy: data-dependent bilinear gathers forward, atomic scatters backward. Write the forward in Triton using the chapter-4 playbook (block over outputs, wide loads over channels, parameters coalesced, fp32 registers). You'll find the design transfers almost mechanically — that's the point of learning it on one operator properly.

**12. Wire it into a real model.** Swap this op into a Deformable-DETR or DINO implementation (the signature matches mmcv's `MultiScaleDeformableAttnFunction`) and confirm training-loss parity against the original op, then measure the wall-clock difference per step. Bonus: capture inference with CUDA graphs and measure the dispatch overhead disappearing.

## Further reading

**Triton**

- [Official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — vector-add through fused attention; tutorial 03 (matmul) is the canonical *compute*-bound counterpart to this course's memory-bound story.
- The [`triton.language` reference](https://triton-lang.org/main/python-api/triton.language.html) — worth skimming end-to-end once, so you know what primitives exist.

**GPU architecture & performance**

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) chapters on the hardware model and memory hierarchy — the primary source under chapter 2.
- ["How to Optimize a CUDA Matmul Kernel" (Simon Boehm)](https://siboehm.com/articles/22/CUDA-MMM) — the best single walkthrough of the compute-bound playbook.
- NVIDIA's [Nsight Compute](https://developer.nvidia.com/nsight-compute) docs — when you outgrow `torch.profiler`, this is where kernel-level truth (achieved bandwidth, sector efficiency, stall reasons) lives.

**The operators and papers**

- Zhu et al., [*Deformable DETR*](https://arxiv.org/abs/2010.04159) (ICLR 2021) — the operator's origin; §4.1 defines MSDA.
- *Efficient Multi-Scale Deformable Attention on GPUs* (TMLR submission) — the diagnostic study this repo reproduces.
- [PyTorch's notes on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) — the official framing of the atomics-nondeterminism trade-off from chapter 5.

---

You now know everything this repo knows. The two kernels total under 300 lines — go read them start to finish one more time, and notice how little of it is mysterious now.

[Back to the course index →](../index.md)
