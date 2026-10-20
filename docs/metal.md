# Running on Apple GPUs (Metal)

`msda-triton` is one package with two backends: Triton kernels on NVIDIA GPUs and a Metal port on Apple silicon. The backend is picked from the tensor's device, and the Python API is identical.

```python
out = multi_scale_deformable_attention(
    value.to("mps"), spatial_shapes.to("mps"), None,
    sampling_locations.to("mps"), attention_weights.to("mps"),
)
```

## Why a separate implementation

Triton has no Apple GPU backend (and no macOS wheels), so the kernels cannot simply be recompiled. The operator is small, though: masked gathers, bilinear arithmetic and atomics — no `tl.dot`, no shared memory — so [`metal.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/metal.py) re-implements both kernels in Metal Shading Language and compiles them at run time with `torch.mps.compile_shader`. The math, memory layout and bilinear convention are identical to the Triton kernels.

| Triton kernel | Metal port |
|---|---|
| One program per block of queries for a (batch, head) | One simdgroup (or a slice of one when D < 32) per query |
| Wide `tl.load` over the channel dim | Lanes stride over channels, so each corner gather is one coalesced row |
| `tl.sum` over channels | Butterfly `simd_shuffle_xor` across the lanes sharing a query |
| `tl.atomic_add(..., sem="relaxed")` into native or FP32 buffer | Relaxed `atomic_float` adds into an FP32 buffer, always |
| `@triton.autotune` over block size and warps | Fixed launch; shape constants (M, D, L, K, dtype) compiled in |
| 64-bit flat indices | 64-bit flat indices; grid chunked to stay within a 32-bit dispatch |

The FP32 accumulator is not a choice here: Metal has no half-precision atomic add, so the "accumulator precision switch" from the CUDA path is permanently on, and `fp32_grad_accum` is ignored on MPS.

## Testing on GitHub's macOS runners

The same correctness suite that validates the CUDA kernels runs on Apple silicon in CI ([`metal.yml`](https://github.com/roulbac/msda-triton/blob/main/.github/workflows/metal.yml)): forward and all three gradients against the FP32 CPU reference, across fp32/fp16/bf16, out-of-bounds locations and non-power-of-two head dims, plus a test of the chunked launch path. 32 tests pass on `macos-15` and `macos-26`.

We probed which hosted runners expose a GPU before building this:

| Runner | Result |
|---|---|
| `macos-15`, `macos-26` | Real Metal compute works, including relaxed float atomics |
| `macos-14` | MPS reports available, but the first allocation fails out-of-memory — unusable, so excluded |

All of them are virtualized Apple M1s (`Apple M1 (Virtual)`), which matters for the benchmark below.

## Benchmark

!!! info "Which chip?"
    **Apple M1 (virtualized), GitHub-hosted `macos-15` runner**, macOS 15.7.9, PyTorch 2.13.0, 3 CPU cores. A single CI run on a shared VM.

[`benchmarks/mps_benchmark.py`](https://github.com/roulbac/msda-triton/blob/main/benchmarks/mps_benchmark.py) times the Metal kernel against the `grid_sample` PyTorch reference on the same GPU, at the paper's operating points (800×1333, D=256 = 8 heads × 32, L=4, K=4): `decoder` is B=4, Q=300; `encoder` is B=2, Q=S=22,223. Wall-clock milliseconds per call, 50 iterations after warmup:

| preset | dtype | ref fwd | Metal fwd | fwd speedup | ref bwd | Metal bwd | bwd speedup |
|---|---|--:|--:|--:|--:|--:|--:|
| decoder | fp32 | 21.4 ms | 1.44 ms | **14.9×** | 28.8 ms | 4.52 ms | **6.4×** |
| decoder | fp16 | 19.0 ms | 1.47 ms | **13.0×** | 25.1 ms | 6.42 ms | **3.9×** |
| encoder | fp32 | 294 ms | 52.3 ms | **5.6×** | 633 ms | 100 ms | **6.3×** |
| encoder | fp16 | 245 ms | 52.4 ms | **4.7×** | 610 ms | 93.6 ms | **6.5×** |

Backward is (forward + backward) − forward.

### Reading these numbers honestly

This is the same lesson as [the benchmarking chapter](course/06-benchmarking.md), applied to a weaker measurement setup:

- **Not comparable to the CUDA tables.** Those are device kernel time (profiler self-time) on datacenter GPUs, against mmcv. These are wall-clock per call on a shared virtual M1, including host dispatch, because MPS exposes no per-kernel device timer here.
- **The baseline is the eager PyTorch fallback**, not a tuned kernel. The result says the Metal kernel is far faster than what a Mac user otherwise gets; it does not say how close it is to the hardware's limit. The Metal kernels are untuned (no threadgroup-size sweep).
- **Indicative, not definitive.** One run on shared hardware. An earlier 800×1216 run on `macos-26` agreed within ~5%. Run `benchmarks/mps_benchmark.py` on your own Mac for numbers you can rely on.
- **fp16 backward is slower than fp32** because it accumulates in the same FP32 scratch buffer and casts once at the end.

Each CI run publishes its table in the job summary and the raw JSON as a build artifact.
