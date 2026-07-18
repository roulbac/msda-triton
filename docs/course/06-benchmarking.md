# 6. Benchmarking without fooling yourself

!!! abstract "What you'll learn"
    Why timing GPU code is genuinely hard, the two specific pitfalls that made this repo's *faster* kernel initially measure as *slower* than mmcv, how the benchmark harness measures instead, and how to read the results tables like someone who's been burned before.

## The cautionary tale

Early versions of this repo's benchmark ranked the Triton kernel **behind** mmcv at decoder scale. The kernel hadn't changed — the *measurement* was wrong, twice over. The corrected harness shows the same kernel up to 4.5× *faster*. If a benchmark can invert a ranking, it can invert yours: this chapter is the most practically transferable one in the course.

## Fact 1: the GPU runs behind your back

CUDA execution is **asynchronous**. `out = kernel(x)` doesn't run the kernel — it *enqueues* it and returns immediately, in microseconds, while the GPU works through the queue on its own clock. So the classic mistake:

```python
t0 = time.perf_counter()
out = my_kernel(x)
t1 = time.perf_counter()      # measures enqueueing, not execution!
```

Everyone learns the fix — put `torch.cuda.synchronize()` (wait until the queue drains) before reading the clock. But *where* you put it matters enormously.

## Pitfall 1: synchronizing inside the timed loop

The "obviously correct" protocol:

```python
for _ in range(iters):
    t0 = time.perf_counter()
    out = my_kernel(x)
    torch.cuda.synchronize()          # ← inside the loop
    times.append(time.perf_counter() - t0)
```

For a kernel that runs 10–30µs, this measures almost everything *except* the kernel:

- **Launch latency is now on the clock.** Each iteration pays the full host-side dispatch (Python wrapper, driver submission, hardware launch) *serially* before the kernel even starts, because the previous sync drained the queue. For a Triton op that's ~40µs of Python dispatch in front of a 10µs kernel.
- **The GPU idles between iterations** — and idle GPUs **downclock**. Drain the queue every iteration and the kernel executes at reduced clocks each time.

The paper measured this protocol inflating decoder-scale timings ~4× — and *inverting the ranking*, because the penalty is proportional to per-call overhead, not kernel time: Triton's ~40µs Python dispatch vs mmcv's ~20µs C++ dispatch. Sync-inside-the-loop is a benchmark of your *launcher*.

The standard correction — enqueue the whole loop, sync once, divide:

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    out = my_kernel(x)
torch.cuda.synchronize()              # ← once, after the loop
per_call = (time.perf_counter() - t0) / iters
```

Back-to-back kernels, warm clocks, dispatch overlapping execution. Better — this is the **e2e** protocol in this repo's benchmark. But for very small kernels there's a subtler trap left.

## Pitfall 2: when the queue is CPU-bound, *everything* measures the dispatcher

**CUDA events** are the usual next refinement — timestamps recorded *by the GPU itself* in stream order, bracketing just the kernel:

```python
start.record(); out = my_kernel(x); end.record()
```

Accurate for big kernels. But think about the small-kernel regime: the GPU finishes each 10µs kernel *faster than the host can enqueue the next one* (~40µs). The queue runs **CPU-bound**: the GPU sits briefly idle before each launch, and successive kernels are spaced not by their execution time but by *dispatch* time. Even perfectly-placed event pairs then report

$$
t_{\text{measured}} \approx \max(t_{\text{dispatch}}, t_{\text{kernel}})
$$

because the inter-kernel gap the GPU observes *is* the dispatch interval. Once again the Python-launcher tax masquerades as kernel time, and any op with cheaper dispatch "wins" regardless of its kernel. (At encoder scale, kernels are hundreds of µs ≫ dispatch, the queue stays GPU-bound, and CUDA events are fine — small kernels are the treacherous regime.)

## What the harness does instead

[`benchmarks/modal_benchmark.py`](https://github.com/roulbac/msda-triton/blob/main/benchmarks/modal_benchmark.py) reports **two separate columns**, because "how fast is the kernel" and "what does a call cost end-to-end" are different questions with different answers:

**Device kernel time** — from `torch.profiler`, which reads the GPU's own per-kernel start/end timestamps and reports **CUDA self-time**: time the kernel actually occupied the device, gaps excluded. Immune to both pitfalls. Cross-checked against **CUDA-graph replay** (below), an independent protocol that eliminates dispatch physically rather than accounting for it — the two agree, which is the kind of consistency check a benchmark should ship with.

**e2e time** — wall-clock per call over a saturated queue (sync-once protocol). This *deliberately includes* host dispatch, because it's what an isolated call costs in practice.

The GPU numbers in the README are the device column. The dispatch story is reported honestly alongside — see below.

The rest is standard hygiene, all visible in the script: **warmup iterations** before timing (first calls include JIT compilation and autotuning — chapter 3); **medians (p50)**, not means, since timing noise is heavy right-tailed; **realistic shapes** (the paper's decoder/encoder operating points, not convenient toy sizes — remember the int64-overflow lesson: behavior changes with scale); and comparing against the FP32 reference for **max-abs-error in the same run**, so a "fast" kernel that's silently wrong can't win a table.

## Reading the results

From the README (decoder preset, device kernel time, speedup vs mmcv):

| GPU | fwd fp32 | fwd fp16 | bwd fp32 | bwd fp16 | bwd bf16 |
|---|---|---|---|---|---|
| L40S (SM 8.9) | 15µs, **1.40×** | 9µs, **3.17×** | 325µs, 0.81× | 73µs, **2.90×** | 254µs, *n/a* |
| A100 (SM 8.0) | 35µs, **1.38×** | 18µs, **3.04×** | 316µs, 0.81× | 132µs, **3.90×** | 411µs, *n/a* |
| H100 (SM 9.0) | 12µs, **2.17×** | 8µs, **4.30×** | 92µs, **1.33×** | 56µs, **4.50×** | 56µs, *n/a* |

Every number connects to a chapter of this course:

- **fp16 ≈ half of fp32 forward time** — the memory-bound signature (chapter 2): half the bytes, half the time. If halving the dtype *didn't* halve runtime, you'd suspect the kernel wasn't bandwidth-limited.
- **Backward ≫ forward everywhere** — the gather-vs-scatter asymmetry (chapter 5), shared by mmcv.
- **H100 bwd bf16 = bwd fp16 (56µs)** — native bf16 atomics: dtype no longer matters. On A100, bf16-backward (411µs, via the fp32-accumulator path) is slower than fp32-backward — the scratch buffer means fp32-sized atomic traffic *plus* a downcast. The switch makes bf16 *viable* on A100, not free.
- **The 0.81× cells** — honest losses: pre-Hopper fp32 backward, where mmcv's shared-memory pre-reduction (partial sums in on-chip memory before touching DRAM atomics — a privatize-then-reduce hybrid from chapter 5's option list) beats plain scattered atomics by ~20%. At half precision that advantage is dwarfed by mmcv's slower half-precision atomic path. Benchmarks that show no losses should worry you.
- ***n/a*** — mmcv ships no bf16 kernel at all; there is nothing to be a speedup *against*. Also a finding.

## The dispatch caveat, stated plainly

Device-side, this Triton forward beats mmcv at every decoder-scale point measured. Host-side, Triton's Python launcher costs ~40µs/call vs ~20µs for mmcv's C++ op — so an *isolated, queue-drained* call is dispatch-bound and mmcv's wall clock can look better. When does each number govern?

- **Training** — the queue is full of surrounding ops; dispatch overlaps GPU work; device time is what you pay. ✅ Triton wins.
- **Latency-critical inference** — capture the model into a **CUDA graph**: record the whole kernel sequence once, replay with a single driver call, Python overhead → zero. The op is graph-capturable precisely because its launch is static. ✅ Device time again.
- **Isolated eager calls with an empty queue** — the ~40µs is real. The e2e column tells you this without flattery.

!!! summary "Key takeaways"
    - `synchronize()` inside the timed loop measures launcher + idle-clock execution — ~4× inflation and ranking inversion here.
    - When kernel time < dispatch time, the queue is CPU-bound and even CUDA events report ≈ max(dispatch, kernel).
    - Profiler CUDA self-time answers "how fast is the kernel"; saturated-queue e2e answers "what does a call cost" — report both; cross-check protocols.
    - Warmup, medians, realistic shapes, in-run correctness — table stakes.
    - Read tables through your model of the operator: memory-bound scaling, scatter-vs-gather cost, and hardware dtype support should all be visible in the numbers. If they aren't, question the benchmark first.

[Next: Testing numerical kernels →](07-testing.md)
