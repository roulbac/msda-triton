# 5. The backward kernel: gradients, atomics, and a precision switch

!!! abstract "What you'll learn"
    How to derive MSDA's three gradients by hand, why the backward pass forces *scattered atomic writes*, what "relaxed ordering" means and why it's safe here, and the hardware detective story behind the FP32-accumulator switch — why BF16 training was mysteriously ~10× slower on A100 than H100, and how four lines in `ops.py` fix it.

## What the backward pass must produce

The forward computed, per (query, head) and with $w_{00} \dots w_{11}$ the bilinear corner weights:

$$
\mathrm{out} = \sum_{l,k} A_{lk} \cdot \underbrace{\Big( w_{00} v_{00} + w_{01} v_{01} + w_{10} v_{10} + w_{11} v_{11} \Big)}_{\text{sampled}(l,k)}
$$

Autograd hands us $g = \partial \mathcal{L} / \partial\, \mathrm{out}$ (a $D$-vector per query/head) and asks for gradients w.r.t. all three differentiable inputs. All three come from routine applications of the chain rule — the interesting part is what they *do to memory*.

**1. Attention weights** — the output is linear in $A_{lk}$, so:

$$
\frac{\partial \mathcal{L}}{\partial A_{lk}} = g \cdot \mathrm{sampled}(l,k)
$$

a dot product over channels. *Memory pattern: one scalar written per (q, m, l, k) — clean, disjoint, no problem.*

**2. Sampling locations** — the bilinear weights depend on the fractional offsets $l_x, l_y$; differentiating the interpolation w.r.t. $x$ gives

$$
\frac{\partial\, \mathrm{sampled}}{\partial x_{\mathrm{im}}} = (v_{01} - v_{00})(1 - l_y) + (v_{11} - v_{10})\, l_y
$$

(the horizontal difference of the corner values, blended vertically — bilinear interpolation is piecewise-linear in each coordinate, so its derivative is the slope between corners). The kernel's `dx`/`dy` lines are exactly this. Then the chain rule through the pixel mapping $x_{\mathrm{im}} = x \cdot W_l - 0.5$ contributes a factor $W_l$:

$$
\frac{\partial \mathcal{L}}{\partial x} = A_{lk} \cdot W_l \cdot \big( g \cdot \tfrac{\partial\, \mathrm{sampled}}{\partial x_{\mathrm{im}}} \big)
$$

*Memory pattern: two scalars per sample point, disjoint writes — no problem.*

**3. Values** — here's the trouble. Corner $v_{00}$ contributed with coefficient $A_{lk} w_{00}$, so it receives gradient $A_{lk} w_{00} \, g$. But a given *pixel* of `value` may have been sampled by **many queries** — every query whose learned location landed nearby. Its total gradient is a *sum over an unpredictable, data-dependent set of contributors*:

$$
\frac{\partial \mathcal{L}}{\partial v_p} \;=\; \sum_{\substack{\text{all } (q,m,l,k) \text{ that} \\ \text{touched pixel } p}} A \, w \, g
$$

The forward *gathered* (scattered reads → private sum). The backward must ***scatter*** (private values → sums at scattered addresses). This asymmetry — gather forward, scatter backward — is generic to every sampling-style operator (grid_sample, ROI-align, point clouds), and it's why the backward is several times slower than the forward across *all* implementations, mmcv included.

## Scatter needs atomics

Different programs — different queries running concurrently on different SMs — must add into the *same* `grad_value` locations whenever their samples land near the same pixels. As chapter 2 explained, concurrent `read-add-write` loses updates. Options:

- **Privatize + reduce**: give each program its own `grad_value` copy, sum afterwards. A full copy per program is absurd here (`grad_value` is tens of MB, thousands of programs).
- **Reorder by destination**: sort contributions by pixel so each pixel is owned by one thread. Requires building the inverse mapping every step — the sampling locations change every iteration. Too expensive.
- **Atomic adds**: let the hardware serialize collisions. Collisions are *sparse* here (128 sample points per query scattered over ~100k pixels), so contention is low and this wins.

The kernel recomputes the bilinear setup (locations, corner indices, masks — cheaper to recompute from saved inputs than to store from the forward) and then, per corner:

```python
wg = g * attn[:, None]                       # (BLOCK_Q, BLOCK_D), fp32
tl.atomic_add(gval_base + off00, (wg * w00[:, None]).to(gdt),
              mask=m00, sem="relaxed")
# ... and off01, off10, off11
```

Four block-valued atomic scatters per sample point. Note the same masks as the forward's loads: an out-of-bounds corner contributed zero to the output, so it *receives* zero gradient — the masked-off atomic simply doesn't happen.

### `sem="relaxed"`: dropping ordering we don't need

By default, GPU atomics come with **memory-ordering guarantees** (`acq_rel`): they act as partial fences constraining how surrounding memory operations may be reordered, so you can build locks and queues out of them. Fences restrict the memory subsystem's freedom to combine and reorder in-flight traffic — costly for a kernel emitting *millions* of atomics.

Gradient accumulation needs none of that. Additions are **commutative and associative** — any order gives the same sum (up to floating-point rounding, more on which below) — and no thread ever *reads* `grad_value` during the kernel; consumers read it only after the kernel completes, and **kernel completion is itself a full barrier** (the happens-before edge). So the atomics are demoted to `sem="relaxed"`: indivisible, but with no ordering constraints. The hardware can then buffer and combine the atomic stream aggressively — and on SM 9.0+ these lower to the native reduction instruction `RED` (fire-and-forget add, no value returned) rather than an ordered read-modify-write.

The paper (§3.3, Fig. 8) measures this and the launch shape; this repo additionally fixes the backward's launch config at `BLOCK_Q=16, num_warps=4` — chosen by an offline sweep because, as chapter 3 noted, an atomic-accumulating kernel can't be autotuned naively (each timing run would double the output). That shape gives each thread a 4-element slice of the (16, 32) tile, spreading the atomic stream across the most warps — up to 2.1× over the runner-up config at decoder scale.

## The BF16 mystery: same code, 10× slower on A100

Now the best part. Run this backward in BF16 on an H100: fast. Run the *identical* code on an A100: the backward craters — roughly **10× lower atomic throughput** (paper Table 2). Nothing in the source differs. Why?

Because **atomic-add support depends on the dtype–hardware pair** (chapter 2, cost #2). What each combination compiles to:

| dtype of `grad_value` | A100 (SM 8.0) | L40S (SM 8.9) | H100 (SM 9.0) |
|---|---|---|---|
| FP32 | ✅ native `red.add.f32` | ✅ native | ✅ native |
| FP16 | ✅ native `red.add.noftz.f16x2` (since SM 6.0!) | ✅ native | ✅ native |
| **BF16** | ❌ **CAS emulation loop** | ✅ native | ✅ native `red.add.noftz.bf16` |

BF16 arrived in the CUDA ecosystem *after* FP16, and its atomic add only became a hardware instruction with SM 9.0 (PTX ISA 7.8). On older GPUs the compiler silently emulates it with a **compare-and-swap (CAS) retry loop**: read the word, compute the new value, attempt an atomic "replace if unchanged", retry on failure. That's a full round-trip read-modify-write with a loop around it, versus a fire-and-forget hardware reduction — and under concurrent traffic the retries multiply. Same source, catastrophically different machine code. You can *see* this in this repo without a GPU: CI compiles every kernel variant to PTX for SM 8.0 and 9.0 and prints which atomic instruction each backward emits (`scripts/compile_kernels_check.py` — chapter 7).

### The fix: accumulate in FP32, downcast at the end

If BF16 atomics are emulated on pre-Hopper GPUs, don't do BF16 atomics there. [`ops.py`](https://github.com/roulbac/msda-triton/blob/main/src/msda_triton/ops.py) allocates the accumulation buffer in fp32 *for that case only*, and converts once at the end:

```python
def _use_fp32_grad_accum(value):
    if value.dtype != torch.bfloat16:
        return False
    return torch.cuda.get_device_capability(value.device)[0] < 9

# in backward():
acc_dtype = torch.float32 if ctx.fp32_grad_accum else value.dtype
grad_value = torch.zeros_like(value, dtype=acc_dtype)   # kernel atomically adds into this
...
if grad_value.dtype != value.dtype:
    grad_value = grad_value.to(value.dtype)             # one cheap elementwise downcast
```

The kernel is dtype-agnostic — it casts contributions to `grad_value_ptr.dtype.element_ty` before the atomic — so the *caller chooses the atomic instruction* by choosing the buffer's dtype. Cost: a 2× larger scratch buffer and a downcast kernel. Benefit: native fp32 atomics instead of a CAS loop. As a bonus, fp32 accumulation is also *more accurate* than accumulating in bf16.

**Why FP16 doesn't take this path**: `red.add.noftz.f16x2` has been hardware-native since SM 6.0 — fp16 atomics were never broken. Forcing fp16 through the fp32 scratch buffer measures 2–4× *slower* than native fp16 atomics on Ampere/Ada (double the atomic bytes plus the extra pass). The switch is deliberately narrow: **bf16 AND pre-Hopper only**, auto-detected, overridable via `fp32_grad_accum=True/False`.

!!! tip "The transferable lesson"
    "The same `tl.atomic_add` line" is not the same instruction on every GPU × dtype. When a kernel is inexplicably slow on one hardware generation, look at what the code *lowers to* — PTX/SASS — not just what it says. And when a hardware fast path is missing, restructuring around it (here: changing a buffer's dtype) can be a 10× lever that costs four lines of Python.

## Atomics make gradients nondeterministic

One honest cost of atomic accumulation: **run-to-run bit-exactness is gone**. Floating-point addition is not associative ($(a+b)+c \neq a+(b+c)$ in the last bits), and atomics commit in whatever order threads happen to arrive, which varies between runs. So `grad_value` can differ in trailing bits across identical runs.

Nuances worth knowing:

- With the **fp32 accumulator**, all contributions enter at fp32; only *ordering* varies, so runs agree to fp32 rounding (~1e-7 relative). Tight.
- With **native half-precision atomics**, each add also *rounds to bf16/fp16 at every step*, so ordering differences are amplified — looser, though harmless for training.
- This is normal territory: PyTorch documents the same behavior for many standard CUDA ops (`index_add`, `scatter_add`, embedding backward…). It only matters if you need `torch.use_deterministic_algorithms(True)`.
- It's also why the test suite compares gradients with *tolerances* rather than exact equality — chapter 7.

## The autograd wrapper

Gluing it into PyTorch is standard machinery — `torch.autograd.Function` with the kernel launches inside:

```python
class _MSDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, ..., fp32_grad_accum):
        ...
        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                              sampling_locations, attention_weights)   # for recompute
        return out.view(B, Q, M * D)

    @staticmethod
    @once_differentiable          # honest: this backward is not itself differentiable
    def backward(ctx, grad_out):
        ...
        return grad_value, None, None, grad_loc, grad_attn, None   # None: non-tensor args
```

`@once_differentiable` declares that the hand-written backward can't be differentiated again (no analytic double-backward here) — attempting a second derivative raises an error instead of silently returning garbage.

!!! summary "Key takeaways"
    - Three gradients: $g\cdot\mathrm{sampled}$ (weights), corner differences × $A \cdot W_l$ (locations) — both disjoint writes — and the scattered sum into `grad_value`, the hard one.
    - Forward gathers, backward scatters; sparse data-dependent collisions make atomic adds the right tool.
    - `sem="relaxed"` drops ordering guarantees that commutative accumulation never needed; the kernel-end barrier provides the only synchronization required.
    - BF16 atomic add is CAS-emulated before SM 9.0 (~10× slower); the fix is an fp32 scratch accumulator — for bf16 on pre-Hopper *only*, since fp16 atomics have been native since SM 6.0.
    - Atomic accumulation ⇒ bitwise nondeterminism; fp32 accumulation bounds it to rounding-level noise.

[Next: Benchmarking without fooling yourself →](06-benchmarking.md)
