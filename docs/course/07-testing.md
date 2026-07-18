# 7. Testing numerical kernels

!!! abstract "What you'll learn"
    How to build trust in a hand-written kernel: chained reference implementations, choosing tolerances when bit-exactness is impossible, which edge cases actually bite for a sampling operator, and how this repo tests *GPU kernels in CI without any GPU*.

## The problem with testing numerics

A kernel bug rarely crashes. It returns *numbers* — plausible-looking, slightly-wrong numbers. A sign flip in one bilinear corner's gradient, an off-by-one in the boundary mask, a missing $W_l$ chain-rule factor: all produce tensors of the right shape that will happily train a model somewhat worse than it should. Worse, you can't test with `==`: the Triton kernel accumulates in a different order than the reference, reduced-precision dtypes round at different points, and the backward's atomics (chapter 5) aren't even deterministic against *themselves*. Every comparison is approximate — so the question "is it correct?" becomes "correct *relative to what, within what tolerance, and why that tolerance?*"

## A chain of trust

Comparing implementation A against reference B proves they agree — not that either is right, especially when both were written by the same person with the same misconceptions. This repo builds a **chain from eyeball-verifiable to fast** (see [`tests/reference_impls.py`](https://github.com/roulbac/msda-triton/blob/main/tests/reference_impls.py)):

```
msda_naive                msda_reference                Triton kernels
─────────────             ──────────────────            ─────────────────
pure Python loops    ⇐    F.grid_sample per level  ⇐    the real thing
+ hand-rolled bilinear    (mmcv's PyTorch fallback)     (fwd + 3 gradients)
CPU, tiny inputs          CPU or GPU, any dtype         CUDA only

"correct by             "correct because it            "correct because it
 inspection"             matches naive"                 matches reference"
```

- **Link 1** (`test_reference_matches_naive_cpu`): the vectorized `grid_sample` reference matches the 30-line naive loop on tiny CPU inputs. The naive loop's value is precisely that it's *dumb* — every convention (the `−0.5`, corner weights, zeros-padding) is visible in three lines you can check against the math in chapter 1. This link also pins down the coordinate convention: if this repo's reading of `align_corners=False` were wrong, the two independent expressions of it would disagree.
- **Link 2** (everything else): the Triton forward *and all three analytic gradients* match the reference computed **in FP32** — the reference runs at high precision even when the kernel under test runs at bf16, so the baseline contributes ~no error of its own and the tolerance budget belongs to the kernel alone.

The gradient tests need no hand-derived expected values: the reference is pure PyTorch, so **autograd differentiates it automatically**, and the kernel's hand-derived chapter-5 formulas are checked against autograd's answer. Wrong sign, missing factor, swapped corner — all caught by construction.

## Tolerances with a reason

Each comparison's tolerance follows from *which effects are legitimately allowed to differ*, dtype rounding first among them (fp32 ~1e-7, fp16 ~1e-3, bf16 ~1e-2 relative). Two effects deserve explicit tests of their own:

**Atomics-induced nondeterminism.** `grad_value` differs bit-wise run to run (chapter 5), so backward tolerances include reordering noise — loosest for native half-precision atomics, where every add also rounds. The suite also runs every backward case under **both accumulator paths** (`native_acc` / `fp32_acc` parametrization): the precision switch is a *feature with hardware-dependent behavior*, so both branches are exercised regardless of which GPU CI happens to get.

**Accumulator-path equivalence** (`test_fp32_accum_paths_agree_at_fp32`): at fp32 the two paths perform mathematically identical operations — buffer dtype and accumulation dtype coincide — so they must agree to tight tolerance. A cheap invariant that would catch a broken scratch-buffer path immediately.

## Edge cases: where sampling kernels actually break

The parametrized cases target the specific failure modes of this kernel's design decisions — each edge case is a test of a particular line in `kernels.py`:

| Edge case | What it would catch |
|---|---|
| Out-of-bounds locations (`loc_lo=-0.5, loc_hi=1.5`), forward **and** backward | Boundary-mask logic (`vx0/vy1…`): wrong clamping vs zeros, gradient leaking through masked corners |
| Locations exactly on grid corners / integer coordinates | `floor` and corner-weight behavior at knots — classic off-by-one territory where $l_x$ is exactly 0 |
| Non-power-of-two head dim $D$ | The `mask_d` path (`BLOCK_D = next_power_of_2(D)` over-allocates lanes) |
| Non-contiguous inputs | The wrapper's `.contiguous()` contract — the kernel itself assumes packed layout |
| `level_start_index=None` vs explicit | The wrapper's derived-offsets branch |
| Mismatched shapes/dtypes | Validation raises *before* a kernel can read garbage |
| Single-level, single-point, tiny configs | Degenerate loop bounds ($L{=}1$, $K{=}1$) |

The general principle: enumerate your kernel's *masks, boundaries, and layout assumptions*, and write one case per assumption. Random mid-range inputs exercise none of them.

## Testing GPU kernels without a GPU

GitHub-hosted CI runners have no GPU. Most kernel projects respond by not testing kernels in CI. This repo gets substantial coverage anyway — [`ci.yml`](https://github.com/roulbac/msda-triton/blob/main/.github/workflows/ci.yml) runs on every PR:

**The CPU-runnable slice of the suite.** Link 1 of the chain of trust (naive vs reference) is pure PyTorch on CPU — it runs everywhere. GPU-only tests skip themselves cleanly.

**Ahead-of-time kernel compilation** ([`scripts/compile_kernels_check.py`](https://github.com/roulbac/msda-triton/blob/main/scripts/compile_kernels_check.py)). Here's the trick: Triton is a *compiler*, and compilers don't need the target hardware to compile. The script invokes `triton.compile` with explicit targets — every kernel variant (fwd/bwd × fp32/fp16/bf16) for **SM 8.0 and SM 9.0** — on a CPU-only runner. This catches the entire class of bugs that a plain `import` cannot: a `@triton.jit` body is only *parsed and type-checked when compiled*, so a typo'd `tl.` API call, a shape mismatch in a block expression, or a broken constexpr path all surface here, before any GPU is involved.

It does one more thing, closing the loop with chapter 5: it greps the generated PTX and **prints which atomic instruction each backward variant lowers to** — CAS loop for native-bf16 on SM 8.0, `red.add` for the fp32-accumulator variant and on SM 9.0. The mechanism the precision switch depends on is *observable in CI output* on every PR. (Printed, not asserted — a future Triton is allowed to change its exact lowering without breaking the build; the design point is visibility, not a brittle string match.)

Full GPU runs remain a one-liner when wanted — the benchmark harness doubles as a remote test runner: `modal run benchmarks/modal_benchmark.py --run-tests`.

!!! summary "Key takeaways"
    - Test against a *chain* of references, rooted in an implementation simple enough to verify by eye — and run the reference in fp32 so the tolerance budget belongs to the kernel.
    - Let autograd differentiate the reference: hand-derived kernel gradients get checked against machine-derived truth for free.
    - Tolerances are per-effect (dtype rounding, atomic reordering), not one global fudge factor; test both branches of hardware-dependent switches, plus regimes where they must agree exactly.
    - Edge cases = your kernel's masks, boundaries, and layout assumptions, one test each.
    - No GPU ≠ no kernel CI: compile to PTX for explicit SM targets to catch kernel-body errors, and inspect the generated code for the properties your performance story depends on.

[Next: Exercises & further reading →](08-exercises.md)
