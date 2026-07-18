# The labs: rebuild this repo's kernel from scratch

A hands-on companion to msda-triton in the spirit of [*From NAND to
Tetris*](https://www.nand2tetris.org/): eleven [marimo](https://marimo.io)
notebooks that take you from "what is deformable attention?" to **an operator
you built yourself passing this repo's own test suite** — the operator itself
as PyTorch layers (lab 0) → bilinear interpolation → flat memory → a paper-GPU
simulator → Triton → the forward kernel → tiling experiments → hand-derived
gradients → atomics → PTX forensics → a shipped library.

Audience: a junior CS student / engineer comfortable with Python and basic
PyTorch. No GPU experience, no CUDA, no Triton, no MSDA background assumed.
Prefer reading to building? The same material is covered expository-style in
the [course](https://roulbac.github.io/msda-triton/); each lab links its
matching chapter.

## Setup

```bash
git clone https://github.com/roulbac/msda-triton && cd msda-triton
uv sync --group labs               # library + test deps + marimo/numpy
uv run marimo edit labs/00_the_idea.py
```

**No CUDA GPU? You can still do almost everything.** The labs auto-enable
Triton's CPU interpreter (`TRITON_INTERPRET=1`), which runs kernels correctly
(slowly) on any machine. On Linux, `uv sync` already pulls in Triton via
torch, so no extra install step is needed. Cells whose *point* is performance
are marked **GPU required** — for those, any CUDA machine works, or use the
repo's Modal harness to watch the shipped kernels run on serious hardware:
`MSDA_GPU=L4 uv run modal run benchmarks/modal_benchmark.py`.

## The ladder

| # | lab | you build | needs |
|---|---|---|---|
| 0 | [`00_the_idea.py`](00_the_idea.py) | deformable attention → cross-attention → **MSDA as PyTorch layers** — what the operator *is* and why | CPU |
| 1 | [`01_the_spec.py`](01_the_spec.py) | bilinear interpolation + naive MSDA, in loops | CPU |
| 2 | [`02_tensors_are_pointers.py`](02_tensors_are_pointers.py) | MSDA on flat 1-D memory with hand-computed offsets; the int32-overflow trap | CPU |
| 3 | [`03_the_machine.py`](03_the_machine.py) | a sector-counting GPU memory simulator; *predicts* the kernel's tiling | CPU |
| 4 | [`04_hello_triton.py`](04_hello_triton.py) | first kernels: add → 2-D copy → gather → bilinear | CPU (interpreter) |
| 5 | [`05_forward_correct.py`](05_forward_correct.py) | **the full MSDA forward kernel** | CPU (interpreter) |
| 6 | [`06_make_it_fast.py`](06_make_it_fast.py) | the tiling race on silicon + autotuning | **GPU** |
| 7 | [`07_the_other_direction.py`](07_the_other_direction.py) | all three gradients, derived by hand, checked against autograd | CPU |
| 8 | [`08_concurrent_writes.py`](08_concurrent_writes.py) | a data race you can watch, atomics, **the full backward kernel** | CPU (interpreter)* |
| 9 | [`09_hardware_detective.py`](09_hardware_detective.py) | PTX forensics: the bf16 CAS cliff + the fp32-accumulator switch | CPU (AOT compile) |
| 10 | [`10_ship_it.py`](10_ship_it.py) | autograd wrapper, validation, and **the repo's test suite run against your op** | CPU subset / GPU full |

\* the race *demo* needs real concurrency (GPU); the kernel work doesn't.

## How the labs work

- **Reactive checking**: each exercise cell is followed by a checker cell
  comparing you against the repo's reference implementations
  (`tests/reference_impls.py`) with the test suite's own tolerances. Edit your
  code and the checks re-run instantly — ✅ / ❌ / 🚧 (not attempted).
- **Progressive hints**: every exercise has collapsible hints, escalating from
  a nudge to the exact lines.
- **Builtin chips** (`labs/solutions/`): complete reference solutions. When a
  later lab needs an earlier artifact, it loads *yours* from `labs/my/labNN.py`
  if you saved it there (each lab's epilogue reminds you), otherwise the
  reference — so being stuck on stage N never blocks stage N+1.
- **Everything is tested in CI** (`tests/test_labs.py`): solutions against
  references, kernels through the interpreter and the AOT compiler, and every
  notebook headlessly.

## Map to the rest of the repo

| after lab | read |
|---|---|
| 0–1 | [course ch. 1](https://roulbac.github.io/msda-triton/course/01-deformable-attention/) — the operator in full |
| 3 | [course ch. 2](https://roulbac.github.io/msda-triton/course/02-gpu-programming-model/) — the real hardware behind the paper GPU |
| 4 | [course ch. 3](https://roulbac.github.io/msda-triton/course/03-triton-basics/) |
| 5–6 | [course ch. 4](https://roulbac.github.io/msda-triton/course/04-forward-kernel/) + `src/msda_triton/kernels.py` (the shipped forward) |
| 7–9 | [course ch. 5](https://roulbac.github.io/msda-triton/course/05-backward-kernel/) + `src/msda_triton/ops.py` |
| 10 | [course ch. 6](https://roulbac.github.io/msda-triton/course/06-benchmarking/) & [7](https://roulbac.github.io/msda-triton/course/07-testing/), then [ch. 8's exercises](https://roulbac.github.io/msda-triton/course/08-exercises/) — including real unclaimed headroom in this repo |
