# The labs: build it yourself

The course you're reading explains this repo's kernels. The **labs** make you
rebuild them — from bilinear interpolation in plain Python loops all the way to
a complete Triton operator that passes the repo's own pytest suite, in the
spirit of [*From NAND to Tetris*](https://www.nand2tetris.org/): every layer
built by you, every stage checked against supplied tests, reference solutions
as "builtin chips" so you're never hard-stuck.

They live in the repo as eleven [marimo](https://marimo.io) notebooks —
interactive, reactive (edit your code and the checkers re-run instantly), and
runnable **without a GPU**: correctness work goes through Triton's CPU
interpreter, and only the explicitly-marked performance sections need real
hardware (any CUDA machine, or the repo's Modal harness).

```bash
git clone https://github.com/roulbac/msda-triton && cd msda-triton
uv sync --group labs
uv run marimo edit labs/00_the_idea.py
```

## The ladder

| # | lab | you build | course chapter |
|---|---|---|---|
| 0 | `00_the_idea.py` | the operator itself: deformable attention → cross-attention → MSDA, as pure-PyTorch layers | [ch. 1](course/01-deformable-attention.md) |
| 1 | `01_the_spec.py` | bilinear interpolation + naive MSDA in loops — the spec everything answers to | [ch. 1](course/01-deformable-attention.md) |
| 2 | `02_tensors_are_pointers.py` | MSDA on flat 1-D memory, hand-computed offsets, the int32-overflow trap | [ch. 4](course/04-forward-kernel.md) |
| 3 | `03_the_machine.py` | a sector-counting paper-GPU simulator that *predicts* the tiling decision | [ch. 2](course/02-gpu-programming-model.md) |
| 4 | `04_hello_triton.py` | vector-add → 2-D tile copy → gather → bilinear kernel | [ch. 3](course/03-triton-basics.md) |
| 5 | `05_forward_correct.py` | **the full MSDA forward kernel**, checked on the repo's edge cases | [ch. 4](course/04-forward-kernel.md) |
| 6 | `06_make_it_fast.py` | the wide-vs-narrow tiling race on silicon; autotuning | [ch. 4](course/04-forward-kernel.md), [ch. 6](course/06-benchmarking.md) |
| 7 | `07_the_other_direction.py` | all three gradients derived by hand, graded by autograd | [ch. 5](course/05-backward-kernel.md) |
| 8 | `08_concurrent_writes.py` | a visible data race, atomics, **the full backward kernel** | [ch. 5](course/05-backward-kernel.md) |
| 9 | `09_hardware_detective.py` | PTX forensics: the bf16 CAS cliff and the fp32-accumulator switch | [ch. 5](course/05-backward-kernel.md), [ch. 7](course/07-testing.md) |
| 10 | `10_ship_it.py` | autograd wrapper + validation, then **the repo's test suite runs against *your* operator** | [ch. 6](course/06-benchmarking.md), [ch. 7](course/07-testing.md) |

Full details — setup, the GPU/CPU matrix, how the stage system works — in
[`labs/README.md`](https://github.com/roulbac/msda-triton/blob/main/labs/README.md).

!!! tip "Course or labs first?"
    They're designed to interleave: do a lab, then read its chapter (the table
    above maps them). The labs teach through your fingers; the course fills in
    the surrounding theory and the measurement war stories.
