"""Benchmark the Triton MSDA kernels on Modal GPUs.

Compares five implementations. The fwd/bwd ms columns are device kernel time
(torch.profiler CUDA self-time per iteration; see _gpu_kernel_ms for why
CUDA-event protocols cannot measure kernels this small behind a Python
launcher). A separate e2e column reports wall-clock per call over a saturated
queue, which surfaces host-side dispatch cost. Implementations:

  * ``reference``      — grid_sample-based PyTorch fallback (msda_reference)
  * ``compiled``       — the same PyTorch reference under ``torch.compile``
  * ``mmcv``           — the standard CUDA kernel (mmcv.ops, Zhu et al. 2021)
  * ``triton``         — this repo, native-dtype grad_value atomics
  * ``triton-fp32acc`` — this repo, FP32-accumulator backward (paper §3.3)

The GPU type is fixed at image/function definition time, so pick it with the
``MSDA_GPU`` environment variable (any type Modal supports: T4, L4, A10G,
A100, A100-80GB, H100, H200, B200, ...):

    MSDA_GPU=H100 modal run benchmarks/modal_benchmark.py
    MSDA_GPU=A100 modal run benchmarks/modal_benchmark.py --preset encoder
    modal run benchmarks/modal_benchmark.py --dtypes bf16 --resolution 1536x2048
    modal run benchmarks/modal_benchmark.py --run-tests   # pytest suite on GPU

Presets (Deformable-DETR operating points from the paper, D=256 = 8 heads x 32):
  * ``decoder``: B=4, Q=300 learned queries
  * ``encoder``: B=2, Q=S (every pyramid token is a query)
"""

import os
from pathlib import Path

import modal

GPU = os.environ.get("MSDA_GPU", "A100")
ROOT = Path(__file__).resolve().parent.parent

# openmmlab only ever published prebuilt mmcv wheels up to torch2.4 (its dist
# index has nothing newer), which would otherwise force this whole benchmark
# onto a two-year-old torch/triton. Instead we build mmcv from source against
# the project's own pinned torch (see the `mmcv` dependency group and
# `[tool.uv.sources]`/`[tool.uv.extra-build-variables]` in pyproject.toml),
# which needs nvcc — hence the CUDA-devel base image instead of debian_slim.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "libgl1", "libglib2.0-0",  # opencv (mmcv dep) needs libGL
        "git",  # mmcv source fetch
        "build-essential",  # g++: the base image's default `c++` resolves to
                             # clang, which fails torch's compiler-ABI check
    )
    .uv_sync(
        uv_project_dir=str(ROOT),
        groups=["bench", "mmcv", "test"],
        env={"CC": "gcc", "CXX": "g++"},
    )
    .add_local_dir(ROOT / "src" / "msda_triton", remote_path="/root/msda_triton")
    .add_local_dir(ROOT / "tests", remote_path="/root/tests")
)

app = modal.App("msda-triton-benchmark", image=image)

WARMUP = 10
ITERS = 100


def _pyramid(height: int, width: int, levels: int = 4, stride0: int = 8):
    """Feature-pyramid shapes at strides 8, 16, 32, ... (ceil division)."""
    shapes = []
    for i in range(levels):
        s = stride0 << i
        shapes.append(((height + s - 1) // s, (width + s - 1) // s))
    return shapes


def _make_inputs(preset, resolution, dtype, device="cuda", M=8, D=32, K=4, seed=0):
    import torch

    h, w = resolution
    shapes = _pyramid(h, w)
    S = sum(hh * ww for hh, ww in shapes)
    B, Q = (4, 300) if preset == "decoder" else (2, S)
    gen = torch.Generator(device=device).manual_seed(seed)
    L = len(shapes)
    value = torch.randn(B, S, M, D, generator=gen, device=device).to(dtype)
    loc = torch.rand(B, Q, M, L, K, 2, generator=gen, device=device).to(dtype)
    attn = torch.rand(B, Q, M, L, K, generator=gen, device=device)
    attn = attn.flatten(3).softmax(-1).view(B, Q, M, L, K).to(dtype)
    spatial_shapes = torch.tensor(shapes, dtype=torch.long, device=device)
    hw = spatial_shapes.prod(-1)
    starts = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    return value, spatial_shapes, starts, loc, attn


def _gpu_kernel_ms(run_once, iters):
    """Mean device execution time per iteration: the sum of CUDA kernel
    self-times from torch.profiler over `iters` calls. This is the only
    protocol that reports kernel speed independent of host-side launch cost.
    CUDA-event timing — even with all iterations enqueued back-to-back —
    reports max(dispatch, kernel) once the queue is CPU-bound, which it is
    for kernels of tens of microseconds behind a Python launcher (Triton's
    dispatch is ~40µs/call); synchronizing inside every timed iteration is
    worse still, adding idle-clock latency and inverting rankings between a
    C++ launcher (mmcv) and Triton. Cross-checked against CUDA-graph replay
    (zero host overhead): both agree to ~0.1µs at decoder scale."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run_once()
        torch.cuda.synchronize()
    total_us = sum(
        e.self_device_time_total
        for e in prof.key_averages()
        if str(e.device_type) == "DeviceType.CUDA"
    )
    return total_us / iters / 1e3


def _time_fwd(fn, warmup=WARMUP, iters=ITERS):
    """(gpu_ms, e2e_ms). gpu_ms: device kernel time (see _gpu_kernel_ms).
    e2e_ms: wall-clock per call over a saturated queue — the throughput a
    training loop sees, max(GPU time, host dispatch time)."""
    import time

    import torch

    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - t0) / iters * 1e3

        gpu_ms = _gpu_kernel_ms(fn, iters)
    return gpu_ms, e2e_ms


def _time_bwd(fn, leaves, grad_out, warmup=WARMUP, iters=ITERS):
    """(gpu_ms, e2e_ms) for the backward alone: one forward builds the graph,
    then ``backward(retain_graph=True)`` re-runs the backward kernels each
    iteration, so only backward kernels appear in the profiled window.
    Gradients are cleared between iterations on the host (no GPU work)."""
    import time

    import torch

    out = fn()

    def one_bwd():
        for t in leaves:
            t.grad = None
        out.backward(grad_out, retain_graph=True)

    for _ in range(warmup):
        one_bwd()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        one_bwd()
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) / iters * 1e3

    gpu_ms = _gpu_kernel_ms(one_bwd, iters)
    return gpu_ms, e2e_ms


def _peak_mem_mb(fn, leaves, grad_out):
    import torch

    for t in leaves:
        t.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    out.backward(grad_out)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def _implementations(value, spatial_shapes, starts, loc, attn):
    """Yields (name, forward_fn) closures over *leaf* copies of the inputs.
    Each forward_fn returns a (B, Q, M*D) tensor with a grad graph."""
    import sys

    import torch

    sys.path[:0] = ["/root", "/root/tests"]
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
    from reference_impls import msda_reference

    from msda_triton import multi_scale_deformable_attention

    B = value.shape[0]

    def leaves():
        return (
            value.clone().requires_grad_(True),
            loc.clone().requires_grad_(True),
            attn.clone().requires_grad_(True),
        )

    v, l, a = leaves()
    yield "reference", (v, l, a), lambda: msda_reference(v, spatial_shapes, l, a)

    # Same PyTorch reference, wrapped in torch.compile. A fresh compiled
    # callable per set of leaves keeps the autograd graph independent; warmup
    # iterations in the timing loops absorb the one-off compilation cost.
    # msda_reference does `int(h) for h, w in spatial_shapes` to build the
    # per-level split sizes -- a Tensor->Python-int conversion that Dynamo
    # otherwise graph-breaks on; capture_scalar_outputs lets it trace through
    # as a single graph instead of falling back to eager around the break.
    torch._dynamo.config.capture_scalar_outputs = True
    vc, lc, ac = leaves()
    compiled_reference = torch.compile(msda_reference)
    yield "compiled", (vc, lc, ac), lambda: compiled_reference(vc, spatial_shapes, lc, ac)

    v2, l2, a2 = leaves()
    yield "mmcv", (v2, l2, a2), lambda: MultiScaleDeformableAttnFunction.apply(
        v2, spatial_shapes, starts, l2, a2, B  # im2col_step = B
    )

    v3, l3, a3 = leaves()
    yield "triton", (v3, l3, a3), lambda: multi_scale_deformable_attention(
        v3, spatial_shapes, starts, l3, a3, fp32_grad_accum=False
    )

    v4, l4, a4 = leaves()
    yield "triton-fp32acc", (v4, l4, a4), lambda: multi_scale_deformable_attention(
        v4, spatial_shapes, starts, l4, a4, fp32_grad_accum=True
    )


@app.function(gpu=GPU, timeout=3600)
def bench(preset: str, resolution: tuple, dtypes: list) -> list:
    import sys

    import torch
    from rich.console import Console

    sys.path[:0] = ["/root", "/root/tests"]
    from reference_impls import msda_reference

    console = Console(width=120)
    device_name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    console.rule(
        f"[bold]{device_name}[/bold] (SM {cap[0]}.{cap[1]}) | preset={preset} "
        f"res={resolution[0]}x{resolution[1]} | p50 of {ITERS} iters"
    )

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    rows = []
    for dtype_name in dtypes:
        dtype = dtype_map[dtype_name]
        value, spatial_shapes, starts, loc, attn = _make_inputs(preset, resolution, dtype)
        B, Q = value.shape[0], loc.shape[1]
        console.print(
            f"\n[bold cyan]dtype={dtype_name}[/bold cyan]  B={B} Q={Q} S={value.shape[1]} "
            f"M={value.shape[2]} D={value.shape[3]} "
            f"L={loc.shape[3]} K={loc.shape[4]}"
        )

        # One-off correctness cross-check against the FP32 reference.
        ref_out = msda_reference(
            value.float(), spatial_shapes, loc.float(), attn.float()
        )

        for name, leaves, fwd in _implementations(value, spatial_shapes, starts, loc, attn):
            row = dict(preset=preset, dtype=dtype_name, impl=name,
                       fwd_ms=None, bwd_ms=None, fwd_e2e_ms=None, bwd_e2e_ms=None,
                       peak_mb=None, max_err=None)
            try:
                with torch.no_grad():
                    row["max_err"] = (fwd().float() - ref_out).abs().max().item()
                grad_out = torch.randn_like(ref_out).to(dtype)
                row["fwd_ms"], row["fwd_e2e_ms"] = _time_fwd(fwd)
                row["bwd_ms"], row["bwd_e2e_ms"] = _time_bwd(fwd, leaves, grad_out)
                row["peak_mb"] = _peak_mem_mb(fwd, leaves, grad_out)
            except torch.cuda.OutOfMemoryError:
                row["error"] = "OOM"
                torch.cuda.empty_cache()
            except Exception as err:  # e.g. mmcv has no bf16 kernel
                row["error"] = f"{type(err).__name__}: {err}"
            rows.append(row)

        _print_table(console, [r for r in rows if r["dtype"] == dtype_name])
        del value, loc, attn, ref_out
        torch.cuda.empty_cache()
    return rows


def _print_table(console, rows):
    from rich.table import Table

    mmcv_row = next((r for r in rows if r["impl"] == "mmcv" and r["fwd_ms"]), None)
    fastest_fwd = min((r["fwd_ms"] for r in rows if r["fwd_ms"] is not None), default=None)
    fastest_bwd = min((r["bwd_ms"] for r in rows if r["bwd_ms"] is not None), default=None)

    table = Table(show_edge=False, header_style="bold")
    table.add_column("impl", no_wrap=True)
    table.add_column("fwd ms", justify="right")
    table.add_column("bwd ms", justify="right")
    table.add_column("fwd e2e", justify="right")
    table.add_column("bwd e2e", justify="right")
    table.add_column("peak MB", justify="right")
    table.add_column("fwd x mmcv", justify="right")
    table.add_column("bwd x mmcv", justify="right")
    table.add_column("max|err|", justify="right")

    for r in rows:
        if r.get("error"):
            table.add_row(r["impl"], f"[red]{r['error']}[/red]", "", "", "", "", "", "", "")
            continue
        fwd_x = f"{mmcv_row['fwd_ms'] / r['fwd_ms']:.2f}x" if mmcv_row else "-"
        bwd_x = f"{mmcv_row['bwd_ms'] / r['bwd_ms']:.2f}x" if mmcv_row else "-"
        fwd_style = "bold green" if r["fwd_ms"] == fastest_fwd else ""
        bwd_style = "bold green" if r["bwd_ms"] == fastest_bwd else ""
        table.add_row(
            r["impl"],
            f"[{fwd_style}]{r['fwd_ms']:.3f}[/{fwd_style}]" if fwd_style else f"{r['fwd_ms']:.3f}",
            f"[{bwd_style}]{r['bwd_ms']:.3f}[/{bwd_style}]" if bwd_style else f"{r['bwd_ms']:.3f}",
            f"{r['fwd_e2e_ms']:.3f}",
            f"{r['bwd_e2e_ms']:.3f}",
            f"{r['peak_mb']:.1f}",
            fwd_x,
            bwd_x,
            f"{r['max_err']:.2e}",
        )
    console.print(table)
    console.print(
        "[dim]fwd/bwd ms: device kernel time (torch.profiler CUDA self-time, "
        "cross-checked against CUDA-graph replay). e2e: wall-clock per call over "
        "a saturated queue — includes each backend's host-side dispatch cost "
        "(Triton's Python launcher is heavier than mmcv's C++ op; at decoder "
        "scale that overhead can exceed the kernel itself). Speedup columns "
        "compare device kernel time.[/dim]"
    )


@app.function(gpu=GPU, timeout=1800)
def run_pytest() -> int:
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH="/root")
    return subprocess.call(
        [sys.executable, "-m", "pytest", "/root/tests", "-v"], env=env
    )


@app.local_entrypoint()
def main(
    preset: str = "decoder",
    resolution: str = "800x1333",
    dtypes: str = "fp32,fp16,bf16",
    run_tests: bool = False,
):
    if run_tests:
        code = run_pytest.remote()
        if code != 0:
            raise SystemExit(code)
        return
    if preset not in ("decoder", "encoder"):
        raise ValueError("--preset must be 'decoder' or 'encoder'")
    h, w = (int(x) for x in resolution.split("x"))
    bench.remote(preset, (h, w), [d.strip() for d in dtypes.split(",")])
