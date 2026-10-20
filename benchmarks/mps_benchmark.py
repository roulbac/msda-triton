"""Benchmark the Metal MSDA kernels on an Apple GPU (local Mac or a GitHub
macOS runner) against the grid_sample-based PyTorch reference.

    uv run python benchmarks/mps_benchmark.py                  # decoder + encoder
    uv run python benchmarks/mps_benchmark.py --preset decoder --dtypes fp32 fp16
    uv run python benchmarks/mps_benchmark.py --json out.json --markdown out.md

Presets match benchmarks/modal_benchmark.py (Deformable-DETR operating points,
D = 256 = 8 heads x 32): ``decoder`` is B=4, Q=300; ``encoder`` is B=2, Q=S.

Timing is wall-clock per call over a saturated queue with a single
``torch.mps.synchronize()`` at the end, so it includes host dispatch cost (MPS
exposes no per-kernel device timer here). On a shared CI VM treat absolute
numbers as indicative: they are good for regression tracking, not headlines.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from reference_impls import msda_reference  # noqa: E402

from msda_triton import multi_scale_deformable_attention  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def pyramid(height, width, levels=4, stride0=8):
    return [
        (-(-height // (stride0 << i)), -(-width // (stride0 << i))) for i in range(levels)
    ]


def make_inputs(preset, resolution, dtype, M=8, D=32, K=4, seed=0):
    h, w = resolution
    shapes = pyramid(h, w)
    S = sum(a * b for a, b in shapes)
    B, Q = (4, 300) if preset == "decoder" else (2, S)
    gen = torch.Generator().manual_seed(seed)
    L = len(shapes)
    value = torch.randn(B, S, M, D, generator=gen)
    loc = torch.rand(B, Q, M, L, K, 2, generator=gen)
    attn = torch.rand(B, Q, M, L, K, generator=gen)
    attn = attn.flatten(3).softmax(-1).view(B, Q, M, L, K)
    spatial_shapes = torch.tensor(shapes, dtype=torch.long)
    hw = spatial_shapes.prod(-1)
    starts = torch.cat([hw.new_zeros(1), hw.cumsum(0)[:-1]])
    to = lambda t: t.to("mps", dtype)
    return (to(value), spatial_shapes.to("mps"), starts.to("mps"), to(loc), to(attn)), (B, Q, S)


def time_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_one(impl, args, grad_out, iters, warmup):
    """(fwd_ms, bwd_ms); None where an implementation is unsupported."""
    value, shapes, starts, loc, attn = args

    def call(v, l, a):
        if impl == "metal":
            return multi_scale_deformable_attention(v, shapes, starts, l, a)
        return msda_reference(v, shapes, l, a)

    try:
        with torch.no_grad():
            fwd = time_ms(lambda: call(value, loc, attn), iters, warmup)
    except Exception as e:  # noqa: BLE001 - report unsupported ops, keep going
        print(f"  {impl} fwd failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None, None

    v = value.clone().requires_grad_(True)
    l = loc.clone().requires_grad_(True)
    a = attn.clone().requires_grad_(True)

    def fwd_bwd():
        out = call(v, l, a)
        torch.autograd.grad(out, (v, l, a), grad_out)

    try:
        both = time_ms(fwd_bwd, max(iters // 2, 5), max(warmup // 2, 2))
    except Exception as e:  # noqa: BLE001
        print(f"  {impl} bwd failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return fwd, None
    return fwd, max(both - fwd, 0.0)


def chip():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or "unknown"


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["decoder", "encoder", "both"], default="both")
    ap.add_argument("--resolution", default="800x1216", help="HxW input image size")
    ap.add_argument("--dtypes", nargs="+", choices=list(DTYPES), default=["fp32", "fp16"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    opts = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("MPS is not available on this machine")
    res = tuple(int(x) for x in opts.resolution.lower().split("x"))
    presets = ["decoder", "encoder"] if opts.preset == "both" else [opts.preset]

    rows = []
    for preset in presets:
        for dname in opts.dtypes:
            args, (B, Q, S) = make_inputs(preset, res, DTYPES[dname])
            grad_out = torch.randn(B, Q, args[0].shape[2] * args[0].shape[3]).to(
                "mps", DTYPES[dname]
            )
            grad_out = grad_out.reshape(B, Q, -1)
            row = dict(preset=preset, dtype=dname, B=B, Q=Q, S=S)
            for impl in ("reference", "metal"):
                print(f"{preset} {dname} {impl} ...", file=sys.stderr)
                f, b = bench_one(impl, args, grad_out, opts.iters, opts.warmup)
                row[impl] = dict(fwd_ms=f, bwd_ms=b)
            rows.append(row)

    meta = dict(
        chip=chip(), macos=platform.mac_ver()[0], torch=torch.__version__,
        resolution=opts.resolution, iters=opts.iters,
    )
    lines = [
        f"**{meta['chip']}**, macOS {meta['macos']}, torch {meta['torch']}, "
        f"{opts.resolution}, {opts.iters} iters (wall-clock per call, ms)",
        "",
        "| preset | dtype | B | Q | ref fwd | metal fwd | fwd x | ref bwd | metal bwd | bwd x |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        rf, mf = r["reference"]["fwd_ms"], r["metal"]["fwd_ms"]
        rb, mb = r["reference"]["bwd_ms"], r["metal"]["bwd_ms"]
        sp = lambda a, b: "n/a" if a is None or not b else f"{a / b:.1f}x"
        lines.append(
            f"| {r['preset']} | {r['dtype']} | {r['B']} | {r['Q']} | {fmt(rf)} | {fmt(mf)} "
            f"| {sp(rf, mf)} | {fmt(rb)} | {fmt(mb)} | {sp(rb, mb)} |"
        )
    table = "\n".join(lines)
    print(table)
    if opts.markdown:
        Path(opts.markdown).write_text(table + "\n")
    if opts.json:
        Path(opts.json).write_text(json.dumps(dict(meta=meta, rows=rows), indent=2))


if __name__ == "__main__":
    main()
