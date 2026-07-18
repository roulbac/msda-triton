"""A/B kernel variants for the MSDA forward: baseline (installed) vs
int32-index and wider-autotune variants. Times kernel-only (CUDA graph
replay) and benchmark-style (per-iter events incl. dispatch).

    MSDA_GPU=L40S uv run modal run benchmarks/modal_variants.py::variants
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_benchmark import _make_inputs, app, image  # noqa: E402

GPU = os.environ.get("MSDA_GPU", "L40S")

var_image = image.add_local_file(
    HERE / "modal_benchmark.py", remote_path="/root/modal_benchmark.py"
)

KERNEL_SRC = r'''
import triton
import triton.language as tl


def make_fwd_kernel(autotune_configs):
    @triton.autotune(configs=autotune_configs, key=["Q", "M", "D", "L", "K"])
    @triton.jit
    def _fwd(
        value_ptr, shapes_ptr, starts_ptr, loc_ptr, attn_ptr, out_ptr,
        Q, S,
        M: tl.constexpr, D: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
        INT64: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_bm = tl.program_id(1)
        if INT64:
            b = (pid_bm // M).to(tl.int64)
            offs_q = (pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)
        else:
            b = pid_bm // M
            offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        m = pid_bm % M

        offs_d = tl.arange(0, BLOCK_D)
        mask_q = offs_q < Q
        mask_d = offs_d < D

        pq = (b * Q + offs_q) * M + m
        val_base = value_ptr + (b * S * M + m) * D + offs_d[None, :]

        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for l in tl.static_range(L):
            H = tl.load(shapes_ptr + 2 * l)
            W = tl.load(shapes_ptr + 2 * l + 1)
            start = tl.load(starts_ptr + l)
            if not INT64:
                H = H.to(tl.int32)
                W = W.to(tl.int32)
                start = start.to(tl.int32)
            fh = H.to(tl.float32)
            fw = W.to(tl.float32)
            for k in tl.static_range(K):
                p = pq * (L * K) + l * K + k
                loc_x = tl.load(loc_ptr + 2 * p, mask=mask_q, other=0.0).to(tl.float32)
                loc_y = tl.load(loc_ptr + 2 * p + 1, mask=mask_q, other=0.0).to(tl.float32)
                attn = tl.load(attn_ptr + p, mask=mask_q, other=0.0).to(tl.float32)

                x = loc_x * fw - 0.5
                y = loc_y * fh - 0.5
                x0f = tl.math.floor(x)
                y0f = tl.math.floor(y)
                lx = x - x0f
                ly = y - y0f
                if INT64:
                    x0 = x0f.to(tl.int64)
                    y0 = y0f.to(tl.int64)
                else:
                    x0 = x0f.to(tl.int32)
                    y0 = y0f.to(tl.int32)
                x1 = x0 + 1
                y1 = y0 + 1

                vx0 = mask_q & (x0 >= 0) & (x0 < W)
                vx1 = mask_q & (x1 >= 0) & (x1 < W)
                vy0 = (y0 >= 0) & (y0 < H)
                vy1 = (y1 >= 0) & (y1 < H)
                row0 = start + y0 * W
                row1 = row0 + W

                v00 = tl.load(val_base + ((row0 + x0) * (M * D))[:, None],
                              mask=(vy0 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                v01 = tl.load(val_base + ((row0 + x1) * (M * D))[:, None],
                              mask=(vy0 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                v10 = tl.load(val_base + ((row1 + x0) * (M * D))[:, None],
                              mask=(vy1 & vx0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                v11 = tl.load(val_base + ((row1 + x1) * (M * D))[:, None],
                              mask=(vy1 & vx1)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

                acc += (
                    v00 * (attn * (1.0 - lx) * (1.0 - ly))[:, None]
                    + v01 * (attn * lx * (1.0 - ly))[:, None]
                    + v10 * (attn * (1.0 - lx) * ly)[:, None]
                    + v11 * (attn * lx * ly)[:, None]
                )

        out_ptrs = out_ptr + pq[:, None] * D + offs_d[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
                 mask=mask_q[:, None] & mask_d[None, :])

    return _fwd
'''


@app.function(gpu=GPU, timeout=2400, image=var_image)
def variants(preset: str, resolution: tuple, dtypes: list):
    import time

    import torch
    import triton

    sys.path[:0] = ["/root", "/root/tests"]
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

    # @triton.jit needs inspectable source, so materialize the variant
    # kernels as a real module file.
    Path("/root/_variant_kernels.py").write_text(KERNEL_SRC)
    from _variant_kernels import make_fwd_kernel

    small_cfgs = [
        triton.Config({"BLOCK_Q": bq}, num_warps=nw)
        for bq in (16, 32, 64) for nw in (2, 4)
    ]
    wide_cfgs = [
        triton.Config({"BLOCK_Q": bq}, num_warps=nw, num_stages=ns_)
        for bq in (16, 32, 64, 128) for nw in (2, 4, 8) for ns_ in (1, 3)
    ]

    kernels = {
        "base-i64": (make_fwd_kernel(small_cfgs), True),
        "base-i32": (make_fwd_kernel(small_cfgs), False),
        "wide-i32": (make_fwd_kernel(wide_cfgs), False),
    }

    dev = torch.cuda.get_device_name()
    print(f"=== {dev} preset={preset} res={resolution} ===")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    for dtype_name in dtypes:
        dtype = dtype_map[dtype_name]
        value, spatial_shapes, starts, loc, attn = _make_inputs(preset, resolution, dtype)
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = loc.shape
        out = value.new_empty(B, Q, M, D)
        print(f"\n--- dtype={dtype_name} B={B} Q={Q} S={S} ---")

        ref = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, starts, loc, attn, B
        )

        def time_graph(launch):
            g = torch.cuda.CUDAGraph()
            launch()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                launch()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(200):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / 200 * 1e3

        def time_eventful(launch):
            times = []
            for _ in range(100):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                launch()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            return sorted(times)[50] * 1e3

        # mmcv baseline
        with torch.no_grad():
            mmcv_launch = lambda: MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, starts, loc, attn, B
            )
            for _ in range(20):
                mmcv_launch()
            torch.cuda.synchronize()
            print(f"mmcv      graph={time_graph(mmcv_launch):7.1f}us eventful={time_eventful(mmcv_launch):7.1f}us")

        for name, (kern, use_i64) in kernels.items():
            grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_Q"]), B * M)

            def launch():
                kern[grid](
                    value, spatial_shapes, starts, loc, attn, out,
                    Q, S, M=M, D=D, L=L, K=K, INT64=use_i64,
                    BLOCK_D=triton.next_power_of_2(D),
                )

            for _ in range(30):  # warmup + autotune
                launch()
            torch.cuda.synchronize()
            err = (out.view(B, Q, M * D).float() - ref.float()).abs().max().item()
            g_us = time_graph(launch)
            e_us = time_eventful(launch)
            best = kern.best_config
            print(f"{name:9s} graph={g_us:7.1f}us eventful={e_us:7.1f}us err={err:.2e} best={best}")

        del value, loc, attn, ref, out
        torch.cuda.empty_cache()


@app.function(gpu=GPU, timeout=2400, image=var_image)
def bwd_sweep(preset: str, resolution: tuple, dtypes: list):
    """Sweep the (fixed, non-autotuned) backward launch config. Timing-only:
    grad_value accumulates garbage across launches, which does not change the
    atomic contention pattern."""
    import torch
    import triton

    sys.path[:0] = ["/root", "/root/tests"]
    from msda_triton.kernels import _msda_backward_kernel

    dev = torch.cuda.get_device_name()
    print(f"=== {dev} preset={preset} res={resolution} backward sweep ===")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    for dtype_name in dtypes:
        dtype = dtype_map[dtype_name]
        value, spatial_shapes, starts, loc, attn = _make_inputs(preset, resolution, dtype)
        B, S, M, D = value.shape
        _, Q, _, L, K, _ = loc.shape
        grad_out = torch.randn(B, Q, M, D, device="cuda", dtype=dtype)
        grad_value = torch.zeros_like(value)
        grad_loc = torch.empty_like(loc)
        grad_attn = torch.empty_like(attn)
        print(f"\n--- dtype={dtype_name} B={B} Q={Q} S={S} ---")

        for bq, nw in [(16, 2), (16, 4), (32, 2), (32, 4), (32, 8), (64, 4), (64, 8), (128, 8)]:
            grid = (triton.cdiv(Q, bq), B * M)

            def launch():
                _msda_backward_kernel[grid](
                    value, spatial_shapes, starts, loc, attn,
                    grad_out, grad_value, grad_loc, grad_attn,
                    Q, S, M=M, D=D, L=L, K=K,
                    BLOCK_Q=bq, BLOCK_D=triton.next_power_of_2(D), num_warps=nw,
                )

            for _ in range(15):
                launch()
            torch.cuda.synchronize()
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(100)
            ]
            for s, e in events:
                s.record()
                launch()
                e.record()
            torch.cuda.synchronize()
            p50 = sorted(s.elapsed_time(e) for s, e in events)[50] * 1e3
            print(f"BLOCK_Q={bq:4d} num_warps={nw}  {p50:8.1f}us")

        del value, loc, attn, grad_out, grad_value, grad_loc, grad_attn
        torch.cuda.empty_cache()


@app.local_entrypoint()
def var_main(
    preset: str = "decoder",
    resolution: str = "800x1333",
    dtypes: str = "fp32,fp16",
    sweep_bwd: bool = False,
):
    h, w = (int(x) for x in resolution.split("x"))
    dt = [d.strip() for d in dtypes.split(",")]
    if sweep_bwd:
        bwd_sweep.remote(preset, (h, w), dt)
    else:
        variants.remote(preset, (h, w), dt)
