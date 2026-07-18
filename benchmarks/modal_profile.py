"""One-off profiling harness: separate pure GPU kernel time from Python/launch
overhead for the triton vs mmcv MSDA forward/backward at decoder scale.

Reuses the benchmark image. Run:
    MSDA_GPU=L40S uv run modal run benchmarks/modal_profile.py --dtypes fp32,fp16
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_benchmark import _make_inputs, app, image  # noqa: E402

GPU = os.environ.get("MSDA_GPU", "L40S")

# The container imports this very module, which imports modal_benchmark —
# mount it next to /root so both resolve remotely.
prof_image = image.add_local_file(
    HERE / "modal_benchmark.py", remote_path="/root/modal_benchmark.py"
)


@app.function(gpu=GPU, timeout=1800, image=prof_image)
def profile(preset: str, resolution: tuple, dtypes: list):
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile

    sys.path[:0] = ["/root", "/root/tests"]
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

    from msda_triton import multi_scale_deformable_attention
    from msda_triton.kernels import _msda_forward_kernel

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    print(f"=== {dev} (SM {cap[0]}.{cap[1]}) preset={preset} res={resolution} ===")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    for dtype_name in dtypes:
        dtype = dtype_map[dtype_name]
        value, spatial_shapes, starts, loc, attn = _make_inputs(preset, resolution, dtype)
        B, Q = value.shape[0], loc.shape[1]
        print(f"\n--- dtype={dtype_name} B={B} Q={Q} S={value.shape[1]} ---")

        impls = {
            "mmcv": lambda: MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, starts, loc, attn, B
            ),
            "triton": lambda: multi_scale_deformable_attention(
                value, spatial_shapes, starts, loc, attn, fp32_grad_accum=False
            ),
        }

        with torch.no_grad():
            for name, fn in impls.items():
                for _ in range(20):  # warmup incl. autotune
                    fn()
                torch.cuda.synchronize()

                # 1) benchmark-style: per-iter events, sync each iter (includes
                #    dispatch latency when the GPU is idle)
                times = []
                for _ in range(100):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                eventful = sorted(times)[50]

                # 2) amortized: enqueue N back-to-back, wall clock / N.
                #    GPU-bound => ~kernel time; launch-bound => ~dispatch time.
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                N = 200
                for _ in range(N):
                    fn()
                torch.cuda.synchronize()
                amort = (time.perf_counter() - t0) / N * 1e3

                # 3) pure CPU dispatch cost (GPU still busy from prior queue)
                for _ in range(50):
                    fn()  # keep queue full
                t0 = time.perf_counter()
                for _ in range(100):
                    fn()
                cpu_dispatch = (time.perf_counter() - t0) / 100 * 1e3
                torch.cuda.synchronize()

                # 4) profiler: pure GPU kernel durations
                with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(50):
                        fn()
                    torch.cuda.synchronize()
                kernels = {}
                for evt in prof.key_averages():
                    if evt.device_type is not None and str(evt.device_type) == "DeviceType.CUDA" and evt.self_device_time_total > 0:
                        kernels[evt.key] = (evt.self_device_time_total / 50, evt.count)
                gpu_total = sum(v[0] for v in kernels.values()) / 1e3  # us->ms

                print(
                    f"{name:8s} eventful={eventful*1e3:7.1f}us amortized={amort*1e3:7.1f}us "
                    f"cpu_dispatch={cpu_dispatch*1e3:7.1f}us gpu_kernel={gpu_total*1e3:7.1f}us"
                )
                for k, (us, cnt) in sorted(kernels.items(), key=lambda x: -x[1][0]):
                    print(f"    {us:8.1f}us x{cnt/50:4.1f}  {k[:90]}")

        # autotuner choice
        try:
            print("forward autotune best:", _msda_forward_kernel.best_config)
        except Exception as err:
            print("no best_config:", err)

        # 5) CUDA-graph capture of the triton forward (zero Python overhead)
        with torch.no_grad():
            try:
                g = torch.cuda.CUDAGraph()
                impls["triton"]()
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    impls["triton"]()
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(100):
                    g.replay()
                e.record()
                torch.cuda.synchronize()
                print(f"triton graph-replay: {s.elapsed_time(e) / 100 * 1e3:7.1f}us")
            except Exception as err:
                print("graph capture failed:", err)
            try:
                g2 = torch.cuda.CUDAGraph()
                impls["mmcv"]()
                torch.cuda.synchronize()
                with torch.cuda.graph(g2):
                    impls["mmcv"]()
                for _ in range(5):
                    g2.replay()
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(100):
                    g2.replay()
                e.record()
                torch.cuda.synchronize()
                print(f"mmcv   graph-replay: {s.elapsed_time(e) / 100 * 1e3:7.1f}us")
            except Exception as err:
                print("mmcv graph capture failed:", err)

        # ---- backward: pure GPU kernel time via retained-graph backwards ----
        print("backward:")
        bwd_impls = {
            "mmcv": lambda v, l, a: MultiScaleDeformableAttnFunction.apply(
                v, spatial_shapes, starts, l, a, B
            ),
            "triton": lambda v, l, a: multi_scale_deformable_attention(
                v, spatial_shapes, starts, l, a, fp32_grad_accum=False
            ),
            "triton-fp32acc": lambda v, l, a: multi_scale_deformable_attention(
                v, spatial_shapes, starts, l, a, fp32_grad_accum=True
            ),
        }
        for name, f in bwd_impls.items():
            v = value.clone().requires_grad_(True)
            l = loc.clone().requires_grad_(True)
            a = attn.clone().requires_grad_(True)
            out = f(v, l, a)
            grad_out = torch.randn_like(out)

            def one_bwd():
                v.grad = l.grad = a.grad = None
                out.backward(grad_out, retain_graph=True)

            for _ in range(20):
                one_bwd()
            torch.cuda.synchronize()

            times = []
            for _ in range(100):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                one_bwd()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            eventful = sorted(times)[50]

            with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(50):
                    one_bwd()
                torch.cuda.synchronize()
            kernels = {}
            for evt in prof.key_averages():
                if str(evt.device_type) == "DeviceType.CUDA" and evt.self_device_time_total > 0:
                    kernels[evt.key] = (evt.self_device_time_total / 50, evt.count)
            gpu_total = sum(us for us, _ in kernels.values()) / 1e3

            print(f"{name:15s} eventful={eventful*1e3:7.1f}us gpu_kernel={gpu_total*1e3:7.1f}us")
            for k, (us, cnt) in sorted(kernels.items(), key=lambda x: -x[1][0]):
                print(f"    {us:8.1f}us x{cnt/50:4.1f}  {k[:90]}")
            del v, l, a, out, grad_out

        del value, loc, attn
        torch.cuda.empty_cache()


@app.local_entrypoint()
def prof(preset: str = "decoder", resolution: str = "800x1333", dtypes: str = "fp32,fp16"):
    h, w = (int(x) for x in resolution.split("x"))
    profile.remote(preset, (h, w), [d.strip() for d in dtypes.split(",")])
