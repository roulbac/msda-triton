"""Ahead-of-time compile the Triton MSDA kernels for SM 8.0 and SM 9.0.

Needs no GPU: Triton compiles to PTX with an explicit target, so this runs on
CPU-only CI and catches kernel-body typing/API errors that a plain import
cannot (the @triton.jit body is only parsed and type-checked at compile time).

It also prints the atomic instructions each backward variant lowers to, which
is the mechanism the accumulator-precision switch relies on (paper Section
4.2): on SM 8.0 a native-bf16 grad_value emits a compare-and-swap loop, while
the FP32-accumulator variant and SM 9.0 emit native relaxed adds. The atomics
are printed rather than asserted so that a future Triton changing its exact
lowering does not fail CI.
"""

import re

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from msda_triton.kernels import _msda_backward_kernel, _msda_forward_kernel

CONSTS = dict(M=8, D=32, L=4, K=4, BLOCK_Q=32, BLOCK_D=32)


def signature(jit_fn, ptr_dtypes):
    sig = {}
    for name in jit_fn.arg_names:
        if name in CONSTS:
            sig[name] = "constexpr"
        elif name.endswith("_ptr"):
            sig[name] = ptr_dtypes[name]
        else:
            sig[name] = "i64"
    return sig


def compile_variant(jit_fn, ptr_dtypes, cc):
    src = ASTSource(jit_fn, signature(jit_fn, ptr_dtypes), dict(CONSTS))
    return triton.compile(src, target=GPUTarget("cuda", cc, 32))


def main():
    fwd = _msda_forward_kernel.fn  # unwrap Autotuner -> JITFunction
    bwd = _msda_backward_kernel

    for dt in ("fp32", "fp16", "bf16"):
        fwd_ptrs = dict(
            value_ptr=f"*{dt}", shapes_ptr="*i64", starts_ptr="*i64",
            loc_ptr=f"*{dt}", attn_ptr=f"*{dt}", out_ptr=f"*{dt}",
        )
        grad_ptrs = dict(
            fwd_ptrs,
            grad_out_ptr=f"*{dt}",
            grad_loc_ptr=f"*{dt}",
            grad_attn_ptr=f"*{dt}",
        )
        del grad_ptrs["out_ptr"]
        variants = [
            ("fwd", fwd, fwd_ptrs),
            ("bwd-native", bwd, dict(grad_ptrs, grad_value_ptr=f"*{dt}")),
            ("bwd-fp32acc", bwd, dict(grad_ptrs, grad_value_ptr="*fp32")),
        ]
        for cc in (80, 90):
            for tag, jit_fn, ptrs in variants:
                kernel = compile_variant(jit_fn, ptrs, cc)
                atoms = sorted(set(re.findall(r"atom\.[\w.]+", kernel.asm["ptx"])))
                extra = f"  atomics: {', '.join(atoms)}" if atoms else ""
                print(f"OK sm{cc} {tag:<12} {dt}{extra}")
    print("all kernel variants compiled")


if __name__ == "__main__":
    main()
