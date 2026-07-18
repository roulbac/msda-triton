"""Lab 9 solution: the accumulator-precision switch, and reading your own PTX.

Compilation needs no GPU: triton.compile takes an explicit target, so this
whole lab runs on a laptop. It does need the real compiler — Lab 9's
notebook sets TRITON_INTERPRET=0 before any triton import.
"""

import re

import torch


# --- the switch (a pure function: trivially testable, no GPU needed) ---------


def use_fp32_grad_accum(dtype: torch.dtype, capability: tuple) -> bool:
    """Should grad_value accumulate in an FP32 scratch buffer?

    Yes exactly when the value dtype's atomic add would be *emulated*: BF16
    has no native atomic add before SM 9.0 (the emulation is a compare-and-
    swap retry loop, ~10x slower). FP16 atomics are native since SM 6.0 and
    FP32 always — routing those through the scratch buffer only adds traffic.
    """
    return dtype == torch.bfloat16 and capability[0] < 9


def pick_accum_dtype(value: torch.Tensor, fp32_grad_accum=None) -> torch.dtype:
    """The dtype grad_value should be allocated in (mirrors ops.py)."""
    if fp32_grad_accum is None:
        fp32_grad_accum = use_fp32_grad_accum(
            value.dtype, torch.cuda.get_device_capability(value.device)
            if value.is_cuda else (8, 0)
        )
    return torch.float32 if fp32_grad_accum else value.dtype


# --- ahead-of-time compilation + PTX forensics -------------------------------

TRITON_DTYPE = {"fp32": "*fp32", "fp16": "*fp16", "bf16": "*bf16"}


def compile_backward(kernel, value_dtype: str, grad_value_dtype: str, cc: int,
                     M=8, D=32, L=4, K=4, BLOCK_Q=16, BLOCK_D=32):
    """AOT-compile a backward kernel for compute capability ``cc`` (80, 90).

    ``kernel`` is a @triton.jit function with lab08's signature. Returns the
    compiled artifact; its ``.asm["ptx"]`` is the generated PTX.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    consts = dict(M=M, D=D, L=L, K=K, BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D)
    dt = TRITON_DTYPE[value_dtype]
    ptrs = dict(
        value_ptr=dt, shapes_ptr="*i64", starts_ptr="*i64",
        loc_ptr=dt, attn_ptr=dt, grad_out_ptr=dt,
        grad_value_ptr=TRITON_DTYPE[grad_value_dtype],
        grad_loc_ptr=dt, grad_attn_ptr=dt,
    )
    sig = {}
    for name in kernel.arg_names:
        if name in consts:
            sig[name] = "constexpr"
        elif name.endswith("_ptr"):
            sig[name] = ptrs[name]
        else:
            sig[name] = "i64"
    src = ASTSource(kernel, sig, consts)
    return triton.compile(src, target=GPUTarget("cuda", cc, 32))


def find_atomics(ptx: str) -> list:
    """Every distinct atomic/reduction instruction in a PTX listing.

    ``red.*`` is the fire-and-forget hardware reduction (the fast path);
    ``atom.*.cas`` inside a retry loop is the emulation (the slow path).
    """
    return sorted(set(re.findall(r"\b(?:red|atom)\.[\w.:]+", ptx)))


def atomic_report(kernel, cc_list=(80, 90), dtypes=("fp32", "fp16", "bf16")):
    """{(cc, value_dtype, accum): [instructions]} for native vs fp32-scratch
    grad_value across compute capabilities — Lab 9's money table."""
    report = {}
    for cc in cc_list:
        for dt in dtypes:
            for accum in ("native", "fp32acc"):
                gv = dt if accum == "native" else "fp32"
                compiled = compile_backward(kernel, dt, gv, cc)
                report[(cc, dt, accum)] = find_atomics(compiled.asm["ptx"])
    return report
