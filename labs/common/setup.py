"""Lab environment detection. Import this BEFORE importing triton.

On a machine without a CUDA GPU, Triton kernels can still *run* through
Triton's CPU interpreter, which is enabled by the environment variable
``TRITON_INTERPRET=1`` — but only if it is set before ``import triton``.
Importing this module first (in the same cell that then imports triton)
guarantees the ordering inside a marimo notebook, where separate cells run
in dependency order, not top-to-bottom.

The interpreter executes kernels element-by-element in Python: results are
correct, timings are meaningless. Labs mark every cell whose *point* is
performance, and those need a real GPU (locally or via the repo's Modal
harness — see labs/README.md).
"""

import os


def _detect_cuda() -> bool:
    import torch

    has_cuda = torch.cuda.is_available()
    # Respect an explicit user setting (e.g. Lab 9 forces "0" because
    # ahead-of-time compilation needs the real compiler, not the interpreter).
    if not has_cuda and os.environ.get("TRITON_INTERPRET") is None:
        os.environ["TRITON_INTERPRET"] = "1"
    return has_cuda


HAS_CUDA = _detect_cuda()
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if HAS_CUDA else "cpu"


def import_triton():
    """Import triton with a friendly error for CPU-only torch builds.

    Returns ``(triton, triton.language)``.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as e:  # e.g. macOS torch wheels do not depend on triton
        raise ImportError(
            "triton is not installed. On Linux, `uv sync` installs it via torch; "
            "CUDA builds of PyTorch also bundle it. The labs need Triton to run "
            "kernels through the CPU interpreter when no GPU is available."
        ) from e
    return triton, tl


def banner(mo):
    """A marimo callout describing the execution mode of this session."""
    if HAS_CUDA:
        import torch

        return mo.md(
            f"🟢 **GPU mode** — kernels run compiled on "
            f"`{torch.cuda.get_device_name()}` "
            f"(SM {'.'.join(map(str, torch.cuda.get_device_capability()))}). "
            f"Both correctness and performance cells are meaningful."
        ).callout(kind="success")
    if INTERPRETED:
        return mo.md(
            "🟡 **CPU interpreter mode** (`TRITON_INTERPRET=1`) — kernels run "
            "correctly but element-by-element in Python. Every ✅/❌ check in "
            "this lab is valid; every *timing* is meaningless. Cells marked "
            "**GPU required** need real hardware (see labs/README.md for the "
            "Modal fallback)."
        ).callout(kind="warn")
    return mo.md(
        "🔵 **Compiled, no GPU** (`TRITON_INTERPRET=0`) — this lab only "
        "*compiles* kernels (ahead-of-time, to PTX), which needs no GPU."
    ).callout(kind="info")
