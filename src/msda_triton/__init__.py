"""Triton multi-scale deformable attention (MSDA).

Implements the query-block Triton kernels from "Efficient Multi-Scale
Deformable Attention on GPUs". Slow reference implementations used for
correctness testing ship with the test suite (tests/reference_impls.py),
not with this package.
"""

from .ops import multi_scale_deformable_attention

__all__ = ["multi_scale_deformable_attention"]
__version__ = "0.1.0"
