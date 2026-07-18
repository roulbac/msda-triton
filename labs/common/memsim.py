"""Scaffold for Lab 3's paper-GPU memory-traffic simulator.

The model: a *warp* is 32 threads that issue one memory access each, at the
same instant. DRAM doesn't deliver individual bytes — it delivers 32-byte
**sectors**. The cost of a warp's access is the number of *distinct* sectors
its 32 addresses touch; the value is the number of bytes the program actually
wanted. Everything Lab 3 teaches falls out of that ratio.

You implement the counting (``sectors_touched`` / ``efficiency`` in the
notebook). This module supplies the shared vocabulary: an ``AccessPattern``
(a list of per-warp address arrays) and generators for the patterns worth
studying — including the two candidate MSDA tilings, so you can *predict*
the paper's ~7x effective-bandwidth gap before ever touching a GPU.
"""

from dataclasses import dataclass, field

import numpy as np

SECTOR_BYTES = 32
WARP_SIZE = 32


@dataclass
class AccessPattern:
    """A named sequence of warp-level memory accesses.

    ``warps`` is a list of int arrays of byte addresses; each array is one
    simultaneous access by one warp (usually WARP_SIZE addresses; fewer means
    some lanes are masked off). ``itemsize`` is the bytes each lane wants.
    """

    name: str
    itemsize: int
    warps: list = field(default_factory=list)

    @property
    def useful_bytes(self) -> int:
        return sum(len(w) for w in self.warps) * self.itemsize


def contiguous(n=1024, itemsize=4, base=0):
    """Lane i reads element i: the pattern every GPU loves."""
    addrs = base + np.arange(n, dtype=np.int64) * itemsize
    return AccessPattern("contiguous", itemsize, _chunk(addrs))


def strided(stride_elems, n=1024, itemsize=4, base=0):
    """Lane i reads element i*stride: what a bad layout choice produces."""
    addrs = base + np.arange(n, dtype=np.int64) * stride_elems * itemsize
    return AccessPattern(f"strided x{stride_elems}", itemsize, _chunk(addrs))


def random_gather(n=1024, span_bytes=64 * 1024 * 1024, itemsize=4, seed=0):
    """Lane i reads a random element: the pathological worst case."""
    rng = np.random.default_rng(seed)
    addrs = (rng.integers(0, span_bytes // itemsize, n) * itemsize).astype(np.int64)
    return AccessPattern("random gather", itemsize, _chunk(addrs))


def _chunk(addrs):
    return [addrs[i:i + WARP_SIZE] for i in range(0, len(addrs), WARP_SIZE)]


# ---------------------------------------------------------------------------
# The two MSDA tilings, as address streams.
#
# Both simulate the corner gathers for one (batch, head): each of n_queries
# queries samples a random pixel row (data-dependent — the network chose it),
# and the kernel must read that row's D consecutive channels. What differs is
# ONLY how the work maps onto threads. Same data, same bytes wanted.
# ---------------------------------------------------------------------------


def _sample_rows(n_queries, n_pixels, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_pixels, n_queries).astype(np.int64)


def msda_query_block(n_queries=256, D=32, itemsize=2, n_pixels=100_000, seed=0):
    """Query-block tiling (this repo): a warp covers one query's D channels
    (D=32 -> one warp per gathered row). Rows are scattered, but *within* a
    warp the addresses are consecutive channels of one row."""
    rows = _sample_rows(n_queries, n_pixels, seed)
    row_stride = D * itemsize
    warps = []
    for r in rows:
        base = r * row_stride
        warps.append(base + np.arange(D, dtype=np.int64) * itemsize)
    return AccessPattern("query-block tiling", itemsize, warps)


def msda_point_parallel(n_queries=256, D=32, itemsize=2, n_pixels=100_000, seed=0):
    """Point-parallel tiling (the "obvious" one): each thread owns one query's
    sample and walks its D channels *serially*. A warp is 32 *different*
    queries reading channel d of 32 unrelated rows at each step."""
    rows = _sample_rows(n_queries, n_pixels, seed)
    row_stride = D * itemsize
    warps = []
    for group in range(0, n_queries, WARP_SIZE):
        lane_rows = rows[group:group + WARP_SIZE]
        bases = lane_rows * row_stride
        for d in range(D):  # step d: every lane reads channel d of its row
            warps.append(bases + d * itemsize)
    return AccessPattern("point-parallel tiling", itemsize, warps)
