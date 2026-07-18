"""Lab 3 solution: the memory-traffic counters for the paper-GPU simulator."""

import numpy as np

from labs.common.memsim import SECTOR_BYTES, AccessPattern


def sectors_touched(pattern: AccessPattern) -> int:
    """Total DRAM sectors fetched: for each warp access, the number of
    *distinct* 32-byte sectors its addresses fall in (addresses are assumed
    aligned, so one element never straddles two sectors)."""
    total = 0
    for warp in pattern.warps:
        total += len(np.unique(np.asarray(warp) // SECTOR_BYTES))
    return total


def efficiency(pattern: AccessPattern) -> float:
    """Useful bytes / bytes actually moved. 1.0 = perfectly coalesced."""
    return pattern.useful_bytes / (sectors_touched(pattern) * SECTOR_BYTES)


def effective_bandwidth(pattern: AccessPattern, peak_gbps: float) -> float:
    """The bandwidth the program *experiences*: peak scaled by efficiency."""
    return peak_gbps * efficiency(pattern)
