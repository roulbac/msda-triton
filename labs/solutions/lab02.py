"""Lab 2 solution: MSDA on flat 1-D memory with hand-computed offsets.

This is the Triton kernel's exact pointer arithmetic, executed safely in
torch: every tensor is a flattened 1-D view, and every access site computes
its flat index by hand — (b·Q+q)·M+m, start + y·W + x, ((b·S+s)·M+m)·D+d.
Vectorized over (B, Q, M) with Python loops only over the small L×K axes,
which is precisely the shape of the kernel in Lab 5.
"""

import torch


def flat_msda(value, spatial_shapes, level_start_index, sampling_locations,
              attention_weights):
    """Same contract as the repo operator; only 1-D views + flat offsets.

    Returns (B, Q, M*D).
    """
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    dev = value.device
    # Accumulate at the input's precision or better (float32 minimum).
    acc_dtype = value.dtype if value.dtype == torch.float64 else torch.float32

    # The "memory": everything as flat 1-D buffers, like pointers.
    v = value.reshape(-1)
    loc = sampling_locations.reshape(-1).to(acc_dtype)
    attn = attention_weights.reshape(-1).to(acc_dtype)

    # Index vectors over the parallel axes (the kernel's program ids).
    b = torch.arange(B, dtype=torch.int64, device=dev)[:, None, None]
    q = torch.arange(Q, dtype=torch.int64, device=dev)[None, :, None]
    m = torch.arange(M, dtype=torch.int64, device=dev)[None, None, :]
    d = torch.arange(D, dtype=torch.int64, device=dev)

    # Flat (b, q, m) index shared by the loc and attn layouts. int64: at
    # encoder scale b*S*M*D overflows int32 (see the lab's overflow exercise).
    pq = (b * Q + q) * M + m                              # (B, Q, M)
    val_base = (b * S + 0) * M + m                        # (B, Q, M) broadcast

    out = torch.zeros(B, Q, M, D, dtype=acc_dtype, device=dev)
    for lvl in range(L):
        H = int(spatial_shapes[lvl, 0])
        W = int(spatial_shapes[lvl, 1])
        start = int(level_start_index[lvl])
        for k in range(K):
            p = pq * (L * K) + lvl * K + k                # (B, Q, M)
            x = loc[2 * p] * W - 0.5
            y = loc[2 * p + 1] * H - 0.5
            a = attn[p]

            x0 = torch.floor(x)
            y0 = torch.floor(y)
            lx, ly = x - x0, y - y0
            x0, y0 = x0.long(), y0.long()
            x1, y1 = x0 + 1, y0 + 1

            vx0 = (x0 >= 0) & (x0 < W)
            vx1 = (x1 >= 0) & (x1 < W)
            vy0 = (y0 >= 0) & (y0 < H)
            vy1 = (y1 >= 0) & (y1 < H)
            row0 = start + y0 * W
            row1 = row0 + W

            def corner(row, xc, valid, wgt):
                # Flat element index of channel 0: ((b*S + s)*M + m)*D.
                s = row + xc
                idx = ((val_base + s * M) * D)[..., None] + d  # (B,Q,M,D)
                # Invalid corners: clamp so the gather is legal, zero after —
                # torch's version of the kernel's mask + other=0.0.
                idx = idx.clamp(0, v.numel() - 1)
                vals = v[idx].to(acc_dtype) * valid[..., None]
                return vals * (a * wgt)[..., None]

            out += corner(row0, x0, vy0 & vx0, (1 - lx) * (1 - ly))
            out += corner(row0, x1, vy0 & vx1, lx * (1 - ly))
            out += corner(row1, x0, vy1 & vx0, (1 - lx) * ly)
            out += corner(row1, x1, vy1 & vx1, lx * ly)
    return out.view(B, Q, M * D).to(value.dtype)
