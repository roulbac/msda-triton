"""Lab 7 solution: MSDA's three gradients, derived by hand, in pure torch.

The scatter into grad_value uses ``index_add_`` on a flat buffer — torch's
safe, serial stand-in for the atomic adds the Triton backward will need.
Structure deliberately mirrors lab02's flat_msda: loops over (l, k),
vectorized over (B, Q, M), all indices flat.
"""

import torch


def msda_backward_torch(value, spatial_shapes, level_start_index,
                        sampling_locations, attention_weights, grad_out):
    """Analytic gradients. grad_out: (B, Q, M*D).

    Returns (grad_value (B,S,M,D), grad_loc (B,Q,M,L,K,2),
    grad_attn (B,Q,M,L,K)), all float32.
    """
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    dev = value.device

    v = value.reshape(-1).float()
    loc = sampling_locations.float()
    attn = attention_weights.float()
    g = grad_out.reshape(B, Q, M, D).float()

    grad_value = torch.zeros(B * S * M * D, dtype=torch.float32, device=dev)
    grad_loc = torch.zeros(B, Q, M, L, K, 2, dtype=torch.float32, device=dev)
    grad_attn = torch.zeros(B, Q, M, L, K, dtype=torch.float32, device=dev)

    b = torch.arange(B, dtype=torch.int64, device=dev)[:, None, None]
    m = torch.arange(M, dtype=torch.int64, device=dev)[None, None, :]
    d = torch.arange(D, dtype=torch.int64, device=dev)
    val_base = (b * S) * M + m                     # (B, 1, M), + s*M selects pixel

    for lvl in range(L):
        H = int(spatial_shapes[lvl, 0])
        W = int(spatial_shapes[lvl, 1])
        start = int(level_start_index[lvl])
        for k in range(K):
            x = loc[:, :, :, lvl, k, 0] * W - 0.5          # (B, Q, M)
            y = loc[:, :, :, lvl, k, 1] * H - 0.5
            a = attn[:, :, :, lvl, k]

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

            def gather(row, xc, valid):
                idx = ((val_base + (row + xc) * M) * D)[..., None] + d
                idx = idx.clamp(0, v.numel() - 1)
                return v[idx] * valid[..., None]           # (B, Q, M, D)

            v00 = gather(row0, x0, vy0 & vx0)
            v01 = gather(row0, x1, vy0 & vx1)
            v10 = gather(row1, x0, vy1 & vx0)
            v11 = gather(row1, x1, vy1 & vx1)

            w00 = (1 - lx) * (1 - ly)
            w01 = lx * (1 - ly)
            w10 = (1 - lx) * ly
            w11 = lx * ly

            # d out / d attn: the bilinearly sampled value, dotted with g.
            sampled = (v00 * w00[..., None] + v01 * w01[..., None]
                       + v10 * w10[..., None] + v11 * w11[..., None])
            grad_attn[:, :, :, lvl, k] = (g * sampled).sum(-1)

            # d out / d location: bilinear is piecewise-linear per axis, so
            # its x-derivative is the corner difference, blended in y —
            # chained through x_im = x*W - 0.5, hence the *W (and *H).
            dx = (v01 - v00) * (1 - ly)[..., None] + (v11 - v10) * ly[..., None]
            dy = (v10 - v00) * (1 - lx)[..., None] + (v11 - v01) * lx[..., None]
            grad_loc[:, :, :, lvl, k, 0] = a * W * (g * dx).sum(-1)
            grad_loc[:, :, :, lvl, k, 1] = a * H * (g * dy).sum(-1)

            # d out / d value: corner (y,x) received coefficient a*w, so it
            # gets back a*w*g — scattered to wherever this sample landed.
            wg = g * a[..., None]                          # (B, Q, M, D)

            def scatter(row, xc, valid, wgt):
                idx = ((val_base + (row + xc) * M) * D)[..., None] + d
                contrib = wg * wgt[..., None] * valid[..., None]
                grad_value.index_add_(
                    0, idx.clamp(0, v.numel() - 1).reshape(-1),
                    contrib.reshape(-1),
                )

            scatter(row0, x0, vy0 & vx0, w00)
            scatter(row0, x1, vy0 & vx1, w01)
            scatter(row1, x0, vy1 & vx0, w10)
            scatter(row1, x1, vy1 & vx1, w11)

    return grad_value.view(B, S, M, D), grad_loc, grad_attn
