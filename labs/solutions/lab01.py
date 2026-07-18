"""Lab 1 solution: bilinear sampling and naive MSDA, in plain Python."""

import math

import torch


def bilinear_sample(img: torch.Tensor, x: float, y: float) -> torch.Tensor:
    """Sample ``img`` (H, W, D) at continuous pixel coords (x, y).

    Blends the 4 surrounding pixels; corners outside the image contribute
    zero ("zeros padding"). Returns a (D,) vector.
    """
    H, W, _ = img.shape
    x0, y0 = math.floor(x), math.floor(y)
    lx, ly = x - x0, y - y0
    out = img.new_zeros(img.shape[-1])
    corners = (
        (y0,     x0,     (1 - lx) * (1 - ly)),
        (y0,     x0 + 1, lx       * (1 - ly)),
        (y0 + 1, x0,     (1 - lx) * ly),
        (y0 + 1, x0 + 1, lx       * ly),
    )
    for yy, xx, w in corners:
        if 0 <= xx < W and 0 <= yy < H:
            out += w * img[yy, xx]
    return out


def msda_naive_student(value, spatial_shapes, sampling_locations, attention_weights):
    """MSDA by definition: pure loops, one bilinear sample per (b,q,m,l,k).

    Args mirror the repo operator: value (B, S, M, D) with the level images
    flattened along S; sampling_locations (B, Q, M, L, K, 2) normalized to
    [0, 1]; attention_weights (B, Q, M, L, K). Returns (B, Q, M*D).
    """
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes]
    starts = [0]
    for h, w in shapes[:-1]:
        starts.append(starts[-1] + h * w)

    out = value.new_zeros(B, Q, M, D)
    for b in range(B):
        for q in range(Q):
            for m in range(M):
                for lvl in range(L):
                    h, w = shapes[lvl]
                    # This level's image for head m, as (h, w, D).
                    img = value[b, starts[lvl]:starts[lvl] + h * w, m].view(h, w, D)
                    for k in range(K):
                        # Normalized [0,1] -> pixel centers: x*W - 0.5.
                        x = float(sampling_locations[b, q, m, lvl, k, 0]) * w - 0.5
                        y = float(sampling_locations[b, q, m, lvl, k, 1]) * h - 0.5
                        a = attention_weights[b, q, m, lvl, k]
                        out[b, q, m] += a * bilinear_sample(img, x, y)
    return out.view(B, Q, M * D)
