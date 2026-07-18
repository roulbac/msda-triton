"""Slow but simple reference implementations of MSDA, used as correctness
ground truth by the test suite and the benchmark harness. They ship with the
testing code rather than the msda_triton package. Both run on CPU or GPU, in
any float dtype, and are fully differentiable through PyTorch autograd."""

import math

import torch
import torch.nn.functional as F


def msda_reference(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """MSDA via per-level ``F.grid_sample`` — the standard PyTorch fallback
    (same math as mmcv's ``multi_scale_deformable_attn_pytorch``).

    Args mirror :func:`msda_triton.multi_scale_deformable_attention`
    (``level_start_index`` is implicit in the level-wise split).
    Returns ``(B, Q, M * D)``.
    """
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    # [0, 1] -> [-1, 1] grid_sample coordinates (align_corners=False maps
    # grid g to x_im = ((g + 1) * W - 1) / 2 = x * W - 0.5).
    grids = 2 * sampling_locations - 1
    sampled = []
    for lvl, (h, w) in enumerate(shapes):
        # (B, H*W, M, D) -> (B*M, D, H, W)
        v = value_list[lvl].flatten(2).transpose(1, 2).reshape(B * M, D, h, w)
        # (B, Q, M, K, 2) -> (B*M, Q, K, 2)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    # L x (B*M, D, Q, K) -> (B*M, D, Q, L*K)
    sampled = torch.stack(sampled, dim=-2).flatten(-2)
    attn = attention_weights.transpose(1, 2).reshape(B * M, Q, L * K).unsqueeze(1)
    out = (sampled * attn).sum(-1)  # (B*M, D, Q)
    return out.view(B, M * D, Q).transpose(1, 2).contiguous()


def msda_naive(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-loop MSDA with hand-rolled bilinear interpolation. O(B*Q*M*L*K)
    Python iterations — only for tiny inputs, where it validates the
    grid_sample-based reference independently. Returns ``(B, Q, M * D)``."""
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
                    for k in range(K):
                        x = float(sampling_locations[b, q, m, lvl, k, 0]) * w - 0.5
                        y = float(sampling_locations[b, q, m, lvl, k, 1]) * h - 0.5
                        a = attention_weights[b, q, m, lvl, k]
                        x0, y0 = math.floor(x), math.floor(y)
                        lx, ly = x - x0, y - y0
                        corners = (
                            (y0, x0, (1 - lx) * (1 - ly)),
                            (y0, x0 + 1, lx * (1 - ly)),
                            (y0 + 1, x0, (1 - lx) * ly),
                            (y0 + 1, x0 + 1, lx * ly),
                        )
                        for yy, xx, wgt in corners:
                            if 0 <= xx < w and 0 <= yy < h:
                                out[b, q, m] += (
                                    a * wgt * value[b, starts[lvl] + yy * w + xx, m]
                                )
    return out.view(B, Q, M * D)
