"""Lab 0 solution: deformable attention, deformable cross-attention, and
multi-scale deformable attention (MSDA) as plain PyTorch layers, built on
``F.grid_sample`` rather than hand-rolled interpolation."""

import torch
import torch.nn.functional as F
from torch import nn


def bilinear_sample(img: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample ``img`` (H, W, D) at continuous pixel coords ``points`` (N, 2).

    ``points[:, 0]`` is x, ``points[:, 1]`` is y. Implemented with
    ``F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)``
    — the same convention the repo uses everywhere (pixel centers at
    integers, zeros padding). Differentiable w.r.t. both ``img`` and
    ``points``. Returns (N, D).
    """
    H, W, D = img.shape
    # pixel coords -> grid_sample's [-1, 1] (align_corners=False maps grid g
    # to x_im = ((g + 1) * W - 1) / 2 = x * W - 0.5; same for y with H).
    scale = points.new_tensor([W, H])
    grid = 2 * (points + 0.5) / scale - 1            # (N, 2), (x, y)
    grid = grid[None, :, None, :]                      # (1, N, 1, 2)
    v = img.permute(2, 0, 1)[None]                      # (1, D, H, W)
    sampled = F.grid_sample(
        v, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )                                                  # (1, D, N, 1)
    return sampled[0, :, :, 0].T                        # (N, D)


def deform_attend(value: torch.Tensor, points: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """The deformable-attention core, one head on one image.

    ``value`` (H, W, D); ``points`` (Q, K, 2) in pixel coords (reference
    point + offset, already combined); ``weights`` (Q, K). Returns (Q, D):
    ``out[q] = sum_k weights[q, k] * bilinear_sample(value, points[q, k])``.
    """
    Q, K, _ = points.shape
    samples = bilinear_sample(value, points.reshape(Q * K, 2)).reshape(Q, K, -1)
    return (weights.unsqueeze(-1) * samples).sum(dim=1)


def _attend_heads(value, points, weights):
    """Batched multi-head wrapper around :func:`deform_attend`.

    value (B, H, W, M, D); points (B, N, M, K, 2); weights (B, N, M, K).
    Returns (B, N, M, D). A double Python loop is fine at lab scale.
    """
    B, N, M, K, _ = points.shape
    outs = [
        deform_attend(value[b, :, :, m], points[b, :, m], weights[b, :, m])
        for b in range(B)
        for m in range(M)
    ]
    return torch.stack(outs).reshape(B, M, N, -1).transpose(1, 2)


class DeformableAttention(nn.Module):
    """Single-scale deformable *self*-attention over a feature map.

    Input and output are (B, H, W, C) with C = num_heads * head_dim. Every
    position is a query: from its feature vector it predicts, per head, K
    sampling offsets (in pixels, relative to its own location) and K
    attention weights (softmax over K), then blends bilinear reads of the
    projected value map. No query-key dot products anywhere.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_points: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_points)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        M, K, D = self.num_heads, self.num_points, self.head_dim
        value = self.value_proj(x).reshape(B, H, W, M, D)
        offsets = self.offset_proj(x).reshape(B, H * W, M, K, 2)
        weights = self.weight_proj(x).reshape(B, H * W, M, K).softmax(dim=-1)
        # Reference points: position (row i, col j) sits at (x=j, y=i).
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=x.dtype, device=x.device),
            torch.arange(W, dtype=x.dtype, device=x.device),
            indexing="ij",
        )
        ref = torch.stack([xs, ys], dim=-1).reshape(H * W, 2)
        points = ref[None, :, None, None, :] + offsets
        out = _attend_heads(value, points, weights)
        return self.out_proj(out.reshape(B, H, W, C))


class DeformableCrossAttention(nn.Module):
    """Deformable cross-attention: object queries sample a feature map.

    ``query`` (B, Q, C) predicts offsets and weights; ``reference_points``
    (B, Q, 2) are normalized to [0, 1] and map to pixel coords as
    x_pix = x * W - 0.5 (grid_sample's align_corners=False convention);
    ``value`` is the feature map (B, H, W, C). Returns (B, Q, C).
    """

    def __init__(self, embed_dim: int, num_heads: int, num_points: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_points)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points, value):
        B, Q, C = query.shape
        _, H, W, _ = value.shape
        M, K, D = self.num_heads, self.num_points, self.head_dim
        v = self.value_proj(value).reshape(B, H, W, M, D)
        offsets = self.offset_proj(query).reshape(B, Q, M, K, 2)
        weights = self.weight_proj(query).reshape(B, Q, M, K).softmax(dim=-1)
        scale = query.new_tensor([W, H])
        ref_pix = reference_points * scale - 0.5
        points = ref_pix[:, :, None, None, :] + offsets
        out = _attend_heads(v, points, weights)
        return self.out_proj(out.reshape(B, Q, C))


def msda(value, spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention via per-level ``F.grid_sample`` — the
    standard PyTorch fallback (same math as mmcv's
    ``multi_scale_deformable_attn_pytorch``).

    value (B, S, M, D) with the L level images flattened along S;
    spatial_shapes (L, 2) of (H_l, W_l); sampling_locations
    (B, Q, M, L, K, 2) normalized to [0, 1] (offsets already added);
    attention_weights (B, Q, M, L, K) softmax-normalized over (L, K).
    Returns (B, Q, M * D).
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
            F.grid_sample(
                v, g, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        )  # (B*M, D, Q, K)
    # L x (B*M, D, Q, K) -> (B*M, D, Q, L*K)
    sampled = torch.stack(sampled, dim=-2).flatten(-2)
    attn = attention_weights.transpose(1, 2).reshape(B * M, Q, L * K).unsqueeze(1)
    out = (sampled * attn).sum(-1)  # (B*M, D, Q)
    return out.view(B, M * D, Q).transpose(1, 2).contiguous()


class MSDACrossAttention(nn.Module):
    """The full Deformable DETR decoder cross-attention layer.

    query (B, Q, C); reference_points (B, Q, 2) normalized to [0, 1];
    value (B, S, C), the feature pyramid flattened level by level along S;
    spatial_shapes (L, 2). Offsets are predicted per (head, level, point)
    in units of each level's pixels and normalized by (W_l, H_l) before
    being added to the reference point; attention weights are softmaxed
    over all L*K sampling points jointly. Returns (B, Q, C).
    """

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int,
                 num_points: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        M, L, K = num_heads, num_levels, num_points
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.offset_proj = nn.Linear(embed_dim, M * L * K * 2)
        self.weight_proj = nn.Linear(embed_dim, M * L * K)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points, value, spatial_shapes):
        B, Q, C = query.shape
        _, S, _ = value.shape
        M, L, K = self.num_heads, self.num_levels, self.num_points
        D = self.head_dim
        v = self.value_proj(value).reshape(B, S, M, D)
        offsets = self.offset_proj(query).reshape(B, Q, M, L, K, 2)
        weights = self.weight_proj(query).reshape(B, Q, M, L * K)
        weights = weights.softmax(dim=-1).reshape(B, Q, M, L, K)
        # Offsets are in each level's pixel units -> normalize by (W_l, H_l)
        # so they can be added to the [0, 1] reference point.
        scale = spatial_shapes.flip(-1).to(query.dtype)  # (L, 2) as (W_l, H_l)
        locs = (reference_points[:, :, None, None, None, :]
                + offsets / scale[None, None, None, :, None, :])
        return self.out_proj(msda(v, spatial_shapes, locs, weights))
