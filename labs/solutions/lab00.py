"""Lab 0 solution: deformable attention, deformable cross-attention, and
multi-scale deformable attention (MSDA) as plain PyTorch layers."""

import torch
from torch import nn


def bilinear_sample(img: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample ``img`` (H, W, D) at continuous pixel coords ``points`` (N, 2).

    ``points[:, 0]`` is x, ``points[:, 1]`` is y. Vectorized over N and
    differentiable w.r.t. both ``img`` and ``points``; corners outside the
    image contribute zero ("zeros padding"). Returns (N, D).
    """
    H, W, D = img.shape
    x, y = points[:, 0], points[:, 1]
    x0f, y0f = x.floor(), y.floor()
    lx, ly = x - x0f, y - y0f
    x0, y0 = x0f.long(), y0f.long()
    flat = img.reshape(H * W, D)
    out = img.new_zeros(points.shape[0], D)
    corners = (
        (y0,     x0,     (1 - lx) * (1 - ly)),
        (y0,     x0 + 1, lx * (1 - ly)),
        (y0 + 1, x0,     (1 - lx) * ly),
        (y0 + 1, x0 + 1, lx * ly),
    )
    for yy, xx, w in corners:
        valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        idx = yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)
        out = out + (w * valid).unsqueeze(-1) * flat[idx]
    return out


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
    """Multi-scale deformable attention — the repo's functional contract.

    value (B, S, M, D) with the L level images flattened along S;
    spatial_shapes (L, 2) of (H_l, W_l); sampling_locations
    (B, Q, M, L, K, 2) normalized to [0, 1] (offsets already added);
    attention_weights (B, Q, M, L, K) softmax-normalized over (L, K).
    Returns (B, Q, M * D).
    """
    B, S, M, D = value.shape
    _, Q, _, L, K, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes]
    starts = [0]
    for h, w in shapes[:-1]:
        starts.append(starts[-1] + h * w)

    outs = []
    for b in range(B):
        for m in range(M):
            acc = None
            for lvl, (h, w) in enumerate(shapes):
                img = value[b, starts[lvl]:starts[lvl] + h * w, m].reshape(h, w, D)
                scale = sampling_locations.new_tensor([w, h])
                pts = sampling_locations[b, :, m, lvl] * scale - 0.5
                lvl_out = deform_attend(img, pts, attention_weights[b, :, m, lvl])
                acc = lvl_out if acc is None else acc + lvl_out
            outs.append(acc)
    out = torch.stack(outs).reshape(B, M, Q, D).transpose(1, 2)
    return out.reshape(B, Q, M * D)


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
