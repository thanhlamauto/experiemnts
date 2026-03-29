"""Token distances, attention metrics, and spatial hidden-state geometry metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def token_pairwise_manhattan_dist(p: int, q: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    d_ij = |x_i - x_j| + |y_i - y_j| on the p x q grid, flattened row-major.
    Returns [N, N] with N = p*q.
    """
    N = p * q
    coords = torch.stack(
        [
            torch.arange(p, device=device, dtype=dtype).repeat_interleave(q),
            torch.arange(q, device=device, dtype=dtype).repeat(p),
        ],
        dim=-1,
    )  # [N, 2]
    dx = (coords[:, 0:1] - coords[:, 0:1].T).abs()
    dy = (coords[:, 1:2] - coords[:, 1:2].T).abs()
    return dx + dy


def off_diagonal_mask(N: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(N, dtype=torch.bool, device=device)


@torch.no_grad()
def mean_attention_distance(
    A: torch.Tensor, dist: torch.Tensor
) -> torch.Tensor:
    """
    A: [B, h, N, N] attention probs (row = query i).
    dist: [N, N] pairwise distances.
    Returns scalar: mean over batch, heads, queries of sum_j A_ij * d_ij.
    """
    # sum_j A_ij d_ij for each i -> [B, h, N]
    d = dist.view(1, 1, dist.shape[0], dist.shape[1]).to(dtype=A.dtype, device=A.device)
    mad = (A * d).sum(dim=-1).mean()
    return mad


@torch.no_grad()
def attention_entropy_mean(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    A: [B, h, N, N] probabilities along last dim.
    Mean entropy per query position: -sum_j p log p, averaged over B,h,N queries.
    """
    ent = -(A * (A + eps).log()).sum(dim=-1)
    return ent.mean()


@torch.no_grad()
def similarity_distance_decay_slope(
    H: torch.Tensor,
    dist: torch.Tensor,
    num_bins: int = 16,
    max_delta: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H: [B, N, D] — one layer hidden states.
    dist: [N, N]
    Returns (mean_cosines_per_bin [num_bins], slope) where slope is OLS of mean cosine vs bin center delta.
    """
    B, N, _D = H.shape
    Hn = F.normalize(H.float(), dim=-1)
    sims = torch.matmul(Hn, Hn.transpose(-1, -2))  # [B, N, N]
    d_flat = dist.view(-1).float()
    s_flat = sims.reshape(B, -1).mean(dim=0)  # mean over batch

    if max_delta is None:
        max_delta = float(d_flat.max().item())
    edges = torch.linspace(0, max_delta, num_bins + 1, device=H.device, dtype=d_flat.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])

    means = []
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (d_flat >= lo) & (d_flat < hi) if i < num_bins - 1 else (d_flat >= lo) & (d_flat <= hi)
        if mask.any():
            means.append(s_flat[mask].mean())
        else:
            means.append(torch.tensor(float("nan"), device=H.device))
    means_t = torch.stack(means)
    valid = ~torch.isnan(means_t)
    if valid.sum() < 2:
        return means_t, torch.tensor(0.0, device=H.device)
    c = centers[valid]
    m = means_t[valid]
    # slope dm/ddelta
    c_mean = c.mean()
    m_mean = m.mean()
    slope = ((c - c_mean) * (m - m_mean)).sum() / ((c - c_mean) ** 2).sum().clamp_min(1e-12)
    return means_t, slope


def _normalized_tokens(H: torch.Tensor) -> torch.Tensor:
    return F.normalize(H.float(), dim=-1)


def _normalized_tokens_and_similarity(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    Hn = _normalized_tokens(H)
    sims = torch.matmul(Hn, Hn.transpose(-1, -2))
    return Hn, sims


def _lds_from_precomputed(
    sims: torch.Tensor,
    dist: torch.Tensor,
    r_near: float,
    r_far: float,
) -> torch.Tensor:
    N = sims.shape[-1]
    diag_mask = off_diagonal_mask(N, sims.device)
    dist_device = dist.to(device=sims.device)
    near_mask = (dist_device < r_near) & diag_mask
    far_mask = (dist_device >= r_far) & diag_mask
    if not near_mask.any():
        raise ValueError(f"No token pairs found with d < {r_near}")
    if not far_mask.any():
        raise ValueError(f"No token pairs found with d >= {r_far}")
    near_mean = sims[:, near_mask].mean(dim=-1)
    far_mean = sims[:, far_mask].mean(dim=-1)
    return (near_mean - far_mean).mean()


def _lgr_from_precomputed(
    sims: torch.Tensor,
    dist: torch.Tensor,
    r_near: float,
    r_far: float,
    tau: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    N = sims.shape[-1]
    diag_mask = off_diagonal_mask(N, sims.device)
    dist_device = dist.to(device=sims.device)
    near_mask = ((dist_device < r_near) & diag_mask).to(dtype=sims.dtype)
    far_mask = ((dist_device >= r_far) & diag_mask).to(dtype=sims.dtype)
    if not near_mask.any():
        raise ValueError(f"No token pairs found with d < {r_near}")
    if not far_mask.any():
        raise ValueError(f"No token pairs found with d >= {r_far}")
    energy = torch.exp(tau * sims)
    near_energy = (energy * near_mask).sum(dim=-1)
    far_energy = (energy * far_mask).sum(dim=-1)
    return (near_energy / far_energy.clamp_min(eps)).mean()


def _cds_from_precomputed(
    sims: torch.Tensor,
    dist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist_int = dist.to(device=sims.device, dtype=torch.int64)
    deltas = torch.unique(dist_int)
    deltas = deltas[deltas > 0]
    if deltas.numel() < 2:
        raise ValueError("Need at least two positive Manhattan distances to fit CDS")

    g_vals = []
    for delta in deltas.tolist():
        mask = dist_int == delta
        g_vals.append(sims[:, mask].mean(dim=-1).mean())
    g = torch.stack(g_vals)
    xs = deltas.to(dtype=torch.float32)
    x_mean = xs.mean()
    y_mean = g.mean()
    slope = ((xs - x_mean) * (g - y_mean)).sum() / ((xs - x_mean) ** 2).sum().clamp_min(1e-12)
    return xs, g, -slope


def _rmsc_from_normalized(Hn: torch.Tensor) -> torch.Tensor:
    mean = Hn.mean(dim=1, keepdim=True)
    return (Hn - mean).pow(2).sum(dim=-1).mean(dim=1).sqrt().mean()


def _activation_norm_map(H: torch.Tensor, p: int, q: int) -> torch.Tensor:
    B, N, _D = H.shape
    if N != p * q:
        raise ValueError(f"Expected N={p*q} tokens for grid {p}x{q}, got {N}")
    return H.float().norm(dim=-1).reshape(B, 1, p, q)


def _gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = max(int(round(3.0 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def _blur_activation_map(m: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel2d(sigma, m.device, m.dtype)
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    return F.conv2d(m, kernel, padding=(pad_h, pad_w))


def _msdr_from_activation_map(
    m: torch.Tensor,
    sigmas: Sequence[float],
    weights: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not sigmas:
        raise ValueError("sigmas must be non-empty for MSDR")
    if weights is None:
        weights = [1.0] * len(sigmas)
    if len(weights) != len(sigmas):
        raise ValueError("weights and sigmas must have the same length")

    base_energy = m.square().sum(dim=(-1, -2, -3)).clamp_min(eps)
    score = torch.zeros(m.shape[0], device=m.device, dtype=m.dtype)
    for sigma, weight in zip(sigmas, weights):
        low = _blur_activation_map(m, sigma)
        high = m - low
        detail_energy = high.square().sum(dim=(-1, -2, -3))
        score = score + float(weight) * (detail_energy / base_energy)
    return score.mean()


def _ubc_from_activation_map(
    m: torch.Tensor,
    topk_ratio: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=m.device,
        dtype=m.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=m.device,
        dtype=m.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(m, sobel_x, padding=1)
    gy = F.conv2d(m, sobel_y, padding=1)
    g = torch.sqrt(gx.square() + gy.square()).flatten(1)
    k = max(1, int(round(g.shape[1] * float(topk_ratio))))
    topk = torch.topk(g, k=k, dim=-1).values.sum(dim=-1)
    total = g.sum(dim=-1).clamp_min(eps)
    return (topk / total).mean()


def _knn_affinity_from_precomputed(
    sims: torch.Tensor,
    k: int,
) -> torch.Tensor:
    B, N, _ = sims.shape
    if N < 3:
        raise ValueError(f"Need at least 3 tokens for graph spectral gap, got N={N}")
    k_eff = max(1, min(int(k), N - 1))
    pos = torch.clamp(sims, min=0.0)
    eye = torch.eye(N, device=sims.device, dtype=torch.bool).unsqueeze(0)
    pos = pos.masked_fill(eye, float("-inf"))
    vals, idx = torch.topk(pos, k=k_eff, dim=-1)
    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
    W = torch.zeros_like(sims)
    W.scatter_(-1, idx, vals)
    W = torch.maximum(W, W.transpose(-1, -2))
    W = W.masked_fill(eye, 0.0)
    return W


def _graph_gap_from_precomputed(
    sims: torch.Tensor,
    k: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    W = _knn_affinity_from_precomputed(sims, k=k)
    deg = W.sum(dim=-1).clamp_min(eps)
    norm = deg.rsqrt().unsqueeze(-1) * W * deg.rsqrt().unsqueeze(-2)
    eye = torch.eye(W.shape[-1], device=W.device, dtype=norm.dtype).unsqueeze(0)
    lap = eye - norm
    evals = torch.linalg.eigvalsh(lap.float())
    return (evals[:, 2] - evals[:, 1]).mean()


@torch.no_grad()
def locality_distance_score(
    H: torch.Tensor,
    dist: torch.Tensor,
    r_near: float,
    r_far: float,
) -> torch.Tensor:
    _, sims = _normalized_tokens_and_similarity(H)
    return _lds_from_precomputed(sims, dist, r_near, r_far)


@torch.no_grad()
def correlogram_decay_score(
    H: torch.Tensor,
    dist: torch.Tensor,
) -> torch.Tensor:
    _, sims = _normalized_tokens_and_similarity(H)
    _, _, cds = _cds_from_precomputed(sims, dist)
    return cds


@torch.no_grad()
def correlogram_by_exact_distance(
    H: torch.Tensor,
    dist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, sims = _normalized_tokens_and_similarity(H)
    xs, g, _ = _cds_from_precomputed(sims, dist)
    return xs, g


@torch.no_grad()
def rms_spatial_contrast(H: torch.Tensor) -> torch.Tensor:
    Hn = _normalized_tokens(H)
    return _rmsc_from_normalized(Hn)


@torch.no_grad()
def local_global_concentration_ratio(
    H: torch.Tensor,
    dist: torch.Tensor,
    r_near: float,
    r_far: float,
    tau: float = 10.0,
) -> torch.Tensor:
    _, sims = _normalized_tokens_and_similarity(H)
    return _lgr_from_precomputed(sims, dist, r_near, r_far, tau=tau)


@torch.no_grad()
def multi_scale_detail_retention(
    H: torch.Tensor,
    p: int,
    q: int,
    sigmas: Sequence[float] = (1.0, 2.0, 4.0),
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    m = _activation_norm_map(H, p, q)
    return _msdr_from_activation_map(m, sigmas=sigmas, weights=weights)


@torch.no_grad()
def token_graph_spectral_gap(
    H: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    _, sims = _normalized_tokens_and_similarity(H)
    return _graph_gap_from_precomputed(sims, k=k)


@torch.no_grad()
def unsupervised_boundary_concentration(
    H: torch.Tensor,
    p: int,
    q: int,
    topk_ratio: float = 0.1,
) -> torch.Tensor:
    m = _activation_norm_map(H, p, q)
    return _ubc_from_activation_map(m, topk_ratio=topk_ratio)


@torch.no_grad()
def spatial_metric_bundle(
    H: torch.Tensor,
    dist: torch.Tensor,
    r_near: float,
    r_far: float,
    *,
    metrics: Iterable[str] | None = None,
    p: int | None = None,
    q: int | None = None,
    lgr_tau: float = 10.0,
    msdr_sigmas: Sequence[float] = (1.0, 2.0, 4.0),
    msdr_weights: Sequence[float] | None = None,
    graph_knn_k: int = 10,
    ubc_topk_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    requested = tuple(metrics) if metrics is not None else ("lds", "cds", "rmsc", "lgr", "msdr", "graph_gap", "ubc")
    out: Dict[str, torch.Tensor] = {}

    need_similarity = bool({"lds", "cds", "rmsc", "lgr", "graph_gap"} & set(requested))
    Hn: torch.Tensor | None = None
    sims: torch.Tensor | None = None
    if need_similarity:
        Hn, sims = _normalized_tokens_and_similarity(H)

    if "lds" in requested:
        assert sims is not None
        out["lds"] = _lds_from_precomputed(sims, dist, r_near, r_far)
    if "cds" in requested:
        assert sims is not None
        _, _, cds = _cds_from_precomputed(sims, dist)
        out["cds"] = cds
    if "rmsc" in requested:
        assert Hn is not None
        out["rmsc"] = _rmsc_from_normalized(Hn)
    if "lgr" in requested:
        assert sims is not None
        out["lgr"] = _lgr_from_precomputed(sims, dist, r_near, r_far, tau=lgr_tau)
    if "graph_gap" in requested:
        assert sims is not None
        out["graph_gap"] = _graph_gap_from_precomputed(sims, k=graph_knn_k)

    need_activation_map = bool({"msdr", "ubc"} & set(requested))
    if need_activation_map:
        if p is None or q is None:
            raise ValueError("p and q are required for activation-map metrics")
        m = _activation_norm_map(H, p, q)
        if "msdr" in requested:
            out["msdr"] = _msdr_from_activation_map(m, sigmas=msdr_sigmas, weights=msdr_weights)
        if "ubc" in requested:
            out["ubc"] = _ubc_from_activation_map(m, topk_ratio=ubc_topk_ratio)

    return out
