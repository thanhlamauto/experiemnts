from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    numerator = (x.T @ y).pow(2).sum()
    denominator = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return numerator / denominator.clamp_min(1e-12)


def token_mean_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_norm = F.normalize(x.float(), dim=-1, eps=eps)
    y_norm = F.normalize(y.float(), dim=-1, eps=eps)
    return (x_norm * y_norm).sum(dim=-1).mean()


def flat_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_flat = x.float().reshape(-1)
    y_flat = y.float().reshape(-1)
    denom = x_flat.norm() * y_flat.norm()
    return torch.dot(x_flat, y_flat) / denom.clamp_min(eps)


def tensor_stats(z: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    z_norm = z / (z.norm(dim=-1, keepdim=True) + eps)
    return {
        "min": float(z.min().item()),
        "max": float(z.max().item()),
        "mean": float(z_norm.mean().item()),
        "var": float(z_norm.var(unbiased=False).item()),
    }


def manhattan_distances(grid_size: int, device: torch.device | None = None) -> torch.Tensor:
    coords = torch.stack(
        [
            torch.arange(grid_size, device=device).repeat_interleave(grid_size),
            torch.arange(grid_size, device=device).repeat(grid_size),
        ],
        dim=-1,
    ).float()
    dx = (coords[:, 0:1] - coords[:, 0:1].T).abs()
    dy = (coords[:, 1:2] - coords[:, 1:2].T).abs()
    return dx + dy


def token_self_similarity(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z_norm = F.normalize(z.float(), dim=-1, eps=eps)
    return z_norm @ z_norm.T


def lds_metric(z: torch.Tensor, grid_size: int, radius: float | None = None) -> float:
    sim = token_self_similarity(z)
    dist = manhattan_distances(grid_size, device=sim.device)
    radius = float(radius if radius is not None else grid_size / 2.0)
    eye_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    near = sim[(dist < radius) & eye_mask]
    far = sim[(dist >= radius) & eye_mask]
    return float((near.mean() - far.mean()).item())


def cds_metric(z: torch.Tensor, grid_size: int) -> float:
    sim = token_self_similarity(z)
    dist = manhattan_distances(grid_size, device=sim.device).to(dtype=torch.int64)
    deltas = torch.arange(1, grid_size, device=sim.device, dtype=torch.int64)
    means = []
    valid_deltas = []
    for delta in deltas.tolist():
        mask = dist == int(delta)
        if mask.any():
            means.append(sim[mask].mean())
            valid_deltas.append(float(delta))
    if len(means) < 2:
        return 0.0
    xs = torch.tensor(valid_deltas, device=sim.device)
    ys = torch.stack(means).float()
    xs_centered = xs - xs.mean()
    ys_centered = ys - ys.mean()
    slope = (xs_centered * ys_centered).sum() / xs_centered.square().sum().clamp_min(1e-12)
    return float((-slope).item())


def rmsc_metric(z: torch.Tensor, eps: float = 1e-6) -> float:
    z_norm = F.normalize(z.float(), dim=-1, eps=eps)
    mean = z_norm.mean(dim=0, keepdim=True)
    value = (z_norm - mean).pow(2).sum(dim=-1).mean().sqrt()
    return float(value.item())


@dataclass
class WaveletEnergies:
    level: int
    approximation_energy: float
    detail_energy: float
    ratio: float


def wavelet_energies(z: torch.Tensor, grid_size: int, levels: Iterable[int] = (1, 2)) -> list[WaveletEnergies]:
    import pywt

    array = z.float().cpu().reshape(grid_size, grid_size, z.shape[-1]).numpy()
    out: list[WaveletEnergies] = []
    for level in levels:
        coeffs = pywt.wavedecn(array, wavelet="haar", level=int(level), axes=(0, 1), mode="periodization")
        approx = coeffs[0]
        details = coeffs[1]
        approx_energy = float(np.square(approx).sum())
        detail_energy = float(sum(np.square(value).sum() for value in details.values()))
        ratio = approx_energy / max(detail_energy, 1e-12)
        out.append(
            WaveletEnergies(
                level=int(level),
                approximation_energy=approx_energy,
                detail_energy=detail_energy,
                ratio=ratio,
            )
        )
    return out


def anchor_cosine_map(z: torch.Tensor, anchor_index: int, eps: float = 1e-6) -> torch.Tensor:
    anchor = z[anchor_index : anchor_index + 1]
    z_norm = F.normalize(z.float(), dim=-1, eps=eps)
    anchor_norm = F.normalize(anchor.float(), dim=-1, eps=eps)
    return torch.matmul(z_norm, anchor_norm.transpose(0, 1)).squeeze(-1)
