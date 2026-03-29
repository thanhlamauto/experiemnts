"""Per-layer class-geometry metrics on pooled hidden features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ClassGeometrySummary:
    means: torch.Tensor
    counts: torch.Tensor
    global_mean: torch.Tensor
    nc1: float
    etf_dev: float
    participation_ratio: float
    effective_rank: float


class ClassGeometryAccumulator:
    """Streaming accumulator for Neural Collapse style class geometry metrics."""

    def __init__(self, num_classes: int, feature_dim: int, eps: float = 1e-12) -> None:
        if int(num_classes) <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)

        self.sum_by_class = torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float64)
        self.sum_sqnorm_by_class = torch.zeros(self.num_classes, dtype=torch.float64)
        self.counts = torch.zeros(self.num_classes, dtype=torch.float64)
        self.global_sum = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.sum_zzT = torch.zeros(self.feature_dim, self.feature_dim, dtype=torch.float64)

    @property
    def num_samples(self) -> int:
        return int(self.counts.sum().item())

    def update(self, Z: torch.Tensor, y: torch.Tensor) -> None:
        """Update from one pooled-feature batch: Z [B, D], y [B]."""
        if Z.ndim != 2:
            raise ValueError(f"Z must be rank-2 [B, D], got shape {tuple(Z.shape)}")
        if y.ndim != 1:
            raise ValueError(f"y must be rank-1 [B], got shape {tuple(y.shape)}")
        if Z.shape[0] != y.shape[0]:
            raise ValueError(f"Z and y batch sizes differ: {Z.shape[0]} vs {y.shape[0]}")
        if Z.shape[1] != self.feature_dim:
            raise ValueError(f"feature dim mismatch: expected {self.feature_dim}, got {Z.shape[1]}")

        Z_cpu = torch.as_tensor(Z, dtype=torch.float64).cpu()
        y_cpu = torch.as_tensor(y, dtype=torch.long).cpu()
        if bool((y_cpu < 0).any()) or bool((y_cpu >= self.num_classes).any()):
            bad = y_cpu[(y_cpu < 0) | (y_cpu >= self.num_classes)][0].item()
            raise ValueError(f"label {bad} is out of range [0, {self.num_classes})")

        self.sum_by_class.index_add_(0, y_cpu, Z_cpu)
        self.sum_sqnorm_by_class.index_add_(0, y_cpu, Z_cpu.square().sum(dim=1))
        self.counts.index_add_(0, y_cpu, torch.ones_like(y_cpu, dtype=torch.float64))
        self.global_sum += Z_cpu.sum(dim=0)
        self.sum_zzT += Z_cpu.T @ Z_cpu

    def summary(self) -> ClassGeometrySummary:
        n_total = self.num_samples
        if n_total <= 0:
            raise RuntimeError("cannot summarize an empty accumulator")

        active = self.counts > 0
        counts = self.counts[active]
        if counts.numel() <= 1:
            raise RuntimeError("need at least two active classes to compute class-geometry metrics")

        means = self.sum_by_class[active] / counts.unsqueeze(1)
        global_mean = self.global_sum / float(n_total)

        sw_total = self.sum_sqnorm_by_class[active].sum() - (counts.unsqueeze(1) * means.square()).sum()
        sw_total = sw_total.clamp_min(0.0)
        sw_trace = sw_total / float(n_total)

        centered_means = means - global_mean.unsqueeze(0)
        sb_trace = centered_means.square().sum(dim=1).mean()
        nc1 = float((sw_trace / sb_trace.clamp_min(self.eps)).item())

        centered_unit = centered_means / centered_means.norm(dim=1, keepdim=True).clamp_min(self.eps)
        gram = centered_unit @ centered_unit.T
        num_active_classes = int(counts.numel())
        off_diag = -1.0 / float(num_active_classes - 1)
        gram_etf = torch.full_like(gram, off_diag)
        gram_etf.fill_diagonal_(1.0)
        etf_dev = float(torch.linalg.matrix_norm(gram - gram_etf, ord="fro").item())

        cov = self.sum_zzT / float(n_total) - torch.outer(global_mean, global_mean)
        cov = 0.5 * (cov + cov.T)
        evals = torch.linalg.eigvalsh(cov).clamp_min(self.eps)

        eval_sum = evals.sum().clamp_min(self.eps)
        participation_ratio = float((eval_sum.square() / evals.square().sum().clamp_min(self.eps)).item())

        probs = evals / eval_sum
        entropy = -(probs * torch.log(probs)).sum()
        effective_rank = float(torch.exp(entropy).item())

        return ClassGeometrySummary(
            means=means,
            counts=counts,
            global_mean=global_mean,
            nc1=nc1,
            etf_dev=etf_dev,
            participation_ratio=participation_ratio,
            effective_rank=effective_rank,
        )


@torch.no_grad()
def ncm_predictions(
    Z: torch.Tensor,
    means: torch.Tensor,
    *,
    metric: str = "cosine",
) -> torch.Tensor:
    """Nearest class mean predictions for pooled features."""
    Z_cpu = torch.as_tensor(Z, dtype=torch.float32).cpu()
    means_cpu = torch.as_tensor(means, dtype=torch.float32).cpu()

    if metric == "cosine":
        z_norm = F.normalize(Z_cpu, dim=-1)
        means_norm = F.normalize(means_cpu, dim=-1)
        return (z_norm @ means_norm.T).argmax(dim=1)
    if metric == "l2":
        dists = (
            Z_cpu.square().sum(dim=1, keepdim=True)
            - 2.0 * (Z_cpu @ means_cpu.T)
            + means_cpu.square().sum(dim=1).unsqueeze(0)
        )
        return dists.argmin(dim=1)
    raise ValueError(f"unsupported NCM metric: {metric}")


@torch.no_grad()
def ncm_accuracy(
    Z: torch.Tensor,
    y: torch.Tensor,
    means: torch.Tensor,
    *,
    metric: str = "cosine",
) -> float:
    pred = ncm_predictions(Z, means, metric=metric)
    y_cpu = torch.as_tensor(y, dtype=torch.long).cpu()
    return float((pred == y_cpu).float().mean().item())
