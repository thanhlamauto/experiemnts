"""Alignment metrics between two feature matrices (same n samples, rows aligned)."""

from __future__ import annotations

import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X: [n, d1], Y: [n, d2], centered per column.
    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = (X.T @ Y).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return num / den.clamp_min(1e-12)


def _row_centered_linear_kernel(X: torch.Tensor) -> torch.Tensor:
    K = X @ X.T
    return K - K.mean(dim=1, keepdim=True)


def _knn_mask(X: torch.Tensor, k: int) -> torch.Tensor:
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to build k-NN neighborhoods.")
    k = max(1, min(int(k), n - 1))
    sims = X @ X.T
    sims.fill_diagonal_(float("-inf"))
    idx = sims.topk(k, dim=1).indices
    mask = torch.zeros((n, n), device=X.device, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def centered_kernel_nearest_neighbor_alignment(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    CKNNA: CKA-style normalization with the relaxed Align term from the paper.

    We use linear kernels and the neighborhood indicator from Eq. (24),
    which keeps only pairs that are k-nearest-neighbors in both spaces.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Expected same number of samples, got {X.shape[0]} and {Y.shape[0]}")

    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    Kc = _row_centered_linear_kernel(X)
    Lc = _row_centered_linear_kernel(Y)

    mask_x = _knn_mask(X, k)
    mask_y = _knn_mask(Y, k)
    alpha_xy = (mask_x & mask_y).to(Kc.dtype)
    alpha_xx = mask_x.to(Kc.dtype)
    alpha_yy = mask_y.to(Kc.dtype)

    norm = float((X.shape[0] - 1) ** 2)
    align_xy = (Kc * Lc * alpha_xy).sum() / norm
    align_xx = (Kc * Kc * alpha_xx).sum() / norm
    align_yy = (Lc * Lc * alpha_yy).sum() / norm
    den = align_xx.clamp_min(1e-12).sqrt() * align_yy.clamp_min(1e-12).sqrt()
    return align_xy / den
