from __future__ import annotations

import torch
import torch.nn.functional as F


def l2_normalize_tokens(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = tensor.norm(dim=-1, keepdim=True)
    return tensor / (denom + eps)


def mean_common(raw_normalized: torch.Tensor) -> torch.Tensor:
    return raw_normalized.mean(dim=0)


def mean_residual(raw_normalized: torch.Tensor, common_tokens: torch.Tensor) -> torch.Tensor:
    return raw_normalized - common_tokens.unsqueeze(0)


def tsvd_basis_v64(raw_normalized: torch.Tensor, rank: int = 64) -> torch.Tensor:
    flat = raw_normalized.reshape(-1, raw_normalized.shape[-1]).float().cpu()
    gram = flat.T @ flat
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)[:rank]
    basis = eigvecs[:, order]
    basis = torch.linalg.qr(basis, mode="reduced")[0]
    return basis


def project_to_basis(raw_normalized: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    basis_rank = basis[:, :rank].to(device=raw_normalized.device, dtype=raw_normalized.dtype)
    coeffs = torch.matmul(raw_normalized, basis_rank)
    return torch.matmul(coeffs, basis_rank.transpose(0, 1))


def tsvd_residual(raw_normalized: torch.Tensor, common_tokens: torch.Tensor) -> torch.Tensor:
    return raw_normalized - common_tokens


def mean_pool_tokens(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with token dimension, got {tuple(tensor.shape)}")
    return tensor.mean(dim=-2)


def cosine_similarity_tokenwise(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_norm = F.normalize(x.float(), dim=-1, eps=eps)
    y_norm = F.normalize(y.float(), dim=-1, eps=eps)
    return (x_norm * y_norm).sum(dim=-1).mean()

