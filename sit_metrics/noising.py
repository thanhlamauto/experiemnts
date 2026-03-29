"""Canonical noise-level helpers for comparing SiT and REPA on the same x_t inputs."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch


def parse_noise_levels(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def validate_noise_level(noise_level: float) -> float:
    value = float(noise_level)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {value}")
    return value


def normalize_path_type(path_type: str) -> str:
    return path_type.strip().lower()


def canonical_noise_level_to_model_t(
    backend: str,
    noise_level: float,
    path_type: str = "linear",
) -> float:
    backend = backend.strip().lower()
    path = normalize_path_type(path_type)
    nu = validate_noise_level(noise_level)

    if backend == "sit":
        if path == "linear":
            return 1.0 - nu
        if path == "gvp":
            return 2.0 * math.acos(nu) / math.pi
        raise NotImplementedError(f"Unsupported SiT path_type={path_type!r} for canonical noise mapping")

    if backend == "repa":
        if path == "linear":
            return nu
        if path == "cosine":
            return 2.0 * math.asin(nu) / math.pi
        raise NotImplementedError(f"Unsupported REPA path_type={path_type!r} for canonical noise mapping")

    raise ValueError(f"Unsupported backend={backend!r}")


def canonical_noise_level_to_xt(
    x_clean: torch.Tensor,
    noise: torch.Tensor,
    noise_level: float,
    path_type: str = "linear",
) -> torch.Tensor:
    path = normalize_path_type(path_type)
    nu = torch.as_tensor(validate_noise_level(noise_level), dtype=x_clean.dtype, device=x_clean.device)

    if path == "linear":
        alpha = 1.0 - nu
        sigma = nu
    elif path in {"cosine", "gvp"}:
        sigma = nu
        alpha = torch.sqrt((1.0 - sigma.square()).clamp_min(0.0))
    else:
        raise NotImplementedError(f"Unsupported path_type={path_type!r} for x_t construction")

    return alpha * x_clean + sigma * noise


def build_noise_bank(
    num_samples: int,
    sample_shape: Iterable[int],
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((num_samples, *tuple(sample_shape)), generator=generator, dtype=dtype)
