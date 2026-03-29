"""High-frequency energy ratio from per-token L2 norm activation maps."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def high_frequency_ratio(
    H: torch.Tensor,
    p: int,
    q: int,
    high_freq_radius_frac: float = 0.5,
) -> torch.Tensor:
    """
    H: [B, N, D] with N = p*q row-major grid.
    m_{b,i} = ||H_{b,i}||_2, reshape (B, p, q).
    HF ratio = energy in Fourier shell with radius > high_freq_radius_frac * max_radius.
    """
    B, N, _D = H.shape
    assert N == p * q
    m = H.float().norm(dim=-1).reshape(B, p, q)
    Fmap = torch.fft.fft2(m)
    power = Fmap.real.pow(2) + Fmap.imag.pow(2)

    # frequency grid
    fy = torch.fft.fftfreq(p, device=H.device, dtype=m.dtype).view(-1, 1)
    fx = torch.fft.fftfreq(q, device=H.device, dtype=m.dtype).view(1, -1)
    r = (fx**2 + fy**2).sqrt()
    r_max = torch.tensor([1.0 / p, 1.0 / q], device=H.device).norm() * math.sqrt(2)  # approx nyquist
    high = r > (high_freq_radius_frac * r_max)

    num = (power * high.view(1, p, q).float()).sum(dim=(-1, -2))
    den = power.sum(dim=(-1, -2)).clamp_min(1e-12)
    return (num / den).mean()
