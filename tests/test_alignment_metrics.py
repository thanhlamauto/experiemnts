from __future__ import annotations

import pytest
import torch

from sit_metrics.cka import centered_kernel_nearest_neighbor_alignment, linear_cka
from sit_metrics.reference_encoders import dinov2_input_resolution


def test_linear_cka_self_alignment_is_one() -> None:
    x = torch.randn(32, 16)
    score = linear_cka(x, x)
    assert score.item() == pytest.approx(1.0, abs=1e-5)


def test_cknna_self_alignment_is_one() -> None:
    x = torch.randn(32, 16)
    score = centered_kernel_nearest_neighbor_alignment(x, x, k=5)
    assert score.item() == pytest.approx(1.0, abs=1e-5)


def test_cknna_is_symmetric() -> None:
    x = torch.randn(24, 12)
    y = torch.randn(24, 10)
    score_xy = centered_kernel_nearest_neighbor_alignment(x, y, k=4)
    score_yx = centered_kernel_nearest_neighbor_alignment(y, x, k=4)
    assert score_xy.item() == pytest.approx(score_yx.item(), abs=1e-6)


def test_dinov2_input_resolution_matches_repa_convention() -> None:
    assert dinov2_input_resolution(256) == 224
    assert dinov2_input_resolution(512) == 448
