from __future__ import annotations

import math

import pytest
torch = pytest.importorskip("torch")

from sit_metrics.class_geometry import ClassGeometryAccumulator, ncm_accuracy


def _summary(Z: torch.Tensor, y: torch.Tensor, num_classes: int) -> object:
    acc = ClassGeometryAccumulator(num_classes=num_classes, feature_dim=Z.shape[1])
    acc.update(Z, y)
    return acc.summary()


def test_perfect_class_collapse_has_zero_nc1_and_perfect_ncm() -> None:
    Z = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1])

    summary = _summary(Z, y, num_classes=2)

    assert summary.nc1 == pytest.approx(0.0, abs=1e-8)
    assert ncm_accuracy(Z, y, summary.means, metric="cosine") == pytest.approx(1.0, abs=1e-8)


def test_simplex_means_have_lower_etf_deviation_than_skewed_means() -> None:
    simplex = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ]
    )
    skewed = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.15, 0.0],
            [0.15, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2])

    simplex_summary = _summary(simplex, y, num_classes=3)
    skewed_summary = _summary(skewed, y, num_classes=3)

    assert simplex_summary.etf_dev < skewed_summary.etf_dev


def test_participation_ratio_and_effective_rank_match_rank_one_and_isotropic_cases() -> None:
    rank_one = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    isotropic = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])

    rank_one_summary = _summary(rank_one, y, num_classes=2)
    isotropic_summary = _summary(isotropic, y, num_classes=2)

    assert rank_one_summary.participation_ratio == pytest.approx(1.0, abs=1e-6)
    assert rank_one_summary.effective_rank == pytest.approx(1.0, abs=1e-6)
    assert isotropic_summary.participation_ratio == pytest.approx(2.0, abs=1e-6)
    assert isotropic_summary.effective_rank == pytest.approx(2.0, abs=1e-6)
