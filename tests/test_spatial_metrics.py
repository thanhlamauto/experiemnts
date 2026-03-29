from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from sit_metrics.geometry import (
    correlogram_decay_score,
    local_global_concentration_ratio,
    locality_distance_score,
    multi_scale_detail_retention,
    rms_spatial_contrast,
    spatial_metric_bundle,
    token_graph_spectral_gap,
    token_pairwise_manhattan_dist,
    unsupervised_boundary_concentration,
)
from sit_metrics.noising import (
    build_noise_bank,
    canonical_noise_level_to_model_t,
    canonical_noise_level_to_xt,
)


def _structured_hidden(p: int = 4, q: int = 4, batch_size: int = 2) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, p),
        torch.linspace(-1.0, 1.0, q),
        indexing="ij",
    )
    features = torch.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            (xx + yy).reshape(-1),
            (xx - yy).reshape(-1),
        ],
        dim=-1,
    )
    return features.unsqueeze(0).repeat(batch_size, 1, 1)


def _checkerboard_hidden(p: int = 8, q: int = 8, batch_size: int = 2) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(p), torch.arange(q), indexing="ij")
    pattern = ((xx + yy) % 2).float().reshape(-1)
    features = torch.stack([pattern, 0.5 * pattern, pattern, 0.25 * pattern], dim=-1)
    return features.unsqueeze(0).repeat(batch_size, 1, 1)


def _smooth_hidden(p: int = 8, q: int = 8, batch_size: int = 2) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, p),
        torch.linspace(0.0, 1.0, q),
        indexing="ij",
    )
    values = (0.6 * xx + 0.4 * yy).reshape(-1)
    features = torch.stack([values, 0.5 * values, values, 0.25 * values], dim=-1)
    return features.unsqueeze(0).repeat(batch_size, 1, 1)


def _clustered_hidden(p: int = 4, q: int = 4, batch_size: int = 2) -> torch.Tensor:
    _yy, xx = torch.meshgrid(torch.arange(p), torch.arange(q), indexing="ij")
    left = (xx < (q // 2)).reshape(-1).float()
    right = 1.0 - left
    features = torch.stack([left, right], dim=-1)
    return features.unsqueeze(0).repeat(batch_size, 1, 1)


def test_lds_and_cds_are_higher_for_spatially_structured_tokens() -> None:
    H = _structured_hidden()
    dist = token_pairwise_manhattan_dist(4, 4, device=H.device, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(123)
    perm = torch.randperm(H.shape[1], generator=generator)
    H_shuffled = H[:, perm]

    lds_structured = float(locality_distance_score(H, dist, r_near=2.0, r_far=4.0))
    cds_structured = float(correlogram_decay_score(H, dist))
    lds_shuffled = float(locality_distance_score(H_shuffled, dist, r_near=2.0, r_far=4.0))
    cds_shuffled = float(correlogram_decay_score(H_shuffled, dist))

    assert lds_structured > lds_shuffled
    assert cds_structured > cds_shuffled


def test_lgr_is_higher_for_spatially_structured_tokens() -> None:
    H = _structured_hidden()
    dist = token_pairwise_manhattan_dist(4, 4, device=H.device, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(321)
    perm = torch.randperm(H.shape[1], generator=generator)
    H_shuffled = H[:, perm]

    lgr_structured = float(local_global_concentration_ratio(H, dist, r_near=2.0, r_far=4.0, tau=10.0))
    lgr_shuffled = float(local_global_concentration_ratio(H_shuffled, dist, r_near=2.0, r_far=4.0, tau=10.0))

    assert lgr_structured > lgr_shuffled


def test_rmsc_is_near_zero_for_constant_tokens_and_positive_for_varied_tokens() -> None:
    constant = torch.ones(2, 16, 8)
    varied = _structured_hidden(batch_size=2)

    assert float(rms_spatial_contrast(constant)) == pytest.approx(0.0, abs=1e-6)
    assert float(rms_spatial_contrast(varied)) > 0.0


def test_msdr_and_ubc_are_higher_for_checkerboard_than_smooth_maps() -> None:
    checker = _checkerboard_hidden()
    smooth = _smooth_hidden()

    msdr_checker = float(multi_scale_detail_retention(checker, p=8, q=8, sigmas=(1.0, 2.0)))
    msdr_smooth = float(multi_scale_detail_retention(smooth, p=8, q=8, sigmas=(1.0, 2.0)))
    ubc_checker = float(unsupervised_boundary_concentration(checker, p=8, q=8, topk_ratio=0.1))
    ubc_smooth = float(unsupervised_boundary_concentration(smooth, p=8, q=8, topk_ratio=0.1))

    assert msdr_checker > msdr_smooth
    assert ubc_checker > ubc_smooth


def test_graph_gap_is_higher_for_clustered_tokens_than_random_tokens() -> None:
    clustered = _clustered_hidden()
    generator = torch.Generator()
    generator.manual_seed(7)
    random_tokens = torch.randn(clustered.shape, generator=generator)

    clustered_gap = float(token_graph_spectral_gap(clustered, k=4))
    random_gap = float(token_graph_spectral_gap(random_tokens, k=4))

    assert clustered_gap > random_gap


def test_spatial_metric_bundle_returns_requested_extended_metrics() -> None:
    H = _structured_hidden(p=4, q=4)
    dist = token_pairwise_manhattan_dist(4, 4, device=H.device, dtype=torch.float32)

    bundle = spatial_metric_bundle(
        H,
        dist,
        r_near=2.0,
        r_far=4.0,
        metrics=("lds", "cds", "rmsc", "lgr", "msdr", "graph_gap", "ubc"),
        p=4,
        q=4,
        msdr_sigmas=(1.0, 2.0),
        graph_knn_k=4,
    )

    assert set(bundle.keys()) == {"lds", "cds", "rmsc", "lgr", "msdr", "graph_gap", "ubc"}
    for value in bundle.values():
        assert torch.isfinite(value)


def test_linear_noise_mapping_and_xt_construction() -> None:
    assert canonical_noise_level_to_model_t("sit", 0.75, path_type="linear") == pytest.approx(0.25)
    assert canonical_noise_level_to_model_t("repa", 0.75, path_type="linear") == pytest.approx(0.75)

    x_clean = torch.ones(2, 1, 2, 2)
    noise = torch.full_like(x_clean, 3.0)

    xt_clean = canonical_noise_level_to_xt(x_clean, noise, 0.0, path_type="linear")
    xt_noise = canonical_noise_level_to_xt(x_clean, noise, 1.0, path_type="linear")
    xt_mid = canonical_noise_level_to_xt(x_clean, noise, 0.25, path_type="linear")

    assert torch.allclose(xt_clean, x_clean)
    assert torch.allclose(xt_noise, noise)
    assert torch.allclose(xt_mid, torch.full_like(x_clean, 1.5))


def test_noise_bank_is_deterministic() -> None:
    bank_a = build_noise_bank(5, (4, 2, 2), seed=17)
    bank_b = build_noise_bank(5, (4, 2, 2), seed=17)
    bank_c = build_noise_bank(5, (4, 2, 2), seed=18)

    assert torch.allclose(bank_a, bank_b)
    assert not torch.allclose(bank_a, bank_c)
