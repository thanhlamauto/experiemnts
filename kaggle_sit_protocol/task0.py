from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ProtocolConfig
from .decomposition import (
    l2_normalize_tokens,
    mean_common,
    mean_pool_tokens,
    mean_residual,
    project_to_basis,
    tsvd_basis_v64,
    tsvd_residual,
)
from .io_utils import estimate_bytes
from .modeling import compute_patch_tokens, forward_features, linear_path_xt, load_sit_model


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_raw_normalized(
    model: torch.nn.Module,
    x1: torch.Tensor,
    x0: torch.Tensor,
    y: int,
    time_values: tuple[float, ...],
    device: torch.device,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = x1.to(device=device, dtype=torch.float32)
    x0 = x0.to(device=device, dtype=torch.float32)
    timesteps = torch.tensor(time_values, device=device, dtype=torch.float32)
    xt = linear_path_xt(x1, x0, timesteps).to(device=device, dtype=torch.float32)
    labels = torch.full((len(time_values),), int(y), device=device, dtype=torch.long)
    _, blocks = forward_features(model, xt, timesteps, labels)
    raw_norm = l2_normalize_tokens(blocks, eps=eps)
    patch_tokens = compute_patch_tokens(model, x1.unsqueeze(0)).squeeze(0)
    return patch_tokens.detach().cpu().to(torch.float16), raw_norm.detach().cpu().to(torch.float16)


def _allocate_descriptor_cache(config: ProtocolConfig) -> dict[str, np.ndarray]:
    n_images = config.main_images_target
    l = config.num_layers
    t = len(config.time_grid_indices)
    d = config.hidden_dim
    cache: dict[str, np.ndarray] = {
        "raw": np.zeros((n_images, l, t, d), dtype=np.float16),
        "mean_common": np.zeros((n_images, t, d), dtype=np.float16),
        "mean_residual": np.zeros((n_images, l, t, d), dtype=np.float16),
    }
    for rank in config.tsvd_ranks:
        cache[f"tsvd_common_k{rank}"] = np.zeros((n_images, l, t, d), dtype=np.float16)
        cache[f"tsvd_residual_k{rank}"] = np.zeros((n_images, l, t, d), dtype=np.float16)
    return cache


def _preview_template(config: ProtocolConfig) -> dict[str, object]:
    preview_images = config.preview_images
    preview_layers = len(config.preview_layers_zeroindexed)
    preview_times = len(config.preview_timestep_positions)
    p = config.patch_grid_size * config.patch_grid_size
    d = config.hidden_dim
    payload: dict[str, object] = {
        "image_ids": [],
        "layer_indices_zero": torch.tensor(config.preview_layers_zeroindexed, dtype=torch.long),
        "time_positions": torch.tensor(config.preview_timestep_positions, dtype=torch.long),
        "patch_tokens_clean": torch.zeros((preview_images, p, d), dtype=torch.float16),
        "raw": torch.zeros((preview_images, preview_layers, preview_times, p, d), dtype=torch.float16),
        "mean_common": torch.zeros((preview_images, preview_times, p, d), dtype=torch.float16),
        "mean_residual": torch.zeros((preview_images, preview_layers, preview_times, p, d), dtype=torch.float16),
    }
    for rank in config.tsvd_ranks:
        payload[f"tsvd_common_k{rank}"] = torch.zeros(
            (preview_images, preview_layers, preview_times, p, d), dtype=torch.float16
        )
        payload[f"tsvd_residual_k{rank}"] = torch.zeros(
            (preview_images, preview_layers, preview_times, p, d), dtype=torch.float16
        )
    return payload


def _orthonormality_error(basis: torch.Tensor) -> float:
    eye = torch.eye(basis.shape[1], dtype=basis.dtype)
    diff = basis.T @ basis - eye
    return float(diff.abs().max().item())


def run_task0(config: ProtocolConfig, stage_dir: Path) -> dict[str, object]:
    device = _device_from_config(config)
    model = load_sit_model(config, device)
    manifest = pd.read_parquet(config.manifest_path)
    latents = torch.load(config.cache_dir / "x1_latents_fp16.pt", map_location="cpu")
    noise = torch.load(config.cache_dir / "x0_noise_seed0_fp16.pt", map_location="cpu")

    main_rows = manifest[manifest["subset_role"] == "main"].reset_index(drop=True)
    control_rows = manifest[manifest["subset_role"] == "control"].reset_index(drop=True)
    descriptors = _allocate_descriptor_cache(config)
    preview_cache = _preview_template(config)
    preview_lookup = {image_id: idx for idx, image_id in enumerate(main_rows[main_rows["preview"]]["image_id"].tolist())}

    main_bases = torch.zeros((len(main_rows), config.hidden_dim, 64), dtype=torch.float16)
    control_bases = torch.zeros((len(control_rows), config.hidden_dim, 64), dtype=torch.float16)

    metadata_rows: list[dict[str, object]] = []
    sanity_residuals: list[float] = []
    orth_errors: list[float] = []

    for idx, row in main_rows.iterrows():
        patch_tokens, raw_norm = _extract_raw_normalized(
            model,
            latents["main_latents"][idx],
            noise["main_noise"][idx],
            int(row["imagenet_idx"]),
            config.time_values,
            device,
            config.stats_eps,
        )
        raw_norm_f = raw_norm.float()
        mean_common_tokens = mean_common(raw_norm_f)
        mean_res = mean_residual(raw_norm_f, mean_common_tokens)
        basis = tsvd_basis_v64(raw_norm_f, rank=64)
        main_bases[idx] = basis.to(torch.float16)
        orth_errors.append(_orthonormality_error(basis))

        descriptors["raw"][idx] = mean_pool_tokens(raw_norm_f).numpy().astype(np.float16)
        descriptors["mean_common"][idx] = mean_pool_tokens(mean_common_tokens).numpy().astype(np.float16)
        descriptors["mean_residual"][idx] = mean_pool_tokens(mean_res).numpy().astype(np.float16)
        sanity_residuals.append(float(mean_res.abs().mean().item()))

        if row["preview"]:
            preview_index = preview_lookup[str(row["image_id"])]
            preview_cache["image_ids"].append(str(row["image_id"]))
            preview_cache["patch_tokens_clean"][preview_index] = patch_tokens
            layer_idx = list(config.preview_layers_zeroindexed)
            time_idx = list(config.preview_timestep_positions)
            preview_cache["raw"][preview_index] = raw_norm[layer_idx][:, time_idx]
            preview_cache["mean_common"][preview_index] = mean_common_tokens[time_idx].to(torch.float16)
            preview_cache["mean_residual"][preview_index] = mean_res[layer_idx][:, time_idx].to(torch.float16)

        for rank in config.tsvd_ranks:
            common_tokens = project_to_basis(raw_norm_f, basis, rank)
            residual_tokens = tsvd_residual(raw_norm_f, common_tokens)
            descriptors[f"tsvd_common_k{rank}"][idx] = mean_pool_tokens(common_tokens).numpy().astype(np.float16)
            descriptors[f"tsvd_residual_k{rank}"][idx] = mean_pool_tokens(residual_tokens).numpy().astype(np.float16)
            sanity_residuals.append(float(residual_tokens.abs().mean().item()))
            if row["preview"]:
                preview_index = preview_lookup[str(row["image_id"])]
                layer_idx = list(config.preview_layers_zeroindexed)
                time_idx = list(config.preview_timestep_positions)
                preview_cache[f"tsvd_common_k{rank}"][preview_index] = (
                    common_tokens[layer_idx][:, time_idx].to(torch.float16)
                )
                preview_cache[f"tsvd_residual_k{rank}"][preview_index] = (
                    residual_tokens[layer_idx][:, time_idx].to(torch.float16)
                )

    for idx, row in control_rows.iterrows():
        _, raw_norm = _extract_raw_normalized(
            model,
            latents["control_latents"][idx],
            noise["control_noise"][idx],
            int(row["imagenet_idx"]),
            config.time_values,
            device,
            config.stats_eps,
        )
        basis = tsvd_basis_v64(raw_norm.float(), rank=64)
        control_bases[idx] = basis.to(torch.float16)
        orth_errors.append(_orthonormality_error(basis))

    descriptor_path = config.cache_dir / "descriptors_fp16.npz"
    np.savez(descriptor_path, **descriptors)
    torch.save(
        {
            "main_image_ids": main_rows["image_id"].tolist(),
            "control_image_ids": control_rows["image_id"].tolist(),
            "main_v64": main_bases,
            "control_v64": control_bases,
        },
        config.cache_dir / "tsvd_v64_basis_fp16.pt",
    )
    torch.save(preview_cache, config.cache_dir / "preview_tokens_fp16.pt")

    metadata_rows.extend(
        [
            {
                "artifact": "descriptors_fp16.npz",
                "shape": "dict",
                "dtype": "float16",
                "bytes": descriptor_path.stat().st_size,
            },
            {
                "artifact": "tsvd_v64_basis_fp16.pt",
                "shape": f"main={tuple(main_bases.shape)},control={tuple(control_bases.shape)}",
                "dtype": "float16",
                "bytes": (config.cache_dir / "tsvd_v64_basis_fp16.pt").stat().st_size,
            },
            {
                "artifact": "preview_tokens_fp16.pt",
                "shape": "dict",
                "dtype": "float16",
                "bytes": (config.cache_dir / "preview_tokens_fp16.pt").stat().st_size,
            },
        ]
    )
    pd.DataFrame(metadata_rows).to_csv(config.cache_dir / "task0_metadata.csv", index=False)

    if len(preview_cache["image_ids"]) != config.preview_images:
        raise ValueError("Preview cache size mismatch.")
    if min(sanity_residuals) <= 0:
        raise ValueError("Residual sanity check failed: found zero residual magnitude.")

    done = {
        "stage": "task0",
        "cache_dir": str(config.cache_dir),
        "output_bytes": int(estimate_bytes(config.cache_dir)),
    }
    sanity = {
        "num_main_images": int(len(main_rows)),
        "num_control_images": int(len(control_rows)),
        "preview_images": int(len(preview_cache["image_ids"])),
        "max_basis_orth_error": float(max(orth_errors) if orth_errors else 0.0),
        "min_residual_abs_mean": float(min(sanity_residuals)),
    }
    return {"done": done, "sanity": sanity}
