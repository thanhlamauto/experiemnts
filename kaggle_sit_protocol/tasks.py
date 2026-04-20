from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from .config import ProtocolConfig
from .decomposition import (
    l2_normalize_tokens,
    mean_common,
    mean_pool_tokens,
    mean_residual,
    project_to_basis,
    spatial_normalize_tokens,
)
from .metrics import (
    WaveletEnergies,
    anchor_cosine_map,
    cds_metric,
    flat_cosine,
    lds_metric,
    linear_cka,
    rmsc_metric,
    tensor_stats,
    token_mean_cosine,
    wavelet_energies,
)
from .modeling import compute_patch_tokens, forward_features, linear_path_xt
from .probes import fit_linear_classifier
from .runtime import RuntimeIndex
from .stages import AnalysisRuntime


@dataclass
class FeatureBundle:
    image_id: str
    class_index_100: int
    imagenet_idx: int
    patch_tokens_clean: torch.Tensor | None
    raw_norm: torch.Tensor
    mean_common_tokens: torch.Tensor
    mean_residual_tokens: torch.Tensor
    basis_v64: torch.Tensor


def _main_rows(runtime: AnalysisRuntime) -> pd.DataFrame:
    return runtime.manifest[runtime.manifest["subset_role"] == "main"].reset_index(drop=True)


def _control_rows(runtime: AnalysisRuntime) -> pd.DataFrame:
    return runtime.manifest[runtime.manifest["subset_role"] == "control"].reset_index(drop=True)


def _time_values_tensor(config: ProtocolConfig, device: torch.device) -> torch.Tensor:
    return torch.tensor(config.time_values, device=device, dtype=torch.float32)


def _extract_bundle(
    runtime: AnalysisRuntime,
    row: pd.Series,
    *,
    source: str = "main",
    include_patch_tokens: bool = False,
    recompute_basis: bool = False,
) -> FeatureBundle:
    if not isinstance(runtime.index, RuntimeIndex):
        raise TypeError("Analysis runtime index is missing.")

    if source == "main":
        position = runtime.index.main_positions[str(row["image_id"])]
        x1 = runtime.latents["main_latents"][position]
        x0 = runtime.noise["main_noise"][position]
        basis = runtime.bases["main_v64"][position].float()
    elif source == "control":
        position = runtime.index.control_positions[str(row["image_id"])]
        x1 = runtime.latents["control_latents"][position]
        x0 = runtime.noise["control_noise"][position]
        basis = runtime.bases["control_v64"][position].float()
    elif source == "control_patchshuffle":
        position = runtime.index.control_positions[str(row["image_id"])]
        x1 = runtime.latents["control_patchshuffle_latents"][position]
        x0 = runtime.noise["control_noise"][position]
        basis = runtime.bases["control_v64"][position].float()
        recompute_basis = True
    elif source == "control_blockshuffle":
        position = runtime.index.control_positions[str(row["image_id"])]
        x1 = runtime.latents["control_blockshuffle_latents"][position]
        x0 = runtime.noise["control_noise"][position]
        basis = runtime.bases["control_v64"][position].float()
        recompute_basis = True
    else:
        raise ValueError(f"Unsupported source {source}")

    x1_device = x1.to(device=runtime.device, dtype=torch.float32)
    x0_device = x0.to(device=runtime.device, dtype=torch.float32)
    timesteps = _time_values_tensor(runtime.config, runtime.device)
    xt = linear_path_xt(x1_device, x0_device, timesteps)
    labels = torch.full((len(runtime.config.time_values),), int(row["imagenet_idx"]), device=runtime.device, dtype=torch.long)
    _, blocks = forward_features(runtime.model, xt, timesteps, labels)
    raw_norm = l2_normalize_tokens(blocks, eps=runtime.config.stats_eps).detach().cpu().float()
    mean_common_tokens = mean_common(raw_norm)
    mean_residual_tokens = mean_residual(raw_norm, mean_common_tokens)
    if recompute_basis:
        from .decomposition import tsvd_basis_v64

        basis = tsvd_basis_v64(raw_norm, rank=64)
    patch_tokens_clean = None
    if include_patch_tokens:
        patch_tokens_clean = compute_patch_tokens(runtime.model, x1_device.unsqueeze(0)).squeeze(0).detach().cpu().float()
    return FeatureBundle(
        image_id=str(row["image_id"]),
        class_index_100=int(row["class_index_100"]),
        imagenet_idx=int(row["imagenet_idx"]),
        patch_tokens_clean=patch_tokens_clean,
        raw_norm=raw_norm,
        mean_common_tokens=mean_common_tokens,
        mean_residual_tokens=mean_residual_tokens,
        basis_v64=basis.float(),
    )


def _variant_tensor(bundle: FeatureBundle, family: str, component: str) -> torch.Tensor:
    raw = bundle.raw_norm
    if family == "mean":
        if component == "raw":
            return raw
        if component == "common":
            return bundle.mean_common_tokens.unsqueeze(0).expand(raw.shape[0], -1, -1, -1)
        if component == "residual":
            return bundle.mean_residual_tokens
        raise KeyError(component)

    if not family.startswith("tsvd_k"):
        raise KeyError(family)
    rank = int(family.split("k", maxsplit=1)[1])
    common = project_to_basis(raw, bundle.basis_v64, rank).float()
    if component == "raw":
        return raw
    if component == "common":
        return common
    if component == "residual":
        return raw - common
    raise KeyError(component)


def _family_names(config: ProtocolConfig) -> list[str]:
    return ["mean"] + [f"tsvd_k{rank}" for rank in config.tsvd_ranks]


def _family_output_name(family: str) -> str:
    return "mean" if family == "mean" else family.replace("tsvd_k", "tsvd_K")


def _family_display_name(family: str) -> str:
    return "mean-common" if family == "mean" else family.replace("tsvd_k", "tsvd-K")


def _component_display_name(component: str) -> str:
    return {
        "raw": "raw",
        "common": "common",
        "residual": "residual",
    }[component]


def _apply_spatial_norm(tensor: torch.Tensor, config: ProtocolConfig) -> torch.Tensor:
    return spatial_normalize_tokens(
        tensor,
        gamma=config.spatial_norm_gamma,
        eps=config.spatial_norm_eps,
    )


def _save_object_npy(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload, allow_pickle=True)


def _load_object_npy(path: Path) -> dict[str, object]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        return loaded.item()
    raise TypeError(f"Expected object array in {path}")


def _probe_arrays(
    descriptors: np.ndarray,
    class_labels: np.ndarray,
    train_image_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_images, n_layers, n_times, dim = descriptors.shape
    samples = descriptors.reshape(n_images * n_layers * n_times, dim).astype(np.float32)
    layer_labels = np.tile(np.repeat(np.arange(n_layers), n_times), n_images)
    time_labels = np.tile(np.arange(n_times), n_images * n_layers)
    semantic_labels = np.repeat(class_labels, n_layers * n_times)
    image_mask = np.repeat(train_image_mask, n_layers * n_times)
    layer_index = np.tile(np.repeat(np.arange(n_layers), n_times), n_images)
    time_index = np.tile(np.arange(n_times), n_images * n_layers)
    return samples, layer_labels, time_labels, semantic_labels, image_mask, layer_index, time_index


def _broadcast_common_descriptors(common: np.ndarray, num_layers: int) -> np.ndarray:
    return np.broadcast_to(common[:, None, :, :], (common.shape[0], num_layers, common.shape[1], common.shape[2]))


def _save_lines_plot(path: Path, title: str, x_values: np.ndarray, grouped: dict[str, np.ndarray], xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, values in grouped.items():
        ax.plot(x_values, values, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _heatmap_page(pdf: PdfPages, matrix: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run_task1(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    by_family_rows: dict[str, list[dict[str, object]]] = {family: [] for family in _family_names(config)}
    main_rows = _main_rows(runtime)

    for _, row in main_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        for family in _family_names(config):
            for component in ("raw", "common", "residual"):
                tensor = _variant_tensor(bundle, family, component)
                for layer in range(config.num_layers):
                    for time_pos, time_index in enumerate(config.time_grid_indices):
                        stats = tensor_stats(tensor[layer, time_pos], eps=config.stats_eps)
                        by_family_rows[family].append(
                            {
                                "image_id": bundle.image_id,
                                "component": component,
                                "layer": layer + 1,
                                "time_position": time_pos,
                                "time_index": time_index,
                                **stats,
                            }
                        )

    summary_layer_curves: dict[str, np.ndarray] = {}
    summary_time_curves: dict[str, np.ndarray] = {}
    for family, rows in by_family_rows.items():
        df = pd.DataFrame(rows)
        family_name = _family_output_name(family)
        df.to_csv(outdir / f"task1_{family_name}_stats.csv", index=False)
        layer_summary = (
            df.groupby(["component", "layer"])[["mean", "var"]].mean().reset_index()
        )
        time_summary = (
            df.groupby(["component", "time_position"])[["mean", "var"]].mean().reset_index()
        )
        layer_summary.to_csv(outdir / f"task1_{family_name}_stats_by_layer.csv", index=False)
        time_summary.to_csv(outdir / f"task1_{family_name}_stats_by_time.csv", index=False)
        for component in ("raw", "common", "residual"):
            summary_layer_curves[f"{family_name}:{component}"] = (
                layer_summary[layer_summary["component"] == component]["mean"].to_numpy()
            )
            summary_time_curves[f"{family_name}:{component}"] = (
                time_summary[time_summary["component"] == component]["mean"].to_numpy()
            )

    _save_lines_plot(
        outdir / "task1_lines_by_layer.pdf",
        "Task 1 Mean Statistic By Layer",
        np.arange(1, config.num_layers + 1),
        summary_layer_curves,
        "Layer",
    )
    _save_lines_plot(
        outdir / "task1_lines_by_time.pdf",
        "Task 1 Mean Statistic By Timestep",
        np.arange(len(config.time_grid_indices)),
        summary_time_curves,
        "Timestep Position",
    )


def _compute_pairwise_heatmaps(tensor: torch.Tensor) -> dict[str, np.ndarray]:
    layers, times = tensor.shape[:2]
    layer_token = np.zeros((times, layers, layers), dtype=np.float32)
    layer_flat = np.zeros((times, layers, layers), dtype=np.float32)
    layer_cka = np.zeros((times, layers, layers), dtype=np.float32)
    time_token = np.zeros((layers, times, times), dtype=np.float32)
    time_flat = np.zeros((layers, times, times), dtype=np.float32)
    time_cka = np.zeros((layers, times, times), dtype=np.float32)

    for time_pos in range(times):
        for i in range(layers):
            for j in range(layers):
                x = tensor[i, time_pos]
                y = tensor[j, time_pos]
                layer_token[time_pos, i, j] = float(token_mean_cosine(x, y).item())
                layer_flat[time_pos, i, j] = float(flat_cosine(x, y).item())
                layer_cka[time_pos, i, j] = float(linear_cka(x, y).item())

    for layer in range(layers):
        for i in range(times):
            for j in range(times):
                x = tensor[layer, i]
                y = tensor[layer, j]
                time_token[layer, i, j] = float(token_mean_cosine(x, y).item())
                time_flat[layer, i, j] = float(flat_cosine(x, y).item())
                time_cka[layer, i, j] = float(linear_cka(x, y).item())

    return {
        "layer_tokenmean": layer_token,
        "layer_flatcos": layer_flat,
        "layer_cka": layer_cka,
        "time_tokenmean": time_token,
        "time_flatcos": time_flat,
        "time_cka": time_cka,
    }


def run_task2(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[str, dict[str, np.ndarray]] = {}
    main_rows = _main_rows(runtime)

    for family in _family_names(config):
        accumulators[family] = {
            "raw_layer_tokenmean": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "raw_layer_flatcos": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "raw_layer_cka": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "raw_time_tokenmean": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "raw_time_flatcos": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "raw_time_cka": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "common_layer_tokenmean": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "common_layer_flatcos": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "common_layer_cka": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "common_time_tokenmean": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "common_time_flatcos": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "common_time_cka": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "residual_layer_tokenmean": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "residual_layer_flatcos": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "residual_layer_cka": np.zeros((len(config.time_grid_indices), config.num_layers, config.num_layers), dtype=np.float64),
            "residual_time_tokenmean": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "residual_time_flatcos": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
            "residual_time_cka": np.zeros((config.num_layers, len(config.time_grid_indices), len(config.time_grid_indices)), dtype=np.float64),
        }

    for _, row in main_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        for family in _family_names(config):
            for component in ("raw", "common", "residual"):
                tensor = _variant_tensor(bundle, family, component)
                heatmaps = _compute_pairwise_heatmaps(tensor)
                for metric_name, values in heatmaps.items():
                    accumulators[family][f"{component}_{metric_name}"] += values

    with PdfPages(outdir / "task2_figures.pdf") as pdf:
        for family, payload in accumulators.items():
            family_name = _family_output_name(family)
            averaged = {key: value / float(len(main_rows)) for key, value in payload.items()}
            tokenmean_payload = {
                "layer_raw": averaged["raw_layer_tokenmean"],
                "layer_common": averaged["common_layer_tokenmean"],
                "layer_residual": averaged["residual_layer_tokenmean"],
                "time_raw": averaged["raw_time_tokenmean"],
                "time_common": averaged["common_time_tokenmean"],
                "time_residual": averaged["residual_time_tokenmean"],
            }
            flatcos_payload = {
                "layer_raw": averaged["raw_layer_flatcos"],
                "layer_common": averaged["common_layer_flatcos"],
                "layer_residual": averaged["residual_layer_flatcos"],
                "time_raw": averaged["raw_time_flatcos"],
                "time_common": averaged["common_time_flatcos"],
                "time_residual": averaged["residual_time_flatcos"],
            }
            cka_payload = {
                "layer_raw": averaged["raw_layer_cka"],
                "layer_common": averaged["common_layer_cka"],
                "layer_residual": averaged["residual_layer_cka"],
                "time_raw": averaged["raw_time_cka"],
                "time_common": averaged["common_time_cka"],
                "time_residual": averaged["residual_time_cka"],
            }
            _save_object_npy(outdir / f"task2_{family_name}_tokenmean_heatmap.npy", tokenmean_payload)
            _save_object_npy(outdir / f"task2_{family_name}_flatcos_heatmap.npy", flatcos_payload)
            _save_object_npy(outdir / f"task2_{family_name}_cka_heatmap.npy", cka_payload)
            _heatmap_page(pdf, tokenmean_payload["layer_raw"][0], f"Task 2 {family_name} raw layer heatmap @ t0", "Layer", "Layer")
            _heatmap_page(pdf, tokenmean_payload["layer_common"][0], f"Task 2 {family_name} common layer heatmap @ t0", "Layer", "Layer")
            _heatmap_page(pdf, tokenmean_payload["layer_residual"][0], f"Task 2 {family_name} residual layer heatmap @ t0", "Layer", "Layer")


def _run_task4_like(
    config: ProtocolConfig,
    runtime: AnalysisRuntime,
    outdir: Path,
    *,
    file_prefix: str,
    title_prefix: str,
    transform=None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows_by_family_metric: dict[tuple[str, str], list[dict[str, object]]] = {}
    main_rows = _main_rows(runtime)

    for _, row in main_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        for family in _family_names(config):
            for component in ("raw", "common", "residual"):
                tensor = _variant_tensor(bundle, family, component)
                if transform is not None:
                    tensor = transform(tensor)
                for layer in range(config.num_layers):
                    for time_pos in range(len(config.time_grid_indices)):
                        z = tensor[layer, time_pos]
                        metrics = {
                            "lds": lds_metric(z, config.patch_grid_size),
                            "cds": cds_metric(z, config.patch_grid_size),
                            "rmsc": rmsc_metric(z, eps=config.stats_eps),
                        }
                        for metric_name, value in metrics.items():
                            rows_by_family_metric.setdefault((family, metric_name), []).append(
                                {
                                    "image_id": bundle.image_id,
                                    "component": component,
                                    "layer": layer + 1,
                                    "time_position": time_pos,
                                    "value": value,
                                }
                            )

    layer_curves: dict[str, np.ndarray] = {}
    time_curves: dict[str, np.ndarray] = {}
    for (family, metric_name), rows in rows_by_family_metric.items():
        df = pd.DataFrame(rows)
        family_name = _family_output_name(family)
        df.to_csv(outdir / f"{file_prefix}_{family_name}_{metric_name}.csv", index=False)
        summary = df.groupby(["component", "layer", "time_position"])["value"].mean().reset_index()
        for component in ("raw", "common", "residual"):
            comp = summary[summary["component"] == component]
            layer_curve = comp.groupby("layer")["value"].mean().to_numpy()
            time_curve = comp.groupby("time_position")["value"].mean().to_numpy()
            layer_curves[f"{family_name}:{metric_name}:{component}"] = layer_curve
            time_curves[f"{family_name}:{metric_name}:{component}"] = time_curve

    _save_lines_plot(
        outdir / f"{file_prefix}_layerwise_curves.pdf",
        f"{title_prefix} Layer-wise Spatial Metrics",
        np.arange(1, config.num_layers + 1),
        layer_curves,
        "Layer",
    )
    _save_lines_plot(
        outdir / f"{file_prefix}_timestep_curves.pdf",
        f"{title_prefix} Timestep-wise Spatial Metrics",
        np.arange(len(config.time_grid_indices)),
        time_curves,
        "Timestep Position",
    )


def run_task4(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    _run_task4_like(
        config,
        runtime,
        outdir,
        file_prefix="task4",
        title_prefix="Task 4",
    )


def run_task4b_spatialnorm(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    _run_task4_like(
        config,
        runtime,
        outdir,
        file_prefix="task4b_spatialnorm",
        title_prefix="Task 4B Spatial-Norm",
        transform=lambda tensor: _apply_spatial_norm(tensor, config),
    )


def run_task5(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    descriptors = np.load(config.cache_dir / "descriptors_fp16.npz")
    main_rows = _main_rows(runtime)
    class_labels = main_rows["class_index_100"].to_numpy(dtype=np.int64)
    train_image_mask = (main_rows["probe_split"].to_numpy() == "train")

    variant_arrays: dict[str, np.ndarray] = {
        "raw": descriptors["raw"].astype(np.float32),
        "mean-common": _broadcast_common_descriptors(descriptors["mean_common"].astype(np.float32), config.num_layers),
        "mean-residual": descriptors["mean_residual"].astype(np.float32),
    }
    for rank in config.tsvd_ranks:
        variant_arrays[f"tsvd-common-k{rank}"] = descriptors[f"tsvd_common_k{rank}"].astype(np.float32)
        variant_arrays[f"tsvd-residual-k{rank}"] = descriptors[f"tsvd_residual_k{rank}"].astype(np.float32)

    layer_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    probe_maps: dict[str, np.ndarray] = {}

    for variant, array in variant_arrays.items():
        samples, layer_labels, time_labels, semantic_labels, train_mask, layer_index, time_index = _probe_arrays(
            array, class_labels, train_image_mask
        )
        test_mask = ~train_mask

        layer_acc, layer_pred = fit_linear_classifier(
            samples[train_mask],
            layer_labels[train_mask],
            samples[test_mask],
            layer_labels[test_mask],
            num_classes=config.num_layers,
            epochs=config.probe_epochs,
            batch_size=config.probe_batch_size,
            lr=config.probe_lr,
            weight_decay=config.probe_weight_decay,
            device=str(runtime.device),
            seed=config.seed,
        )
        time_acc, time_pred = fit_linear_classifier(
            samples[train_mask],
            time_labels[train_mask],
            samples[test_mask],
            time_labels[test_mask],
            num_classes=len(config.time_grid_indices),
            epochs=config.probe_epochs,
            batch_size=config.probe_batch_size,
            lr=config.probe_lr,
            weight_decay=config.probe_weight_decay,
            device=str(runtime.device),
            seed=config.seed,
        )
        semantic_acc, _ = fit_linear_classifier(
            samples[train_mask],
            semantic_labels[train_mask],
            samples[test_mask],
            semantic_labels[test_mask],
            num_classes=100,
            epochs=config.probe_epochs,
            batch_size=config.probe_batch_size,
            lr=config.probe_lr,
            weight_decay=config.probe_weight_decay,
            device=str(runtime.device),
            seed=config.seed,
        )

        layer_rows.append({"variant": variant, "accuracy": layer_acc})
        time_rows.append({"variant": variant, "accuracy": time_acc})
        semantic_rows.append({"variant": variant, "accuracy": semantic_acc})

        layer_truth = layer_labels[test_mask]
        time_truth = time_labels[test_mask]
        layer_idx_test = layer_index[test_mask]
        time_idx_test = time_index[test_mask]
        cell_map = np.zeros((config.num_layers, len(config.time_grid_indices)), dtype=np.float32)
        cell_counts = np.zeros_like(cell_map)
        cell_layer_correct = np.zeros_like(cell_map)
        cell_time_correct = np.zeros_like(cell_map)
        for sample_idx in range(len(layer_truth)):
            li = int(layer_idx_test[sample_idx])
            ti = int(time_idx_test[sample_idx])
            cell_counts[li, ti] += 1.0
            cell_layer_correct[li, ti] += float(layer_pred[sample_idx] == layer_truth[sample_idx])
            cell_time_correct[li, ti] += float(time_pred[sample_idx] == time_truth[sample_idx])
        valid = cell_counts > 0
        cell_map[valid] = 0.5 * (
            cell_layer_correct[valid] / cell_counts[valid] + cell_time_correct[valid] / cell_counts[valid]
        )
        probe_maps[variant] = cell_map

    pd.DataFrame([row for row in layer_rows if row["variant"].startswith("mean") or row["variant"] == "raw"]).to_csv(
        outdir / "task5_layerprobe_mean.csv", index=False
    )
    pd.DataFrame([row for row in time_rows if row["variant"].startswith("mean") or row["variant"] == "raw"]).to_csv(
        outdir / "task5_timeprobe_mean.csv", index=False
    )
    pd.DataFrame([row for row in layer_rows if row["variant"].startswith("tsvd") or row["variant"] == "raw"]).to_csv(
        outdir / "task5_layerprobe_tsvd.csv", index=False
    )
    pd.DataFrame([row for row in time_rows if row["variant"].startswith("tsvd") or row["variant"] == "raw"]).to_csv(
        outdir / "task5_timeprobe_tsvd.csv", index=False
    )
    pd.DataFrame(semantic_rows).to_csv(outdir / "task5_semanticprobe.csv", index=False)
    np.savez(outdir / "task5_probe_maps.npz", **probe_maps)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layer_rows))
    ax.bar(x, [row["accuracy"] for row in layer_rows])
    ax.set_xticks(x)
    ax.set_xticklabels([row["variant"] for row in layer_rows], rotation=45, ha="right")
    ax.set_title("Task 5 Layer Probe Accuracy")
    fig.tight_layout()
    fig.savefig(outdir / "task5_probe_bars.pdf")
    plt.close(fig)


def run_task3(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    main_rows = _main_rows(runtime)
    probe_maps = np.load(outdir / "task5_probe_maps.npz")
    raw_lds = np.zeros((config.num_layers, len(config.time_grid_indices)), dtype=np.float64)
    cos_maps: dict[str, np.ndarray] = {
        family: np.zeros((config.num_layers, len(config.time_grid_indices)), dtype=np.float64)
        for family in _family_names(config)
    }

    for _, row in main_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        for layer in range(config.num_layers):
            for time_pos in range(len(config.time_grid_indices)):
                raw_lds[layer, time_pos] += lds_metric(bundle.raw_norm[layer, time_pos], config.patch_grid_size)
        for family in _family_names(config):
            common = _variant_tensor(bundle, family, "common")
            for layer in range(config.num_layers):
                for time_pos in range(len(config.time_grid_indices)):
                    cos_maps[family][layer, time_pos] += float(
                        token_mean_cosine(bundle.raw_norm[layer, time_pos], common[layer, time_pos]).item()
                    )

    raw_lds /= float(len(main_rows))
    np.save(outdir / "task3_mean_lds_map.npy", raw_lds)
    with PdfPages(outdir / "task3_maps.pdf") as pdf:
        _heatmap_page(pdf, raw_lds, "Task 3 Raw LDS Map", "Timestep", "Layer")
        for family in _family_names(config):
            family_name = _family_output_name(family)
            cos_map = cos_maps[family] / float(len(main_rows))
            residual_variant = "mean-residual" if family == "mean" else f"tsvd-residual-k{family.split('k', 1)[1]}"
            probe_map = probe_maps[residual_variant]
            np.save(outdir / f"task3_{family_name}_cos_map.npy", cos_map)
            np.save(outdir / f"task3_{family_name}_probe_map.npy", probe_map)
            if family == "mean":
                np.save(outdir / "task3_mean_cos_map.npy", cos_map)
                np.save(outdir / "task3_mean_probe_map.npy", probe_map)
            _heatmap_page(pdf, cos_map, f"Task 3 {family_name} Cosine-to-Common", "Timestep", "Layer")
            _heatmap_page(pdf, probe_map, f"Task 3 {family_name} Residual Probe Map", "Timestep", "Layer")


def run_task6(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    layer_mean = pd.read_csv(outdir / "task5_layerprobe_mean.csv")
    time_mean = pd.read_csv(outdir / "task5_timeprobe_mean.csv")
    layer_tsvd = pd.read_csv(outdir / "task5_layerprobe_tsvd.csv")
    time_tsvd = pd.read_csv(outdir / "task5_timeprobe_tsvd.csv")
    semantic = pd.read_csv(outdir / "task5_semanticprobe.csv")

    def spatial_summary(family_name: str) -> dict[str, float]:
        summary: dict[str, float] = {}
        for metric in ("lds", "cds", "rmsc"):
            df = pd.read_csv(outdir / f"task4_{family_name}_{metric}.csv")
            grouped = df.groupby("component")["value"].mean()
            for component, value in grouped.items():
                summary[f"{component}_{metric}"] = float(value)
        return summary

    mean_spatial = spatial_summary("mean")
    tsvd_spatial = {rank: spatial_summary(f"tsvd_K{rank}") for rank in config.tsvd_ranks}

    mean_rows = []
    for variant in ["raw", "mean-common", "mean-residual"]:
        component = "common" if "common" in variant else ("residual" if "residual" in variant else "raw")
        mean_rows.append(
            {
                "variant": variant,
                "layer_probe_acc": float(layer_mean[layer_mean["variant"] == variant]["accuracy"].iloc[0]),
                "time_probe_acc": float(time_mean[time_mean["variant"] == variant]["accuracy"].iloc[0]),
                "semantic_probe_acc": float(semantic[semantic["variant"] == variant]["accuracy"].iloc[0]),
                "lds": mean_spatial[f"{component}_lds"],
                "cds": mean_spatial[f"{component}_cds"],
                "rmsc": mean_spatial[f"{component}_rmsc"],
            }
        )
    pd.DataFrame(mean_rows).to_csv(outdir / "task6_ablation_mean.csv", index=False)

    tsvd_rows = []
    for rank in config.tsvd_ranks:
        for variant in ["raw", f"tsvd-common-k{rank}", f"tsvd-residual-k{rank}"]:
            component = "common" if "common" in variant else ("residual" if "residual" in variant else "raw")
            tsvd_rows.append(
                {
                    "variant": variant,
                    "layer_probe_acc": float(layer_tsvd[layer_tsvd["variant"] == variant]["accuracy"].iloc[0]),
                    "time_probe_acc": float(time_tsvd[time_tsvd["variant"] == variant]["accuracy"].iloc[0]),
                    "semantic_probe_acc": float(semantic[semantic["variant"] == variant]["accuracy"].iloc[0]),
                    "lds": tsvd_spatial[rank][f"{component}_lds"],
                    "cds": tsvd_spatial[rank][f"{component}_cds"],
                    "rmsc": tsvd_spatial[rank][f"{component}_rmsc"],
                }
            )
    pd.DataFrame(tsvd_rows).to_csv(outdir / "task6_ablation_tsvd.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(mean_rows)), [row["layer_probe_acc"] for row in mean_rows], label="layer probe")
    ax.bar(np.arange(len(mean_rows)), [row["time_probe_acc"] for row in mean_rows], alpha=0.5, label="time probe")
    ax.set_xticks(np.arange(len(mean_rows)))
    ax.set_xticklabels([row["variant"] for row in mean_rows], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "task6_ablation_bars.pdf")
    plt.close(fig)


def _anchor_indices(grid_size: int) -> list[int]:
    coords = [(grid_size // 4, grid_size // 4), (grid_size // 2, grid_size // 2), ((3 * grid_size) // 4, (3 * grid_size) // 4)]
    return [y * grid_size + x for y, x in coords]


def run_task7(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    preview = torch.load(config.cache_dir / "preview_tokens_fp16.pt", map_location="cpu")
    anchors = _anchor_indices(config.patch_grid_size)

    with PdfPages(outdir / "task7_patch_cosmap.pdf") as patch_pdf:
        for preview_idx, image_id in enumerate(preview["image_ids"]):
            for anchor in anchors:
                cosmap = anchor_cosine_map(preview["patch_tokens_clean"][preview_idx].float(), anchor).reshape(config.patch_grid_size, config.patch_grid_size)
                _heatmap_page(patch_pdf, cosmap.numpy(), f"Patch cosine map {image_id} anchor {anchor}", "X", "Y")

    with PdfPages(outdir / "task7_mean_cosmap.pdf") as mean_pdf:
        for preview_idx, image_id in enumerate(preview["image_ids"]):
            for layer_offset, layer in enumerate(config.preview_layers_1indexed):
                for time_offset, time_pos in enumerate(config.preview_timestep_positions):
                    for anchor in anchors:
                        raw_map = anchor_cosine_map(preview["raw"][preview_idx, layer_offset, time_offset].float(), anchor)
                        common_map = anchor_cosine_map(preview["mean_common"][preview_idx, time_offset].float(), anchor)
                        residual_map = anchor_cosine_map(preview["mean_residual"][preview_idx, layer_offset, time_offset].float(), anchor)
                        _heatmap_page(mean_pdf, raw_map.reshape(config.patch_grid_size, config.patch_grid_size).numpy(), f"Raw {image_id} L{layer} T{time_pos} A{anchor}", "X", "Y")
                        _heatmap_page(mean_pdf, common_map.reshape(config.patch_grid_size, config.patch_grid_size).numpy(), f"Mean-common {image_id} L{layer} T{time_pos} A{anchor}", "X", "Y")
                        _heatmap_page(mean_pdf, residual_map.reshape(config.patch_grid_size, config.patch_grid_size).numpy(), f"Mean-residual {image_id} L{layer} T{time_pos} A{anchor}", "X", "Y")

    with PdfPages(outdir / "task7_tsvd_cosmap.pdf") as tsvd_pdf:
        for rank in config.tsvd_ranks:
            for preview_idx, image_id in enumerate(preview["image_ids"]):
                for layer_offset, layer in enumerate(config.preview_layers_1indexed):
                    for time_offset, time_pos in enumerate(config.preview_timestep_positions):
                        for anchor in anchors:
                            common_map = anchor_cosine_map(preview[f"tsvd_common_k{rank}"][preview_idx, layer_offset, time_offset].float(), anchor)
                            residual_map = anchor_cosine_map(preview[f"tsvd_residual_k{rank}"][preview_idx, layer_offset, time_offset].float(), anchor)
                            _heatmap_page(tsvd_pdf, common_map.reshape(config.patch_grid_size, config.patch_grid_size).numpy(), f"TSVD-{rank} common {image_id} L{layer} T{time_pos} A{anchor}", "X", "Y")
                            _heatmap_page(tsvd_pdf, residual_map.reshape(config.patch_grid_size, config.patch_grid_size).numpy(), f"TSVD-{rank} residual {image_id} L{layer} T{time_pos} A{anchor}", "X", "Y")


def _select_tensor_block(
    tensor: torch.Tensor,
    layer_positions: tuple[int, ...],
    time_positions: tuple[int, ...],
) -> torch.Tensor:
    return tensor[list(layer_positions)][:, list(time_positions)]


def _tokens_to_pca_rgb(tokens: torch.Tensor, pca_model, grid_size: int) -> np.ndarray:
    flat = tokens.reshape(-1, tokens.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
    comps = pca_model.transform(flat).reshape(grid_size, grid_size, 3)
    comps = comps - comps.min()
    comps = comps / np.clip(comps.max(), 1e-6, None)
    return comps


def _load_preview_image(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((image_size, image_size)))


def _render_pca_panel_page(
    pdf: PdfPages,
    *,
    image_path: str,
    image_id: str,
    family: str,
    component: str,
    spatial_norm: bool,
    layer_labels: tuple[int, ...],
    time_labels: list[str],
    rgb_grid: list[list[np.ndarray]],
    image_size: int,
) -> None:
    rows = len(time_labels)
    cols = len(layer_labels)
    fig = plt.figure(figsize=(2.2 + 1.3 * cols, 0.8 + 1.35 * rows))
    grid = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[1.25] + [1.0] * cols,
        left=0.04,
        right=0.99,
        top=0.90,
        bottom=0.06,
        wspace=0.08,
        hspace=0.08,
    )

    source_ax = fig.add_subplot(grid[:, 0])
    source_ax.imshow(_load_preview_image(image_path, image_size=image_size))
    source_ax.set_title("input", fontsize=10)
    source_ax.axis("off")

    for row_idx in range(rows):
        for col_idx in range(cols):
            ax = fig.add_subplot(grid[row_idx, col_idx + 1])
            ax.imshow(rgb_grid[row_idx][col_idx])
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"layer{layer_labels[col_idx]}", fontsize=10)
            if col_idx == 0:
                ax.text(
                    -0.16,
                    0.5,
                    time_labels[row_idx],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

    norm_label = "after spatial norm" if spatial_norm else "before spatial norm"
    fig.suptitle(
        f"Task 8 PCA-RGB | {_family_display_name(family)} | {_component_display_name(component)} | {norm_label} | {image_id}",
        fontsize=12,
        fontweight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def run_task8(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import IncrementalPCA, PCA
    from sklearn.manifold import TSNE
    import umap

    preview_rows = _main_rows(runtime)
    preview_rows = preview_rows[preview_rows["preview"]].reset_index(drop=True)
    families = _family_names(config)
    components = ("raw", "common", "residual")
    pca_models = {
        (family, component, spatial_norm): IncrementalPCA(n_components=3)
        for family in families
        for component in components
        for spatial_norm in (False, True)
    }
    token_sample_blocks: list[np.ndarray] = []
    hidden_rows: list[list[float]] = []

    for _, row in preview_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        for family in families:
            for component in components:
                if family != "mean" and component == "raw":
                    continue
                base_tensor = _variant_tensor(bundle, family, component)
                panel_tensor = _select_tensor_block(
                    base_tensor,
                    config.pca_panel_layers_zeroindexed,
                    config.pca_panel_timestep_positions,
                )
                pca_models[(family, component, False)].partial_fit(
                    panel_tensor.reshape(-1, panel_tensor.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
                )
                spatial_panel = _apply_spatial_norm(panel_tensor, config)
                pca_models[(family, component, True)].partial_fit(
                    spatial_panel.reshape(-1, spatial_panel.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
                )

        mean_residual_subset = _select_tensor_block(
            _variant_tensor(bundle, "mean", "residual"),
            config.preview_layers_zeroindexed,
            config.preview_timestep_positions,
        )
        token_sample_blocks.append(
            mean_residual_subset.reshape(-1, mean_residual_subset.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
        )

        mean_common = _variant_tensor(bundle, "mean", "common")
        for layer in range(config.num_layers):
            for time_pos in range(len(config.time_grid_indices)):
                raw = bundle.raw_norm[layer, time_pos]
                hidden_rows.append(
                    [
                        lds_metric(raw, config.patch_grid_size),
                        cds_metric(raw, config.patch_grid_size),
                        rmsc_metric(raw, eps=config.stats_eps),
                        float(token_mean_cosine(raw, mean_common[layer, time_pos]).item()),
                        float(raw.mean().item()),
                        float(raw.var(unbiased=False).item()),
                    ]
                )

    layer_labels = config.pca_panel_layers_1indexed
    time_labels = [f"t={config.time_values[pos]:.2f}" for pos in config.pca_panel_timestep_positions]
    with PdfPages(outdir / "task8_mean_pca_rgb.pdf") as mean_pdf, PdfPages(outdir / "task8_tsvd_visuals.pdf") as tsvd_pdf:
        for _, row in preview_rows.iterrows():
            bundle = _extract_bundle(runtime, row, source="main")
            for family in families:
                target_pdf = mean_pdf if family == "mean" else tsvd_pdf
                for component in components:
                    if family != "mean" and component == "raw":
                        continue
                    base_tensor = _variant_tensor(bundle, family, component)
                    for spatial_norm in (False, True):
                        panel_tensor = _select_tensor_block(
                            base_tensor,
                            config.pca_panel_layers_zeroindexed,
                            config.pca_panel_timestep_positions,
                        )
                        if spatial_norm:
                            panel_tensor = _apply_spatial_norm(panel_tensor, config)
                        rgb_grid: list[list[np.ndarray]] = []
                        for time_offset in range(len(config.pca_panel_timestep_positions)):
                            row_images = []
                            for layer_offset in range(len(config.pca_panel_layers_zeroindexed)):
                                row_images.append(
                                    _tokens_to_pca_rgb(
                                        panel_tensor[layer_offset, time_offset],
                                        pca_models[(family, component, spatial_norm)],
                                        config.patch_grid_size,
                                    )
                                )
                            rgb_grid.append(row_images)
                        _render_pca_panel_page(
                            target_pdf,
                            image_path=str(row["absolute_path"]),
                            image_id=str(row["image_id"]),
                            family=family,
                            component=component,
                            spatial_norm=spatial_norm,
                            layer_labels=layer_labels,
                            time_labels=time_labels,
                            rgb_grid=rgb_grid,
                            image_size=config.image_size,
                        )

    token_samples = np.concatenate(token_sample_blocks, axis=0)
    pca50 = PCA(n_components=min(50, token_samples.shape[1]), random_state=config.seed).fit_transform(token_samples)
    tsne = TSNE(n_components=2, init="pca", random_state=config.seed, perplexity=30).fit_transform(pca50)
    reducer = umap.UMAP(n_components=2, random_state=config.seed)
    umap_token = reducer.fit_transform(pca50)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(tsne[:, 0], tsne[:, 1], s=4)
    ax.set_title("Task 8 t-SNE token-level mean residual")
    fig.tight_layout()
    fig.savefig(outdir / "task8_mean_tsne.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(umap_token[:, 0], umap_token[:, 1], s=4)
    ax.set_title("Task 8 UMAP token-level mean residual")
    fig.tight_layout()
    fig.savefig(outdir / "task8_mean_umap_token.pdf")
    plt.close(fig)

    hidden = np.asarray(hidden_rows, dtype=np.float32)
    hidden_umap = umap.UMAP(n_components=2, random_state=config.seed).fit_transform(hidden)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(hidden_umap[:, 0], hidden_umap[:, 1], s=10)
    ax.set_title("Task 8 UMAP hidden-state mean family")
    fig.tight_layout()
    fig.savefig(outdir / "task8_mean_umap_hiddenstate.pdf")
    plt.close(fig)


def run_task9(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    control_rows = _control_rows(runtime)
    mean_rows: list[dict[str, object]] = []
    tsvd_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []

    for _, row in control_rows.iterrows():
        raw_bundle = _extract_bundle(runtime, row, source="control")
        patch_bundle = _extract_bundle(runtime, row, source="control_patchshuffle")
        block_bundle = _extract_bundle(runtime, row, source="control_blockshuffle")

        for broken_name, broken_bundle in (("patchshuffle", patch_bundle), ("blockshuffle", block_bundle)):
            raw_common = raw_bundle.mean_common_tokens.unsqueeze(0).expand(config.num_layers, -1, -1, -1)
            broken_common = broken_bundle.mean_common_tokens.unsqueeze(0).expand(config.num_layers, -1, -1, -1)
            raw_metric = np.mean(
                [lds_metric(raw_common[layer, time], config.patch_grid_size) for layer in range(config.num_layers) for time in range(len(config.time_grid_indices))]
            )
            broken_metric = np.mean(
                [lds_metric(broken_common[layer, time], config.patch_grid_size) for layer in range(config.num_layers) for time in range(len(config.time_grid_indices))]
            )
            mean_rows.append(
                {
                    "image_id": raw_bundle.image_id,
                    "broken": broken_name,
                    "common_lds_raw": raw_metric,
                    "common_lds_broken": broken_metric,
                    "delta": raw_metric - broken_metric,
                }
            )
            delta_rows.append(
                {
                    "image_id": raw_bundle.image_id,
                    "family": "mean",
                    "broken": broken_name,
                    "delta_common_lds": raw_metric - broken_metric,
                }
            )

            for rank in config.tsvd_ranks:
                raw_common_rank = _variant_tensor(raw_bundle, f"tsvd_k{rank}", "common")
                broken_common_rank = _variant_tensor(broken_bundle, f"tsvd_k{rank}", "common")
                raw_metric_rank = np.mean(
                    [lds_metric(raw_common_rank[layer, time], config.patch_grid_size) for layer in range(config.num_layers) for time in range(len(config.time_grid_indices))]
                )
                broken_metric_rank = np.mean(
                    [lds_metric(broken_common_rank[layer, time], config.patch_grid_size) for layer in range(config.num_layers) for time in range(len(config.time_grid_indices))]
                )
                tsvd_rows.append(
                    {
                        "image_id": raw_bundle.image_id,
                        "broken": broken_name,
                        "rank": rank,
                        "common_lds_raw": raw_metric_rank,
                        "common_lds_broken": broken_metric_rank,
                        "delta": raw_metric_rank - broken_metric_rank,
                    }
                )
                delta_rows.append(
                    {
                        "image_id": raw_bundle.image_id,
                        "family": f"tsvd_k{rank}",
                        "broken": broken_name,
                        "delta_common_lds": raw_metric_rank - broken_metric_rank,
                    }
                )

    pd.DataFrame([row for row in mean_rows if row["broken"] == "patchshuffle"]).to_csv(outdir / "task9_mean_patchshuffle.csv", index=False)
    pd.DataFrame([row for row in mean_rows if row["broken"] == "blockshuffle"]).to_csv(outdir / "task9_mean_blockshuffle.csv", index=False)
    pd.DataFrame(tsvd_rows).to_csv(outdir / "task9_tsvd_stress.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(outdir / "task9_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    delta_df = pd.DataFrame(delta_rows)
    for family, group in delta_df.groupby("family"):
        grouped = group.groupby("broken")["delta_common_lds"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=family)
    ax.set_title("Task 9 Common LDS Drop")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "task9_metric_drop.pdf")
    plt.close(fig)


def _flatten_wavelet_rows(
    image_id: str,
    family: str,
    component: str,
    layer: int,
    time_position: int,
    energies: Iterable[WaveletEnergies],
) -> list[dict[str, object]]:
    rows = []
    for energy in energies:
        rows.append(
            {
                "image_id": image_id,
                "family": family,
                "component": component,
                "layer": layer + 1,
                "time_position": time_position,
                "level": energy.level,
                "approximation_energy": energy.approximation_energy,
                "detail_energy": energy.detail_energy,
                "ratio": energy.ratio,
            }
        )
    return rows


def run_task10(config: ProtocolConfig, runtime: AnalysisRuntime, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    main_rows = _main_rows(runtime)
    mean_rows: list[dict[str, object]] = []
    tsvd_rows: list[dict[str, object]] = []

    for _, row in main_rows.iterrows():
        bundle = _extract_bundle(runtime, row, source="main")
        mean_common_tensor = bundle.mean_common_tokens.unsqueeze(0).expand(config.num_layers, -1, -1, -1)
        for time_pos in range(len(config.time_grid_indices)):
            mean_rows.extend(
                _flatten_wavelet_rows(
                    bundle.image_id,
                    "mean",
                    "common",
                    0,
                    time_pos,
                    wavelet_energies(mean_common_tensor[0, time_pos], config.patch_grid_size),
                )
            )
        for layer in range(config.num_layers):
            for time_pos in range(len(config.time_grid_indices)):
                mean_rows.extend(
                    _flatten_wavelet_rows(
                        bundle.image_id,
                        "mean",
                        "residual",
                        layer,
                        time_pos,
                        wavelet_energies(bundle.mean_residual_tokens[layer, time_pos], config.patch_grid_size),
                    )
                )

        for rank in config.tsvd_ranks:
            common = _variant_tensor(bundle, f"tsvd_k{rank}", "common")
            residual = _variant_tensor(bundle, f"tsvd_k{rank}", "residual")
            for layer in range(config.num_layers):
                for time_pos in range(len(config.time_grid_indices)):
                    tsvd_rows.extend(
                        _flatten_wavelet_rows(
                            bundle.image_id,
                            f"tsvd_k{rank}",
                            "common",
                            layer,
                            time_pos,
                            wavelet_energies(common[layer, time_pos], config.patch_grid_size),
                        )
                    )
                    tsvd_rows.extend(
                        _flatten_wavelet_rows(
                            bundle.image_id,
                            f"tsvd_k{rank}",
                            "residual",
                            layer,
                            time_pos,
                            wavelet_energies(residual[layer, time_pos], config.patch_grid_size),
                        )
                    )

    mean_df = pd.DataFrame(mean_rows)
    tsvd_df = pd.DataFrame(tsvd_rows)
    mean_df.to_csv(outdir / "task10_wavelet_mean.csv", index=False)
    tsvd_df.to_csv(outdir / "task10_wavelet_tsvd.csv", index=False)
    ratio_df = pd.concat([mean_df, tsvd_df], ignore_index=True)
    ratio_df.to_csv(outdir / "task10_wavelet_ratio.csv", index=False)

    with PdfPages(outdir / "task10_figures.pdf") as pdf:
        grouped = ratio_df.groupby(["family", "component", "level"])["ratio"].mean().reset_index()
        for (family, component), subdf in grouped.groupby(["family", "component"]):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(subdf["level"].astype(str), subdf["ratio"])
            ax.set_title(f"Wavelet ratio {family} {component}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def run_all_tasks(config: ProtocolConfig, runtime: AnalysisRuntime, stage_dir: Path) -> dict[str, object]:
    run_task1(config, runtime, stage_dir)
    run_task2(config, runtime, stage_dir)
    run_task4(config, runtime, stage_dir)
    run_task4b_spatialnorm(config, runtime, stage_dir)
    run_task5(config, runtime, stage_dir)
    run_task3(config, runtime, stage_dir)
    run_task6(config, runtime, stage_dir)
    run_task7(config, runtime, stage_dir)
    run_task8(config, runtime, stage_dir)
    run_task9(config, runtime, stage_dir)
    run_task10(config, runtime, stage_dir)
    done = {
        "stage": "analysis",
        "tasks": [1, 2, 3, 4, "4b_spatialnorm", 5, 6, 7, 8, 9, 10],
    }
    sanity = {
        "task5_probe_maps_exists": bool((stage_dir / "task5_probe_maps.npz").exists()),
        "task4b_spatialnorm_exists": bool((stage_dir / "task4b_spatialnorm_mean_lds.csv").exists()),
        "task9_delta_exists": bool((stage_dir / "task9_delta.csv").exists()),
        "task10_wavelet_exists": bool((stage_dir / "task10_wavelet_ratio.csv").exists()),
    }
    return {"done": done, "sanity": sanity}
