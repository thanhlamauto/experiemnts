#!/usr/bin/env python3
"""Plot SiT vs REPA spatial-metric heatmaps plus their difference maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_metrics_tsv(path: str | Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "backend": row["backend"],
                    "ckpt": row["ckpt"],
                    "metric": row["metric"],
                    "layer": int(row["layer"]),
                    "noise_level": float(row["noise_level"]),
                    "model_t": float(row["model_t"]),
                    "value": float(row["value"]),
                    "num_images": int(row["num_images"]),
                    "path_type": row["path_type"],
                    "noise_protocol": row["noise_protocol"],
                }
            )
    return rows


def metric_rows_to_matrix(rows: Sequence[Dict[str, object]], metric: str):
    metric_rows = [row for row in rows if row["metric"] == metric]
    if not metric_rows:
        raise ValueError(f"No rows found for metric={metric!r}")
    layers = sorted({int(row["layer"]) for row in metric_rows})
    noise_levels = sorted({float(row["noise_level"]) for row in metric_rows}, reverse=True)
    matrix = np.full((len(layers), len(noise_levels)), np.nan, dtype=np.float64)
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    noise_to_idx = {noise: i for i, noise in enumerate(noise_levels)}
    for row in metric_rows:
        matrix[layer_to_idx[int(row["layer"])], noise_to_idx[float(row["noise_level"])]] = float(row["value"])
    return layers, noise_levels, matrix


def plot_metric_triptych(
    sit_rows: Sequence[Dict[str, object]],
    repa_rows: Sequence[Dict[str, object]],
    metric: str,
    out_path: str | Path,
) -> None:
    sit_layers, sit_noise, sit_matrix = metric_rows_to_matrix(sit_rows, metric)
    repa_layers, repa_noise, repa_matrix = metric_rows_to_matrix(repa_rows, metric)
    if sit_layers != repa_layers or sit_noise != repa_noise:
        raise ValueError("SiT and REPA TSVs must share the same layer/noise grid")

    diff_matrix = repa_matrix - sit_matrix
    finite_values = np.concatenate([sit_matrix[np.isfinite(sit_matrix)], repa_matrix[np.isfinite(repa_matrix)]])
    if finite_values.size == 0:
        raise ValueError(f"No finite values available for metric={metric}")
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    dmax = float(np.nanmax(np.abs(diff_matrix)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    common_kwargs = dict(origin="upper", aspect="auto", interpolation="nearest")

    im0 = axes[0].imshow(sit_matrix, cmap="viridis", vmin=vmin, vmax=vmax, **common_kwargs)
    axes[0].set_title(f"SiT {metric.upper()}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(repa_matrix, cmap="viridis", vmin=vmin, vmax=vmax, **common_kwargs)
    axes[1].set_title(f"REPA {metric.upper()}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_matrix, cmap="coolwarm", vmin=-dmax, vmax=dmax, **common_kwargs)
    axes[2].set_title(f"REPA - SiT {metric.upper()}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    xlabels = [f"{noise:.2f}" for noise in sit_noise]
    ylabels = [str(layer) for layer in sit_layers]
    for ax in axes:
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(sit_noise)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(sit_layers)))
        ax.set_yticklabels(ylabels)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_metrics(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_formats(spec: str) -> List[str]:
    return [x.strip().lower() for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sit-tsv", type=str, required=True)
    parser.add_argument("--repa-tsv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="all")
    parser.add_argument("--formats", type=str, default="png,pdf")
    args = parser.parse_args()

    sit_rows = read_metrics_tsv(args.sit_tsv)
    repa_rows = read_metrics_tsv(args.repa_tsv)
    if args.metrics.strip().lower() == "all":
        sit_metrics = {str(row["metric"]) for row in sit_rows}
        repa_metrics = {str(row["metric"]) for row in repa_rows}
        metrics = sorted(sit_metrics & repa_metrics)
    else:
        metrics = parse_metrics(args.metrics)
    formats = parse_formats(args.formats)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        for ext in formats:
            plot_metric_triptych(
                sit_rows=sit_rows,
                repa_rows=repa_rows,
                metric=metric,
                out_path=outdir / f"{metric}_triptych.{ext}",
            )


if __name__ == "__main__":
    main()
