from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

COMPONENT_COLORS = {
    "raw": "#1f2937",
    "common": "#0f766e",
    "residual": "#b91c1c",
}

FAMILY_COLORS = {
    "mean": "#0f766e",
    "tsvd_K16": "#2563eb",
    "tsvd_K32": "#7c3aed",
    "tsvd_K64": "#ea580c",
}

METRIC_LABELS = {
    "lds": "LDS",
    "cds": "CDS",
    "rmsc": "RMSC",
}

COMPONENT_LABELS = {
    "raw": "Raw",
    "common": "Common",
    "residual": "Residual",
}


def _ordered_families(families: Iterable[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        if name == "mean":
            return (0, name)
        if name.startswith("tsvd_K"):
            return (1, name)
        return (2, name)

    return sorted(set(families), key=key)


def _metric_grid(df: pd.DataFrame) -> np.ndarray:
    grouped = (
        df.groupby(["component", "layer", "time_position"])["value"]
        .mean()
        .reset_index()
    )
    components = ["raw", "common", "residual"]
    layers = np.sort(grouped["layer"].unique())
    times = np.sort(grouped["time_position"].unique())
    out = np.zeros((len(components), len(layers), len(times)), dtype=np.float32)
    for c_idx, component in enumerate(components):
        subset = grouped[grouped["component"] == component]
        pivot = subset.pivot(index="layer", columns="time_position", values="value").reindex(index=layers, columns=times)
        out[c_idx] = pivot.to_numpy(dtype=np.float32)
    return out


def _curve_stats(df: pd.DataFrame, axis: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped = (
        df.groupby(["component", "layer", "time_position"])["value"]
        .mean()
        .reset_index()
    )
    stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if axis == "layer":
        x_name = "layer"
        other_name = "time_position"
    elif axis == "time":
        x_name = "time_position"
        other_name = "layer"
    else:
        raise ValueError(axis)
    x_values = np.sort(grouped[x_name].unique())
    for component in ("raw", "common", "residual"):
        subset = grouped[grouped["component"] == component]
        pivot = subset.pivot(index=x_name, columns=other_name, values="value").reindex(index=x_values)
        values = pivot.to_numpy(dtype=np.float32)
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        stats[component] = (x_values.astype(np.float32), mean, std)
    return stats


def _format_family_label(family: str) -> str:
    return "Mean Common" if family == "mean" else family.replace("_", " ")


def save_task4_overview_curves(
    family_metric_tables: dict[str, dict[str, pd.DataFrame]],
    out_path: Path,
    *,
    title: str,
    axis: str,
    time_value_labels: list[str] | None = None,
) -> None:
    families = _ordered_families(family_metric_tables.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False)
    for row_idx, metric in enumerate(("lds", "cds", "rmsc")):
        for col_idx, component in enumerate(("raw", "common", "residual")):
            ax = axes[row_idx, col_idx]
            for family in families:
                if component == "raw" and family != "mean":
                    continue
                x, mean, std = _curve_stats(family_metric_tables[family][metric], axis)[component]
                if axis == "time" and time_value_labels is not None:
                    x_plot = np.arange(len(x))
                else:
                    x_plot = x
                label = "Raw" if component == "raw" else _format_family_label(family)
                color = COMPONENT_COLORS["raw"] if component == "raw" else FAMILY_COLORS.get(family, "#374151")
                ax.plot(x_plot, mean, label=label, color=color, linewidth=2.2)
                ax.fill_between(x_plot, mean - std, mean + std, color=color, alpha=0.12)
            ax.set_title(f"{METRIC_LABELS[metric]} | {COMPONENT_LABELS[component]}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.22)
            if row_idx == 2:
                ax.set_xlabel("Layer" if axis == "layer" else "Timestep")
            if col_idx == 0:
                ax.set_ylabel("Metric value")
            if axis == "time" and time_value_labels is not None:
                ticks = np.arange(len(time_value_labels))
                ax.set_xticks(ticks)
                ax.set_xticklabels(time_value_labels, rotation=0, fontsize=8)
            if row_idx == 0:
                ax.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def save_task4_family_figures(
    family: str,
    metric_tables: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    title_prefix: str,
    time_value_labels: list[str] | None = None,
) -> None:
    family_label = _format_family_label(family)
    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=False)
        for row_idx, metric in enumerate(("lds", "cds", "rmsc")):
            layer_stats = _curve_stats(metric_tables[metric], "layer")
            time_stats = _curve_stats(metric_tables[metric], "time")
            for component in ("raw", "common", "residual"):
                x, mean, std = layer_stats[component]
                ax = axes[row_idx, 0]
                color = COMPONENT_COLORS[component]
                ax.plot(x, mean, color=color, linewidth=2.2, label=COMPONENT_LABELS[component])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)
                ax.grid(True, alpha=0.22)
                ax.set_title(f"{METRIC_LABELS[metric]} by Layer", fontsize=11, fontweight="bold")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Metric value")

                tx, tmean, tstd = time_stats[component]
                tx_plot = np.arange(len(tx))
                ax_t = axes[row_idx, 1]
                ax_t.plot(tx_plot, tmean, color=color, linewidth=2.2, label=COMPONENT_LABELS[component])
                ax_t.fill_between(tx_plot, tmean - tstd, tmean + tstd, color=color, alpha=0.14)
                ax_t.grid(True, alpha=0.22)
                ax_t.set_title(f"{METRIC_LABELS[metric]} by Timestep", fontsize=11, fontweight="bold")
                ax_t.set_xlabel("Timestep")
                ax_t.set_ylabel("Metric value")
                if time_value_labels is not None:
                    ax_t.set_xticks(tx_plot)
                    ax_t.set_xticklabels(time_value_labels, fontsize=8)

        axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
        axes[0, 1].legend(loc="best", fontsize=8, frameon=False)
        fig.suptitle(f"{title_prefix} | {family_label} | Summary Curves", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
        for row_idx, metric in enumerate(("lds", "cds", "rmsc")):
            grids = _metric_grid(metric_tables[metric])
            vmin = float(np.nanmin(grids))
            vmax = float(np.nanmax(grids))
            for col_idx, component in enumerate(("raw", "common", "residual")):
                ax = axes[row_idx, col_idx]
                image = ax.imshow(grids[col_idx], aspect="auto", cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
                ax.set_title(f"{METRIC_LABELS[metric]} | {COMPONENT_LABELS[component]}", fontsize=11, fontweight="bold")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Layer")
                if time_value_labels is not None:
                    ticks = np.arange(len(time_value_labels))
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(time_value_labels, fontsize=8)
                if row_idx == 0:
                    ax.set_yticks(np.linspace(0, grids.shape[1] - 1, 7, dtype=int))
                    ax.set_yticklabels([str(v + 1) for v in np.linspace(0, grids.shape[1] - 1, 7, dtype=int)])
            cbar = fig.colorbar(image, ax=axes[row_idx, :], shrink=0.8, location="right")
            cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{title_prefix} | {family_label} | Layer-Timestep Heatmaps", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)


def load_task4_metric_tables(input_dir: Path, prefix: str) -> dict[str, dict[str, pd.DataFrame]]:
    metric_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for path in sorted(input_dir.glob(f"{prefix}_*_lds.csv")):
        stem = path.stem
        family = stem[len(prefix) + 1 : -len("_lds")]
        metric_tables.setdefault(family, {})
        for metric in ("lds", "cds", "rmsc"):
            metric_path = input_dir / f"{prefix}_{family}_{metric}.csv"
            if not metric_path.exists():
                raise FileNotFoundError(metric_path)
            metric_tables[family][metric] = pd.read_csv(metric_path)
    if not metric_tables:
        raise FileNotFoundError(f"No CSV files found for prefix {prefix} in {input_dir}")
    return metric_tables
