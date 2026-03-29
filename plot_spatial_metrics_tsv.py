#!/usr/bin/env python3
"""Visualize spatial metrics TSVs across arbitrary per-layer metric sets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    _turbo = matplotlib.colormaps["turbo"]
except AttributeError:
    _turbo = matplotlib.cm.get_cmap("turbo")


def schedule_label(nl: float, mt: float) -> str:
    return f"σ={nl:.2f}, t={mt:.2f}"


PREFERRED_METRIC_ORDER = (
    "lds",
    "cds",
    "rmsc",
    "lgr",
    "msdr",
    "graph_gap",
    "ubc",
    "hf_ratio",
    "patch_miou",
    "boundary_f1",
    "objectness_iou",
)
METRIC_TITLES = {
    "lds": "LDS (local distance structure)",
    "cds": "CDS (cross distance structure)",
    "rmsc": "RMSC (relative magnitude / scale)",
    "lgr": "LGR (local / global concentration)",
    "msdr": "MSDR (multi-scale detail retention)",
    "graph_gap": "Token Graph Spectral Gap",
    "ubc": "UBC (unsupervised boundary concentration)",
    "hf_ratio": "HF ratio",
    "patch_miou": "Patch mIoU",
    "boundary_f1": "Boundary F1",
    "objectness_iou": "Objectness IoU",
}


def metric_display_order(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["metric"].astype(str).tolist()))
    preferred = [metric for metric in PREFERRED_METRIC_ORDER if metric in present]
    extras = sorted(set(present) - set(preferred))
    return preferred + extras


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot spatial metrics TSV across all metrics present in the file.")
    ap.add_argument("tsv", type=Path, nargs="?", default=Path("/workspace/outputs/sit_miniimagenet_spatial_metrics/metrics.tsv"))
    ap.add_argument("--outdir", type=Path, default=None, help="Default: same dir as TSV")
    args = ap.parse_args()

    path = args.tsv
    outdir = args.outdir if args.outdir is not None else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.parent.name if path.stem == "metrics" else path.stem

    df = pd.read_csv(path, sep="\t")
    df = df.drop_duplicates()
    if "metric" not in df.columns or "layer" not in df.columns:
        raise SystemExit(f"Unexpected columns: {list(df.columns)}")

    # schedule tuples, stable order: by noise_level desc then model_t
    sched = (
        df[["noise_level", "model_t"]]
        .drop_duplicates()
        .sort_values(["noise_level", "model_t"], ascending=[False, True])
        .itertuples(index=False, name=None)
    )
    schedules = list(sched)
    nsched = len(schedules)
    if hasattr(_turbo, "resampled"):
        cmap = _turbo.resampled(max(nsched, 1))
    else:
        cmap = matplotlib.cm.get_cmap("turbo", max(nsched, 1))

    def line_color(si: int):
        if nsched <= 1:
            return cmap(0.0)
        return cmap(si / (nsched - 1))

    metrics = metric_display_order(df)
    if not metrics:
        raise SystemExit("No metric rows found in TSV.")

    # --- Figure: line plots (one panel per metric)
    n_layers = int(df["layer"].max()) + 1
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, max(3.0 * len(metrics), 4.5)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        if sub.empty:
            ax.text(0.5, 0.5, f"No rows for {metric}", ha="center", va="center", transform=ax.transAxes)
            continue
        for si, (nl, mt) in enumerate(schedules):
            s = sub[(sub["noise_level"] == nl) & (sub["model_t"] == mt)].sort_values("layer")
            layers = s["layer"].astype(int).values
            vals = s["value"].astype(float).values
            ax.plot(layers, vals, "-o", ms=3, lw=1.35, color=line_color(si), label=schedule_label(nl, mt))
        ax.set_ylabel(metric)
        ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=12, fontweight="600")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_xticks(range(0, n_layers, 2))

    axes[-1].set_xlabel("Transformer block (layer)")
    fig.suptitle(f"Spatial metrics — {path.name}", fontsize=13, fontweight="600", y=1.02)

    ncol = 4 if nsched > 8 else 3
    handles = [
        mlines.Line2D([], [], color=line_color(i), marker="o", lw=1.5, label=schedule_label(nl, mt))
        for i, (nl, mt) in enumerate(schedules)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=7,
        frameon=True,
        title="noise schedule",
    )
    fig.subplots_adjust(bottom=0.18)

    out1 = outdir / f"{stem}_lines.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}_lines.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Heatmaps: layer × schedule
    ncols = min(4, len(metrics))
    nrows = int(math.ceil(len(metrics) / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.3 * nrows), constrained_layout=True)
    axes2 = np.atleast_1d(axes2).reshape(nrows, ncols)
    sched_labels = [schedule_label(nl, mt) for nl, mt in schedules]
    x = np.arange(nsched)

    for ax, metric in zip(axes2.flat, metrics):
        sub = df[df["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        mat = np.full((n_layers, nsched), np.nan)
        for j, (nl, mt) in enumerate(schedules):
            s = sub[(sub["noise_level"] == nl) & (sub["model_t"] == mt)]
            for _, row in s.iterrows():
                li = int(row["layer"])
                if 0 <= li < mat.shape[0]:
                    mat[li, j] = float(row["value"])
        im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 7)))
        ax.set_xticks(x)
        ax.set_xticklabels(sched_labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("layer")
        ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=11, fontweight="600")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in list(axes2.flat)[len(metrics):]:
        ax.set_visible(False)
    fig2.suptitle(f"Spatial metrics (heatmap) — {path.name}", fontsize=13, fontweight="600")
    out2 = outdir / f"{stem}_heatmap.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    fig2.savefig(outdir / f"{stem}_heatmap.pdf", bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
