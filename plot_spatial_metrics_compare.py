#!/usr/bin/env python3
"""Compare two spatial metrics TSVs (REPA vs SiT) across arbitrary per-layer spatial metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS_ORDER = (
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
TITLES = {
    "lds": "LDS",
    "cds": "CDS",
    "rmsc": "RMSC",
    "lgr": "LGR",
    "msdr": "MSDR",
    "graph_gap": "Graph Gap",
    "ubc": "UBC",
    "hf_ratio": "HF Ratio",
    "patch_miou": "Patch mIoU",
    "boundary_f1": "Boundary F1",
    "objectness_iou": "Objectness IoU",
}


def load_spatial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df.drop_duplicates()


def merge_on_schedule(repa: pd.DataFrame, sit: pd.DataFrame) -> pd.DataFrame:
    r = repa[["metric", "layer", "noise_level", "value"]].rename(columns={"value": "value_repa"})
    s = sit[["metric", "layer", "noise_level", "value"]].rename(columns={"value": "value_sit"})
    m = r.merge(s, on=["metric", "layer", "noise_level"], how="inner")
    if m.empty:
        raise SystemExit("Merge produced no rows — check TSV schemas and noise_level grids.")
    return m


def noise_schedule_labels(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    """Order columns by noise_level descending (high noise → low), label σ."""
    sched = (
        df[["noise_level", "model_t"]]
        .drop_duplicates()
        .sort_values(["noise_level", "model_t"], ascending=[False, True])
    )
    out = []
    for _, row in sched.iterrows():
        nl, mt = float(row["noise_level"]), float(row["model_t"])
        out.append((nl, mt, f"σ={nl:.2f}"))
    return out


def metric_display_order(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["metric"].astype(str).tolist()))
    preferred = [metric for metric in METRICS_ORDER if metric in present]
    extras = sorted(set(present) - set(preferred))
    return preferred + extras


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot REPA vs SiT spatial metrics comparison.")
    ap.add_argument("--repa", type=Path, default=Path("/workspace/outputs/repa_miniimagenet_spatial_metrics/metrics.tsv"))
    ap.add_argument("--sit", type=Path, default=Path("/workspace/outputs/sit_miniimagenet_spatial_metrics/metrics.tsv"))
    ap.add_argument("--outdir", type=Path, default=Path("/workspace/outputs/spatial_metrics_compare_plots"))
    args = ap.parse_args()

    repa = load_spatial(args.repa)
    sit = load_spatial(args.sit)
    merged = merge_on_schedule(repa, sit)

    metrics = metric_display_order(merged)
    n_metrics = len(metrics)
    if n_metrics == 0:
        raise SystemExit("No shared metrics found after merge.")

    n_layers = int(merged["layer"].max()) + 1
    sched_labels = noise_schedule_labels(sit)
    nsched = len(sched_labels)
    noise_levels = [s[0] for s in sched_labels]

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "repa_vs_sit_spatial"

    # --- Figure 1: metric × schedule grid — each cell = one schedule, REPA vs SiT vs layer
    fig1, axes1 = plt.subplots(
        n_metrics,
        nsched,
        figsize=(2.2 * max(nsched, 1), max(2.3 * n_metrics, 3.0)),
        sharex="col",
        sharey="row",
    )
    axes1 = np.asarray(axes1)
    if axes1.ndim == 1:
        axes1 = axes1.reshape(n_metrics, -1)
    for mi, metric in enumerate(metrics):
        for j, nl in enumerate(noise_levels):
            ax = axes1[mi, j]
            sub = merged[(merged["metric"] == metric) & (merged["noise_level"] == nl)].sort_values("layer")
            if sub.empty:
                ax.set_visible(False)
                continue
            layers = sub["layer"].astype(int).values
            ax.plot(layers, sub["value_repa"], "-o", ms=2, lw=1.2, color="#2ecc71", label="REPA")
            ax.plot(layers, sub["value_sit"], "--s", ms=2, lw=1.2, color="#3498db", label="SiT")
            if mi == 0:
                ax.set_title(sched_labels[j][2], fontsize=9, fontweight="600")
            if j == 0:
                ax.set_ylabel(TITLES.get(metric, metric), fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(range(0, n_layers, 4))
            if mi == n_metrics - 1:
                ax.set_xlabel("layer", fontsize=8)
    handles = [
        plt.Line2D([0], [0], color="#2ecc71", marker="o", lw=1.5, label="REPA"),
        plt.Line2D([0], [0], color="#3498db", marker="s", ls="--", lw=1.5, label="SiT"),
    ]
    fig1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    fig1.suptitle("Spatial metrics: REPA vs SiT (by noise level σ)", fontsize=13, fontweight="600", y=1.02)
    fig1.tight_layout()
    p1 = outdir / f"{stem}_grid.png"
    fig1.savefig(p1, dpi=160, bbox_inches="tight")
    fig1.savefig(outdir / f"{stem}_grid.pdf", bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: heatmaps of (REPA − SiT)
    ncols = min(4, n_metrics)
    nrows = int(math.ceil(n_metrics / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes2 = np.atleast_1d(axes2).reshape(nrows, ncols)
    for ax, metric in zip(axes2.flat, metrics):
        mat = np.full((n_layers, nsched), np.nan)
        for j, nl in enumerate(noise_levels):
            sub = merged[(merged["metric"] == metric) & (merged["noise_level"] == nl)]
            for _, row in sub.iterrows():
                li = int(row["layer"])
                if 0 <= li < n_layers:
                    mat[li, j] = float(row["value_repa"]) - float(row["value_sit"])
        vmax = np.nanmax(np.abs(mat))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-8
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(nsched))
        ax.set_xticklabels([s[2] for s in sched_labels], rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 7)))
        ax.set_ylabel("layer")
        ax.set_title(f"{TITLES.get(metric, metric)}  (REPA − SiT)", fontsize=11, fontweight="600")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in list(axes2.flat)[len(metrics):]:
        ax.set_visible(False)
    fig2.suptitle("Difference heatmaps", fontsize=13, fontweight="600")
    fig2.tight_layout()
    p2 = outdir / f"{stem}_delta_heatmap.png"
    fig2.savefig(p2, dpi=160, bbox_inches="tight")
    fig2.savefig(outdir / f"{stem}_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
