#!/usr/bin/env python3
"""Compare two metrics.tsv (REPA vs SiT) into readable figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


SPATIAL_LAYER_METRICS = ("mad", "entropy", "decay_slope", "hf_ratio")
REP_LAYER_METRICS = ("linear_top1", "linear_top5", "knn_top1", "knn_recall_at_k", "cka", "cknna")
GEOMETRY_LAYER_METRICS = ("nc1", "ncm_acc", "etf_dev", "participation_ratio", "effective_rank")
LEGACY_SCALAR_METRICS = ("linear_top1", "linear_top5", "knn_top1", "knn_recall_at_k", "cka", "cknna")


def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df.drop_duplicates()


def merge_pair(repa: pd.DataFrame, sit: pd.DataFrame) -> pd.DataFrame:
    return repa.merge(sit, on=["metric", "layer", "timestep"], suffixes=("_repa", "_sit"), how="outer")


def plot_layer_metrics(ax, sub: pd.DataFrame, title: str, ylabel: str, *, log_y: bool = False) -> None:
    sub = sub.dropna(subset=["value_repa", "value_sit"]).sort_values("layer_num")
    layers = sub["layer_num"].to_numpy()
    ax.plot(layers, sub["value_repa"], label="REPA", color="#2ecc71", marker="o", linewidth=2, markersize=3)
    ax.plot(layers, sub["value_sit"], label="SiT", color="#3498db", marker="s", linewidth=2, markersize=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="600")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(alpha=0.35, linestyle="--")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    if log_y:
        vals = np.concatenate([sub["value_repa"].to_numpy(), sub["value_sit"].to_numpy()])
        positive = vals[np.isfinite(vals) & (vals > 0)]
        if positive.size > 0:
            span = positive.max() / max(float(positive.min()), 1e-12)
            if span >= 10.0:
                ax.set_yscale("log")


def plot_scalar_metrics(ax, sub: pd.DataFrame) -> None:
    sub = sub.copy()
    sub["label"] = sub["metric"] + " (" + sub["layer"].astype(str) + ")"
    metrics = sub["label"].tolist()
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w / 2, sub["value_repa"], width=w, label="REPA", color="#2ecc71", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, sub["value_sit"], width=w, label="SiT", color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("Probes & CKA", fontsize=11, fontweight="600")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.35, linestyle="--")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot REPA vs SiT metrics.tsv comparison.")
    p.add_argument("--repa", type=Path, default=Path("/workspace/outputs/repa_imagenet_metrics/metrics.tsv"))
    p.add_argument("--sit", type=Path, default=Path("/workspace/outputs/sit_imagenet_metrics/metrics.tsv"))
    p.add_argument("--outdir", type=Path, default=Path("/workspace/outputs/metrics_compare_plots"))
    args = p.parse_args()

    repa = load_tsv(args.repa)
    sit = load_tsv(args.sit)
    m = merge_pair(repa, sit)

    m["layer_num"] = pd.to_numeric(m["layer"], errors="coerce")
    m["layer_str"] = m["layer"].astype(str)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "repa_vs_sit_metrics"

    # --- Figure 1: layer-wise metrics (2x2)
    fig1, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    fig1.suptitle("ImageNet metrics: REPA vs SiT (by transformer depth)", fontsize=13, fontweight="600")

    titles = {
        "mad": "MAD (token distance)",
        "entropy": "Attention entropy",
        "decay_slope": "Similarity–distance decay slope",
        "hf_ratio": "High-frequency ratio",
    }
    ylabels = {
        "mad": "MAD",
        "entropy": "entropy (nats)",
        "decay_slope": "slope",
        "hf_ratio": "HF ratio",
    }

    for ax, metric in zip(axes.ravel(), SPATIAL_LAYER_METRICS):
        sub = m[(m["metric"] == metric) & m["layer_num"].notna()].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data: {metric}", ha="center", va="center", transform=ax.transAxes)
            continue
        plot_layer_metrics(ax, sub, titles[metric], ylabels[metric])

    fig1.savefig(outdir / f"{stem}_layers.png", dpi=160)
    fig1.savefig(outdir / f"{stem}_layers.pdf")
    plt.close(fig1)

    # --- Figure 2: layer-wise representation metrics
    rep_titles = {
        "linear_top1": "Linear probe top-1",
        "linear_top5": "Linear probe top-5",
        "knn_top1": "k-NN top-1",
        "knn_recall_at_k": "k-NN recall@k",
        "cka": "CKA vs DINO",
        "cknna": "CKNNA vs DINO",
    }
    rep_ylabels = {
        "linear_top1": "acc",
        "linear_top5": "acc",
        "knn_top1": "acc",
        "knn_recall_at_k": "recall",
        "cka": "CKA",
        "cknna": "CKNNA",
    }
    geometry_titles = {
        "nc1": "NC1",
        "ncm_acc": "NCM top-1",
        "etf_dev": "ETF deviation",
        "participation_ratio": "Participation ratio",
        "effective_rank": "Effective rank",
    }
    geometry_ylabels = {
        "nc1": "NC1",
        "ncm_acc": "acc",
        "etf_dev": "ETF dev",
        "participation_ratio": "PR",
        "effective_rank": "rank",
    }
    rep_metrics_present = [
        metric
        for metric in REP_LAYER_METRICS
        if not m[(m["metric"] == metric) & m["layer_num"].notna()].empty
    ]
    geometry_metrics_present = [
        metric
        for metric in GEOMETRY_LAYER_METRICS
        if not m[(m["metric"] == metric) & m["layer_num"].notna()].empty
    ]

    if rep_metrics_present:
        ncols = 2
        nrows = int(np.ceil(len(rep_metrics_present) / ncols))
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows), constrained_layout=True)
        axes2 = np.atleast_1d(axes2).ravel()
        fig2.suptitle("Linear / k-NN probes & alignment", fontsize=13, fontweight="600")
        for ax, metric in zip(axes2, rep_metrics_present):
            sub = m[(m["metric"] == metric) & m["layer_num"].notna()].copy()
            plot_layer_metrics(ax, sub, rep_titles[metric], rep_ylabels[metric])
            if metric in {"linear_top1", "linear_top5", "knn_top1", "knn_recall_at_k", "cka", "cknna"}:
                ax.set_ylim(0, 1.05)
        for ax in axes2[len(rep_metrics_present):]:
            ax.set_visible(False)
        fig2.savefig(outdir / f"{stem}_probes.png", dpi=160)
        fig2.savefig(outdir / f"{stem}_probes.pdf")
        plt.close(fig2)
    else:
        legacy_scalars = m[m["metric"].isin(LEGACY_SCALAR_METRICS) & m["layer_num"].isna()].copy()
        if not legacy_scalars.empty:
            fig2, ax = plt.subplots(1, 1, figsize=(11, 4.2), constrained_layout=True)
            fig2.suptitle("Linear / k-NN probes & alignment", fontsize=13, fontweight="600")
            plot_scalar_metrics(ax, legacy_scalars.dropna(subset=["value_repa", "value_sit"]))
            fig2.savefig(outdir / f"{stem}_probes.png", dpi=160)
            fig2.savefig(outdir / f"{stem}_probes.pdf")
            plt.close(fig2)

    # --- Figure 3: layer-wise class geometry metrics
    if geometry_metrics_present:
        ncols = 2
        nrows = int(np.ceil(len(geometry_metrics_present) / ncols))
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows), constrained_layout=True)
        axes3 = np.atleast_1d(axes3).ravel()
        fig3.suptitle("Class geometry", fontsize=13, fontweight="600")
        for ax, metric in zip(axes3, geometry_metrics_present):
            sub = m[(m["metric"] == metric) & m["layer_num"].notna()].copy()
            plot_layer_metrics(
                ax,
                sub,
                geometry_titles[metric],
                geometry_ylabels[metric],
                log_y=metric in {"nc1", "etf_dev"},
            )
            if metric == "ncm_acc":
                ax.set_ylim(0, 1.05)
        for ax in axes3[len(geometry_metrics_present):]:
            ax.set_visible(False)
        fig3.savefig(outdir / f"{stem}_geometry.png", dpi=160)
        fig3.savefig(outdir / f"{stem}_geometry.pdf")
        plt.close(fig3)

    # --- Figure 4: one-page overview (compact): 2x2 bars + full-width table
    fig4 = plt.figure(figsize=(10, 11), constrained_layout=True)
    gs = fig4.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85])
    fig4.suptitle("REPA vs SiT — overview", fontsize=13, fontweight="600")
    for i, metric in enumerate(SPATIAL_LAYER_METRICS):
        r, c = divmod(i, 2)
        ax = fig4.add_subplot(gs[r, c])
        sub = m[(m["metric"] == metric) & m["layer_num"].notna()].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        plot_layer_metrics(ax, sub, titles[metric], ylabels[metric])

    ax_tbl = fig4.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    rows = []
    scal = m[m["metric"].isin(LEGACY_SCALAR_METRICS) & m["layer_num"].isna()].copy()
    for _, r in scal.sort_values(["metric", "layer"]).iterrows():
        rows.append([r["metric"], str(r["layer"]), f'{r["value_repa"]:.4f}', f'{r["value_sit"]:.4f}'])
    if rows:
        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=["metric", "layer", "REPA", "SiT"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.15, 1.35)
        ax_tbl.set_title("Scalar metrics", fontsize=11, fontweight="600", pad=12)
    else:
        ax_tbl.text(
            0.5,
            0.5,
            "Per-layer probe/alignment and class-geometry metrics are saved in separate figures.",
            ha="center",
            va="center",
            fontsize=10,
        )

    fig4.savefig(outdir / f"{stem}_overview.png", dpi=160)
    fig4.savefig(outdir / f"{stem}_overview.pdf")
    plt.close(fig4)

    print(f"Wrote PNG/PDF to {outdir}/")


if __name__ == "__main__":
    main()
