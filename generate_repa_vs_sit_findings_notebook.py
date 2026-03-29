#!/usr/bin/env python3
"""Build and execute a notebook that summarizes REPA vs SiT findings from /workspace/outputs."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter


OUT_IPYNB = Path("/workspace/outputs/repa_vs_sit_findings.ipynb")
OUT_HTML = Path("/workspace/outputs/repa_vs_sit_findings.html")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python3 (main venv)",
        "language": "python",
        "name": "main",
    }
    nb.metadata["language_info"] = {"name": "python"}

    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # REPA vs SiT-XL/2 Findings from `/workspace/outputs`

                This notebook inventories the outputs, reads the main TSV files, groups metrics into
                `coarse`, `fine`, and `spatial`, then ranks them with an oriented gain score where
                positive means `REPA` is better than `SiT vanilla`.

                Main quantitative sources:
                - `/workspace/outputs/sit_imagenet_metrics/metrics.tsv`
                - `/workspace/outputs/repa_imagenet_metrics/metrics.tsv`
                - `/workspace/outputs/sit_imagenet_spatial_metrics/metrics.tsv`
                - `/workspace/outputs/repa_imagenet_spatial_metrics/metrics.tsv`

                Main visual sources:
                - `/workspace/outputs/metrics_compare_plots/*.png`
                - `/workspace/outputs/spatial_metrics_compare_plots/*.png`
                - `/workspace/outputs/activation_heatmap_compare/sit_vs_repa_activations.png`
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path
                import warnings

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                from IPython.display import Markdown, display

                warnings.filterwarnings("ignore")
                plt.style.use("seaborn-v0_8-whitegrid")
                sns.set_context("talk", font_scale=0.9)
                pd.set_option("display.max_rows", 200)
                pd.set_option("display.max_colwidth", 140)

                OUTPUTS = Path("/workspace/outputs")
                MAIN_FILES = {
                    "coarse_sit": OUTPUTS / "sit_imagenet_metrics" / "metrics.tsv",
                    "coarse_repa": OUTPUTS / "repa_imagenet_metrics" / "metrics.tsv",
                    "spatial_sit": OUTPUTS / "sit_imagenet_spatial_metrics" / "metrics.tsv",
                    "spatial_repa": OUTPUTS / "repa_imagenet_spatial_metrics" / "metrics.tsv",
                }
                GALLERY = [
                    OUTPUTS / "metrics_compare_plots" / "repa_vs_sit_metrics_geometry.png",
                    OUTPUTS / "metrics_compare_plots" / "repa_vs_sit_metrics_probes.png",
                    OUTPUTS / "spatial_metrics_compare_plots" / "repa_vs_sit_spatial_delta_heatmap.png",
                    OUTPUTS / "activation_heatmap_compare" / "sit_vs_repa_activations.png",
                ]

                all_output_files = sorted([p for p in OUTPUTS.rglob("*") if p.is_file()])
                inventory = pd.DataFrame(
                    {
                        "path": [str(p) for p in all_output_files],
                        "ext": [p.suffix.lower() for p in all_output_files],
                        "size_kb": [round(p.stat().st_size / 1024.0, 1) for p in all_output_files],
                    }
                )
                inventory["kind"] = inventory["ext"].map(
                    {
                        ".tsv": "table",
                        ".png": "image",
                        ".pdf": "figure",
                        ".npy": "array",
                        ".npz": "archive",
                    }
                ).fillna("other")
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 1. Output inventory"))
                inventory_summary = (
                    inventory.groupby(["kind", "ext"]).size().reset_index(name="count").sort_values(["kind", "ext"])
                )
                display(inventory_summary)
                display(inventory)
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                coarse_sit = pd.read_csv(MAIN_FILES["coarse_sit"], sep="\\t")
                coarse_repa = pd.read_csv(MAIN_FILES["coarse_repa"], sep="\\t")
                spatial_sit = pd.read_csv(MAIN_FILES["spatial_sit"], sep="\\t")
                spatial_repa = pd.read_csv(MAIN_FILES["spatial_repa"], sep="\\t")

                coarse = coarse_repa.merge(
                    coarse_sit,
                    on=["metric", "layer", "timestep"],
                    suffixes=("_repa", "_sit"),
                )
                spatial = spatial_repa.merge(
                    spatial_sit,
                    on=["metric", "layer", "noise_level"],
                    suffixes=("_repa", "_sit"),
                )

                higher_better = {
                    "linear_top1": True,
                    "linear_top5": True,
                    "knn_top1": True,
                    "knn_recall_at_k": True,
                    "cka": True,
                    "cknna": True,
                    "nc1": False,
                    "ncm_acc": True,
                    "etf_dev": False,
                    "lds": True,
                    "cds": True,
                    "rmsc": True,
                    "lgr": True,
                    "msdr": True,
                    "graph_gap": True,
                    "ubc": True,
                    "hf_ratio": True,
                    "patch_miou": True,
                    "boundary_f1": True,
                    "objectness_iou": True,
                }

                coarse_primary = [
                    "linear_top1",
                    "linear_top5",
                    "knn_top1",
                    "knn_recall_at_k",
                    "cka",
                    "cknna",
                    "nc1",
                    "ncm_acc",
                    "etf_dev",
                ]
                coarse_diagnostic = ["participation_ratio", "effective_rank"]
                spatial_primary = ["lds", "cds", "rmsc", "lgr", "msdr", "graph_gap", "ubc", "hf_ratio"]
                spatial_diagnostic = ["mad", "entropy", "decay_slope"]
                fine_primary = ["patch_miou", "boundary_f1", "objectness_iou"]


                def add_gain(df):
                    out = df.copy()
                    out["value_repa"] = out["value_repa"].astype(float)
                    out["value_sit"] = out["value_sit"].astype(float)
                    out["delta"] = out["value_repa"] - out["value_sit"]
                    out["oriented_gain"] = out.apply(
                        lambda r: r["delta"] if higher_better.get(r["metric"], True) else -r["delta"],
                        axis=1,
                    )
                    return out


                def summarize_compare(df):
                    rows = []
                    for metric, sub in df.groupby("metric"):
                        rows.append(
                            {
                                "metric": metric,
                                "mean_repa": float(sub["value_repa"].mean()),
                                "mean_sit": float(sub["value_sit"].mean()),
                                "mean_delta": float(sub["delta"].mean()),
                                "mean_abs_delta": float(sub["delta"].abs().mean()),
                                "mean_oriented_gain": float(sub["oriented_gain"].mean()),
                                "win_rate": float((sub["oriented_gain"] > 0).mean()),
                                "best_gain": float(sub["oriented_gain"].max()),
                                "worst_gain": float(sub["oriented_gain"].min()),
                                "num_points": int(len(sub)),
                            }
                        )
                    return pd.DataFrame(rows).sort_values("mean_oriented_gain", ascending=False)


                def layer_band(layer):
                    layer = int(layer)
                    if layer <= 8:
                        return "early"
                    if layer <= 18:
                        return "mid"
                    return "late"


                coarse = add_gain(coarse)
                spatial = add_gain(spatial)
                coarse["layer_band"] = coarse["layer"].map(layer_band)
                spatial["layer_band"] = spatial["layer"].map(layer_band)

                coarse_summary = summarize_compare(coarse)
                spatial_summary = summarize_compare(spatial)
                category_tables = {
                    "coarse": coarse_summary[coarse_summary["metric"].isin(coarse_primary)].copy(),
                    "fine": spatial_summary[spatial_summary["metric"].isin(fine_primary)].copy(),
                    "spatial": spatial_summary[spatial_summary["metric"].isin(spatial_primary)].copy(),
                    "coarse_diagnostic": coarse_summary[coarse_summary["metric"].isin(coarse_diagnostic)].copy(),
                    "spatial_diagnostic_t05": coarse_summary[coarse_summary["metric"].isin(spatial_diagnostic)].copy(),
                }
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 2. Visual survey of existing figures"))
                fig, axes = plt.subplots(2, 2, figsize=(18, 14))
                for ax, path in zip(axes.ravel(), GALLERY):
                    img = plt.imread(path)
                    ax.imshow(img)
                    ax.set_title(path.name, fontsize=13)
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 3. Ranking tables by category"))
                for name in ["coarse", "fine", "spatial"]:
                    display(Markdown(f"### {name.capitalize()}"))
                    table = category_tables[name][
                        ["metric", "mean_oriented_gain", "win_rate", "mean_delta", "num_points"]
                    ].copy()
                    table["mean_oriented_gain"] = table["mean_oriented_gain"].round(4)
                    table["win_rate"] = table["win_rate"].round(4)
                    table["mean_delta"] = table["mean_delta"].round(4)
                    display(table)

                display(Markdown("### Coarse diagnostics"))
                coarse_diag = category_tables["coarse_diagnostic"][
                    ["metric", "mean_delta", "win_rate", "mean_abs_delta"]
                ].copy()
                coarse_diag["mean_delta"] = coarse_diag["mean_delta"].round(4)
                coarse_diag["win_rate"] = coarse_diag["win_rate"].round(4)
                coarse_diag["mean_abs_delta"] = coarse_diag["mean_abs_delta"].round(4)
                display(coarse_diag)

                display(Markdown("### Spatial diagnostics at timestep 0.5"))
                spatial_diag = category_tables["spatial_diagnostic_t05"][["metric", "mean_delta", "mean_abs_delta"]].copy()
                spatial_diag["mean_delta"] = spatial_diag["mean_delta"].round(4)
                spatial_diag["mean_abs_delta"] = spatial_diag["mean_abs_delta"].round(4)
                display(spatial_diag)
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 4. Ranking bars"))
                fig, axes = plt.subplots(1, 3, figsize=(22, 6))
                for ax, name, color in zip(
                    axes,
                    ["coarse", "fine", "spatial"],
                    ["#2e8b57", "#b85c38", "#3b6ea5"],
                ):
                    sub = category_tables[name].sort_values("mean_oriented_gain", ascending=True)
                    ax.barh(sub["metric"], sub["mean_oriented_gain"], color=color, alpha=0.9)
                    ax.axvline(0.0, color="black", linewidth=1)
                    ax.set_title(f"{name.capitalize()} ranking by oriented gain")
                    ax.set_xlabel("mean oriented gain (REPA better > 0)")
                plt.tight_layout()
                plt.show()
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 5. Early / Mid / Late layer summary"))
                fig, axes = plt.subplots(1, 3, figsize=(22, 7))
                for ax, title, df, metrics in [
                    (axes[0], "Coarse", coarse, coarse_primary),
                    (axes[1], "Fine", spatial, fine_primary),
                    (axes[2], "Spatial", spatial, spatial_primary),
                ]:
                    pivot = (
                        df[df["metric"].isin(metrics)]
                        .groupby(["metric", "layer_band"])["oriented_gain"]
                        .mean()
                        .unstack("layer_band")
                    )
                    pivot = pivot[["early", "mid", "late"]]
                    sns.heatmap(pivot, cmap="RdBu_r", center=0.0, annot=True, fmt=".3f", ax=ax)
                    ax.set_title(f"{title}: mean oriented gain by layer band")
                    ax.set_xlabel("layer band")
                    ax.set_ylabel("metric")
                plt.tight_layout()
                plt.show()
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 6. Noise profile for fine and spatial metrics"))
                fig, axes = plt.subplots(1, 2, figsize=(22, 6), sharex=True)
                for metric in fine_primary:
                    sub = spatial[spatial["metric"] == metric].groupby("noise_level")["oriented_gain"].mean().reset_index()
                    axes[0].plot(sub["noise_level"], sub["oriented_gain"], marker="o", linewidth=2, label=metric)
                axes[0].axhline(0.0, color="black", linewidth=1)
                axes[0].set_title("Fine metrics vs noise")
                axes[0].set_xlabel("noise level")
                axes[0].set_ylabel("mean oriented gain")
                axes[0].legend()

                for metric in spatial_primary:
                    sub = spatial[spatial["metric"] == metric].groupby("noise_level")["oriented_gain"].mean().reset_index()
                    axes[1].plot(sub["noise_level"], sub["oriented_gain"], marker="o", linewidth=2, label=metric)
                axes[1].axhline(0.0, color="black", linewidth=1)
                axes[1].set_title("Spatial metrics vs noise")
                axes[1].set_xlabel("noise level")
                axes[1].set_ylabel("mean oriented gain")
                axes[1].legend(ncol=2, fontsize=10)
                plt.tight_layout()
                plt.show()
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 7. Peak gains and hardest failures"))


                def peak_rows(df, metrics, extra_col):
                    rows = []
                    for metric in metrics:
                        sub = df[df["metric"] == metric].copy()
                        best_idx = sub["oriented_gain"].idxmax()
                        worst_idx = sub["oriented_gain"].idxmin()
                        best = sub.loc[best_idx]
                        worst = sub.loc[worst_idx]
                        rows.append(
                            {
                                "metric": metric,
                                f"best_{extra_col}": best[extra_col],
                                "best_layer": int(best["layer"]),
                                "best_gain": float(best["oriented_gain"]),
                                f"worst_{extra_col}": worst[extra_col],
                                "worst_layer": int(worst["layer"]),
                                "worst_gain": float(worst["oriented_gain"]),
                            }
                        )
                    return pd.DataFrame(rows)


                coarse_peaks = peak_rows(coarse, coarse_primary, "timestep")
                spatial_peaks = peak_rows(spatial, fine_primary + spatial_primary, "noise_level")

                display(Markdown("### Coarse peaks"))
                display(coarse_peaks.round(4))

                display(Markdown("### Fine + spatial peaks"))
                display(spatial_peaks.round(4))
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                '''
                display(Markdown("## 8. Findings"))

                coarse_rank = category_tables["coarse"].sort_values("mean_oriented_gain", ascending=False).reset_index(drop=True)
                fine_rank = category_tables["fine"].sort_values("mean_oriented_gain", ascending=False).reset_index(drop=True)
                spatial_rank = category_tables["spatial"].sort_values("mean_oriented_gain", ascending=False).reset_index(drop=True)

                findings_md = f"""
                ### Executive summary

                - **Coarse:** REPA's strongest coarse wins are **{coarse_rank.loc[0, 'metric']}**, **{coarse_rank.loc[1, 'metric']}**, and **{coarse_rank.loc[2, 'metric']}**.
                  On this run, coarse gains concentrate most in **mid layers**, especially around `ncm_acc`, `etf_dev`, `cka`, and `cknna`.
                - **Fine:** all three fine metrics favor REPA overall. The ranking is **{fine_rank.loc[0, 'metric']} > {fine_rank.loc[1, 'metric']} > {fine_rank.loc[2, 'metric']}**
                  by mean oriented gain. Fine wins are strongest in **early-to-mid layers**, with `objectness_iou` standing out most clearly.
                - **Spatial:** the strongest spatial win is **{spatial_rank.loc[0, 'metric']}**, followed by **{spatial_rank.loc[1, 'metric']}** and **{spatial_rank.loc[2, 'metric']}**.
                  The overall pattern is that REPA improves **local concentration / detail retention**, but not every graph or edge-based metric moves in the same direction.

                ### Coarse findings

                - `ncm_acc` and `etf_dev` clearly favor REPA, while `linear_top1` and `linear_top5` are so saturated that they are weak discriminators here.
                - `nc1` does **not** improve uniformly: on average REPA is slightly worse, even though a mid-layer region still favors REPA.
                - `participation_ratio` and `effective_rank` are much larger for REPA, which suggests REPA is **less collapsed / more spread out**, not simply more compressed.

                ### Fine findings

                - `objectness_iou` is the clearest fine-grained REPA win.
                - `boundary_f1` also improves quite consistently, with a high pointwise win-rate.
                - `patch_miou` is positive over most layer/noise settings, but the gain shrinks at very high noise and can occasionally flip sign there.

                ### Spatial findings

                - `lgr` is the strongest spatial signal in favor of REPA and stays positive across most noise levels.
                - `msdr`, `lds`, and `hf_ratio` also lean positive for REPA, especially from low to mid noise.
                - `graph_gap` and `ubc` are not favorable on average, so the cleanest interpretation is that REPA improves **detail/locality**, but not uniformly **graph partition quality** or **boundary concentration**.

                ### Layer-wise reading

                - REPA's **coarse** wins are strongest in **mid layers**.
                - REPA's **fine** wins are strongest in **early-to-mid layers** and then weaken in late layers.
                - REPA's **spatial** pattern is mixed: `lgr` is strongest early, while `rmsc` tilts more to late layers.
                """

                display(Markdown(findings_md))
                '''
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(Markdown("## 9. Save compact report tables"))
                report_dir = OUTPUTS / "findings_report_assets"
                report_dir.mkdir(parents=True, exist_ok=True)
                coarse_summary.to_csv(report_dir / "coarse_summary.tsv", sep="\\t", index=False)
                spatial_summary.to_csv(report_dir / "spatial_summary.tsv", sep="\\t", index=False)
                category_tables["coarse"].to_csv(report_dir / "coarse_ranking.tsv", sep="\\t", index=False)
                category_tables["fine"].to_csv(report_dir / "fine_ranking.tsv", sep="\\t", index=False)
                category_tables["spatial"].to_csv(report_dir / "spatial_ranking.tsv", sep="\\t", index=False)
                coarse_peaks.to_csv(report_dir / "coarse_peaks.tsv", sep="\\t", index=False)
                spatial_peaks.to_csv(report_dir / "fine_spatial_peaks.tsv", sep="\\t", index=False)
                display(Markdown(f"Saved summary TSVs to `{report_dir}`"))
                """
            )
        ),
    ]

    nb.cells = cells
    OUT_IPYNB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_IPYNB)

    client = NotebookClient(nb, timeout=1200, kernel_name="main", resources={"metadata": {"path": "/workspace"}})
    client.execute()
    nbf.write(nb, OUT_IPYNB)

    html_exporter = HTMLExporter(template_name="lab")
    html, _ = html_exporter.from_notebook_node(nb)
    OUT_HTML.write_text(html, encoding="utf-8")

    print(OUT_IPYNB)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
