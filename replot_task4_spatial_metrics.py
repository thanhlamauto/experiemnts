#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_sit_protocol.spatial_plots import (
    load_task4_metric_tables,
    save_task4_family_figures,
    save_task4_overview_curves,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate readable Task 4 / Task 4B spatial-metric PDFs from CSV files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing task4_*.csv or task4b_spatialnorm_*.csv files.")
    parser.add_argument(
        "--prefix",
        type=str,
        default="task4",
        choices=("task4", "task4b_spatialnorm"),
        help="CSV prefix to replot.",
    )
    parser.add_argument(
        "--time-labels",
        type=str,
        default="0.00,0.11,0.22,0.33,0.44,0.56,0.67,0.78,0.89,1.00",
        help="Comma-separated labels for timestep positions.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    prefix = args.prefix
    title_prefix = "Task 4" if prefix == "task4" else "Task 4B Spatial-Norm"
    time_value_labels = [part.strip() for part in args.time_labels.split(",") if part.strip()]

    tables_by_family = load_task4_metric_tables(input_dir, prefix)
    save_task4_overview_curves(
        tables_by_family,
        input_dir / f"{prefix}_layerwise_curves_pretty.pdf",
        title=f"{title_prefix} Layer-wise Overview",
        axis="layer",
    )
    save_task4_overview_curves(
        tables_by_family,
        input_dir / f"{prefix}_timestep_curves_pretty.pdf",
        title=f"{title_prefix} Timestep-wise Overview",
        axis="time",
        time_value_labels=time_value_labels,
    )
    for family_name, metric_tables in tables_by_family.items():
        save_task4_family_figures(
            family_name,
            metric_tables,
            input_dir / f"{prefix}_{family_name}_figures_pretty.pdf",
            title_prefix=title_prefix,
            time_value_labels=time_value_labels,
        )


if __name__ == "__main__":
    main()
