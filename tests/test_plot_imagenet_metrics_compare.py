from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]


def _write_metrics_tsv(path: Path, repa_offset: float) -> None:
    rows = [
        ["metric", "layer", "timestep", "value", "extra"],
        ["mad", 0, 0.5, 0.10 + repa_offset, ""],
        ["mad", 1, 0.5, 0.20 + repa_offset, ""],
        ["linear_top1", 0, 0.5, 0.70 + repa_offset, ""],
        ["linear_top1", 1, 0.5, 0.80 + repa_offset, ""],
        ["nc1", 0, 0.5, 0.90 + repa_offset, "pool=mean_tokens"],
        ["nc1", 1, 0.5, 0.30 + repa_offset, "pool=mean_tokens"],
        ["ncm_acc", 0, 0.5, 0.60 + repa_offset, "pool=mean_tokens"],
        ["ncm_acc", 1, 0.5, 0.85 + repa_offset, "pool=mean_tokens"],
        ["effective_rank", 0, 0.5, 8.0 + repa_offset, "pool=mean_tokens"],
        ["effective_rank", 1, 0.5, 6.0 + repa_offset, "pool=mean_tokens"],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def test_plot_imagenet_metrics_compare_writes_geometry_figure(tmp_path: Path) -> None:
    repa_tsv = tmp_path / "repa.tsv"
    sit_tsv = tmp_path / "sit.tsv"
    outdir = tmp_path / "plots"
    _write_metrics_tsv(repa_tsv, repa_offset=0.05)
    _write_metrics_tsv(sit_tsv, repa_offset=0.0)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_imagenet_metrics_compare.py"),
            "--repa",
            str(repa_tsv),
            "--sit",
            str(sit_tsv),
            "--outdir",
            str(outdir),
        ],
        check=True,
    )

    assert (outdir / "repa_vs_sit_metrics_layers.png").is_file()
    assert (outdir / "repa_vs_sit_metrics_probes.png").is_file()
    assert (outdir / "repa_vs_sit_metrics_geometry.png").is_file()
    assert (outdir / "repa_vs_sit_metrics_overview.png").is_file()
