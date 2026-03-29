from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("matplotlib")

from plot_sit_spatial_metrics import plot_metric_triptych, read_metrics_tsv


def _write_tsv(path: Path, backend: str, offset: float) -> None:
    rows = [
        ["backend", "ckpt", "metric", "layer", "noise_level", "model_t", "value", "num_images", "path_type", "noise_protocol"],
        [backend, f"{backend}.pt", "lds", 0, 1.0, 1.0, 0.1 + offset, 8, "linear", "fixed_per_image"],
        [backend, f"{backend}.pt", "lds", 0, 0.0, 0.0, 0.2 + offset, 8, "linear", "fixed_per_image"],
        [backend, f"{backend}.pt", "lds", 1, 1.0, 1.0, 0.3 + offset, 8, "linear", "fixed_per_image"],
        [backend, f"{backend}.pt", "lds", 1, 0.0, 0.0, 0.4 + offset, 8, "linear", "fixed_per_image"],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def test_plot_metric_triptych_smoke(tmp_path: Path) -> None:
    sit_tsv = tmp_path / "sit.tsv"
    repa_tsv = tmp_path / "repa.tsv"
    out_png = tmp_path / "lds_triptych.png"
    _write_tsv(sit_tsv, backend="sit", offset=0.0)
    _write_tsv(repa_tsv, backend="repa", offset=0.5)

    sit_rows = read_metrics_tsv(sit_tsv)
    repa_rows = read_metrics_tsv(repa_tsv)
    plot_metric_triptych(sit_rows, repa_rows, metric="lds", out_path=out_png)

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
