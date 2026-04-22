from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot residual-delta cosine outputs into PNGs and a single multi-page PDF."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing residual_delta_cosine_matrices.npz.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Residual Delta Cosine",
        help="Title prefix for figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated plots. Defaults to <input-dir>/plots.",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="residual_delta_cosine_heatmaps.pdf",
        help="Filename of the multi-page PDF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG/PDF raster DPI.",
    )
    return parser.parse_args()


def _draw_metric_panel(
    matrix: np.ndarray,
    ax: plt.Axes,
    labels: list[str],
    title: str,
):
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    return image


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(input_dir / "residual_delta_cosine_matrices.npz")
    delta_layers = data["delta_layers"].tolist()
    labels = [f"L{i}-L{i-1}" for i in delta_layers]
    time_indices = data["time_indices"].tolist()
    time_values = data["time_values"].tolist()
    metrics = [
        ("flat_bnd", "Flat BND"),
        ("tokenwise", "Tokenwise"),
        ("batchmean_nd", "BatchMean ND"),
    ]

    pdf_path = output_dir / args.pdf_name
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
        last_image = None
        for ax, (key, title) in zip(axes, metrics):
            last_image = _draw_metric_panel(
                data[f"{key}_mean_over_time"],
                ax,
                labels,
                f"{args.title} | {title} | mean over time",
            )
        fig.colorbar(last_image, ax=axes, shrink=0.8)
        fig.savefig(output_dir / "mean_over_time_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for time_pos, (time_index, time_value) in enumerate(zip(time_indices, time_values)):
            fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
            last_image = None
            for ax, (key, title) in zip(axes, metrics):
                last_image = _draw_metric_panel(
                    data[key][time_pos],
                    ax,
                    labels,
                    f"{args.title} | {title} | pos={time_pos} idx={time_index} t={float(time_value):.3f}",
                )
            fig.colorbar(last_image, ax=axes, shrink=0.8)
            fig.savefig(output_dir / f"timestep_{time_pos:02d}_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved plots to: {output_dir}")
    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
