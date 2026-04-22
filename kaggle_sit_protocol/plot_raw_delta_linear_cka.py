from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw-activation and layer-delta linear-CKA outputs into PNGs and a multi-page PDF."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw_delta_linear_cka_matrices.npz.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title prefix. Defaults to the model name saved in the NPZ.",
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
        default="raw_delta_linear_cka_heatmaps.pdf",
        help="Filename of the multi-page PDF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG/PDF raster DPI.",
    )
    return parser.parse_args()


def _string_scalar(value: np.ndarray) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _draw_metric_panel(
    matrix: np.ndarray,
    ax: plt.Axes,
    labels: list[str],
    title: str,
):
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
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

    data = np.load(input_dir / "raw_delta_linear_cka_matrices.npz")
    model_name = _string_scalar(data["model_name"])
    subset_role = _string_scalar(data["subset_role"])
    title_prefix = args.title or f"{model_name} | subset={subset_role}"

    raw_labels = [f"L{i}" for i in data["raw_layers"].tolist()]
    delta_labels = [f"L{i}-L{i-1}" for i in data["delta_layers"].tolist()]
    time_indices = data["time_indices"].tolist()
    time_values = data["time_values"].tolist()

    pdf_path = output_dir / args.pdf_name
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
        last_image = _draw_metric_panel(
            data["raw_mean_over_time"],
            axes[0],
            raw_labels,
            f"{title_prefix} | raw activations | mean over time",
        )
        _draw_metric_panel(
            data["delta_mean_over_time"],
            axes[1],
            delta_labels,
            f"{title_prefix} | deltas L_i-L_(i-1) | mean over time",
        )
        fig.colorbar(last_image, ax=axes, shrink=0.82)
        fig.savefig(output_dir / "mean_over_time_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for time_pos, (time_index, time_value) in enumerate(zip(time_indices, time_values)):
            fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
            last_image = _draw_metric_panel(
                data["raw"][time_pos],
                axes[0],
                raw_labels,
                f"{title_prefix} | raw | pos={time_pos} idx={time_index} t={float(time_value):.3f}",
            )
            _draw_metric_panel(
                data["delta"][time_pos],
                axes[1],
                delta_labels,
                f"{title_prefix} | delta | pos={time_pos} idx={time_index} t={float(time_value):.3f}",
            )
            fig.colorbar(last_image, ax=axes, shrink=0.82)
            fig.savefig(output_dir / f"timestep_{time_pos:02d}_heatmaps.png", dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved plots to: {output_dir}")
    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
