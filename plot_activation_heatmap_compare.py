#!/usr/bin/env python3
"""Ghép hai PNG heatmap (SiT vs REPA) cạnh nhau để so sánh."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side activation heatmap comparison.")
    ap.add_argument("--sit", type=Path, required=True, help="PNG từ plot_sit_xl2 ... --backend sit")
    ap.add_argument("--repa", type=Path, required=True, help="PNG từ plot_sit_xl2 ... --backend repa")
    ap.add_argument("--out", type=Path, default=Path("/workspace/outputs/activation_heatmap_compare/sit_vs_repa_activations.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    a = mpimg.imread(args.sit)
    b = mpimg.imread(args.repa)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), dpi=args.dpi)
    axes[0].imshow(a)
    axes[0].set_title("SiT (vanilla)", fontsize=14, fontweight="600")
    axes[0].axis("off")
    axes[1].imshow(b)
    axes[1].set_title("REPA", fontsize=14, fontweight="600")
    axes[1].axis("off")
    fig.suptitle("|h| activation heatmap — layer × magnitude", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=args.dpi)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
