#!/usr/bin/env python3
"""Generate binary pseudo masks from DINOv2 patch tokens for the current ImageNet subset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sit_metrics.data import build_imagenet_indexed_loader
from sit_metrics.reference_encoders import dinov2_patch_features, load_dinov2_model


def _grid_size_from_tokens(num_tokens: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Expected square token grid, got T={num_tokens}")
    return side, side


def _center_prior(p: int, q: int, sigma_frac: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(p, dtype=np.float32), np.arange(q, dtype=np.float32), indexing="ij")
    cy = 0.5 * (p - 1)
    cx = 0.5 * (q - 1)
    sigma = float(max(p, q)) * float(sigma_frac)
    sigma = max(sigma, 1e-3)
    score = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return score.reshape(-1)


def _derive_boundary(mask_flat: np.ndarray, p: int, q: int) -> np.ndarray:
    grid = mask_flat.reshape(p, q).astype(np.int64)
    boundary = np.zeros_like(grid, dtype=np.int64)
    diff_down = grid[1:, :] != grid[:-1, :]
    diff_right = grid[:, 1:] != grid[:, :-1]
    boundary[1:, :] |= diff_down
    boundary[:-1, :] |= diff_down
    boundary[:, 1:] |= diff_right
    boundary[:, :-1] |= diff_right
    return boundary


def _fit_binary_kmeans(feats: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scikit-learn is required to generate DINO pseudo masks.") from exc

    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    return km.fit_predict(feats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DINOv2-based binary pseudo masks as NPZ.")
    parser.add_argument("--imagenet-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--center-sigma-frac", type=float, default=0.35)
    parser.add_argument(
        "--out",
        type=str,
        default="/workspace/outputs/dinov2_pseudo_masks/pseudo_masks.npz",
        help="Output NPZ path. Saved keys: patch_labels, object_masks, boundary_labels, dataset_indices.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, subset_indices = build_imagenet_indexed_loader(
        root=args.imagenet_root,
        split=args.split,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
    )

    model = load_dinov2_model(args.dino_model, device, args.resolution)

    object_masks = None
    boundary_labels = None
    patch_labels = None
    dataset_indices = np.asarray(subset_indices, dtype=np.int64)

    offset = 0
    for images, _labels, _sample_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            tokens = dinov2_patch_features(model, images, args.resolution).detach().cpu().numpy()

        batch_size = tokens.shape[0]
        p, q = _grid_size_from_tokens(tokens.shape[1])
        if object_masks is None:
            object_masks = np.zeros((len(subset_indices), p, q), dtype=np.int64)
            boundary_labels = np.zeros((len(subset_indices), p, q), dtype=np.int64)
            patch_labels = np.zeros((len(subset_indices), p, q), dtype=np.int64)
            center_prior = _center_prior(p, q, args.center_sigma_frac)

        for bi in range(batch_size):
            feats = tokens[bi].astype(np.float32)
            cluster = _fit_binary_kmeans(feats, seed=args.seed + offset + bi)
            cluster_scores = []
            for cid in (0, 1):
                mask = cluster == cid
                if not np.any(mask):
                    cluster_scores.append(float("-inf"))
                else:
                    cluster_scores.append(float(center_prior[mask].mean()))
            foreground_cluster = int(np.argmax(cluster_scores))
            obj = (cluster == foreground_cluster).astype(np.int64)

            object_masks[offset + bi] = obj.reshape(p, q)
            patch_labels[offset + bi] = obj.reshape(p, q)
            boundary_labels[offset + bi] = _derive_boundary(obj, p, q)
        offset += batch_size

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        patch_labels=patch_labels,
        object_masks=object_masks,
        boundary_labels=boundary_labels,
        dataset_indices=dataset_indices,
        p=np.asarray([patch_labels.shape[1]], dtype=np.int64),
        q=np.asarray([patch_labels.shape[2]], dtype=np.int64),
    )
    print(f"Saved pseudo masks: {out_path}")
    print(f"Subset size: {len(subset_indices)} | token grid: {patch_labels.shape[1]}x{patch_labels.shape[2]}")


if __name__ == "__main__":
    main()
