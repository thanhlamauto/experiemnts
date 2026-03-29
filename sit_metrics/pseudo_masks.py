"""Load and align pseudo-mask targets for token-level dense metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PseudoMaskTargets:
    patch_labels: torch.Tensor | None = None  # [N, T], int64, ignore_index marks invalid tokens
    boundary_labels: torch.Tensor | None = None  # [N, T], int64 in {-1, 0, 1}
    object_masks: torch.Tensor | None = None  # [N, T], int64 in {-1, 0, 1}


def _resolve_index_key(files: Iterable[str]) -> str | None:
    for key in ("dataset_indices", "subset_indices", "indices"):
        if key in files:
            return key
    return None


def _align_first_dim(
    arr: np.ndarray,
    subset_indices: Sequence[int],
    index_array: np.ndarray | None,
) -> np.ndarray:
    if index_array is None:
        if arr.shape[0] != len(subset_indices):
            raise ValueError(
                "Pseudo-mask arrays without dataset/subset indices must already match the current subset order: "
                f"got {arr.shape[0]} rows for {len(subset_indices)} subset items."
            )
        return arr

    if arr.shape[0] != index_array.shape[0]:
        raise ValueError(
            f"Index array length mismatch: data has {arr.shape[0]} rows but indices have {index_array.shape[0]} rows."
        )

    mapping = {}
    for row, idx in enumerate(index_array.tolist()):
        idx_int = int(idx)
        if idx_int in mapping:
            raise ValueError(f"Duplicate dataset index {idx_int} in pseudo-mask NPZ.")
        mapping[idx_int] = row

    missing = [int(idx) for idx in subset_indices if int(idx) not in mapping]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"Pseudo-mask NPZ is missing {len(missing)} subset indices, for example: {preview}")

    order = [mapping[int(idx)] for idx in subset_indices]
    return arr[np.asarray(order, dtype=np.int64)]


def _to_token_grid(
    arr: np.ndarray,
    p: int,
    q: int,
    *,
    kind: str,
) -> torch.Tensor:
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]

    if arr.ndim == 2:
        if arr.shape[1] != p * q:
            raise ValueError(f"Expected flattened token grid with second dim {p*q}, got {arr.shape}")
        return torch.as_tensor(arr, dtype=torch.int64)

    if arr.ndim != 3:
        raise ValueError(f"Expected shape [N, T], [N, H, W], or [N, 1, H, W], got {arr.shape}")

    t = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(1)
    if arr.shape[1] != p or arr.shape[2] != q:
        t = F.interpolate(t, size=(p, q), mode="nearest")
    t = t.squeeze(1)
    if kind == "binary":
        # Preserve ignore=-1 exactly when present; otherwise threshold to {0,1}.
        if torch.any(t < 0):
            return t.round().to(torch.int64).reshape(t.shape[0], p * q)
        return (t > 0.5).to(torch.int64).reshape(t.shape[0], p * q)
    return t.round().to(torch.int64).reshape(t.shape[0], p * q)


def _derive_object_masks_from_patch_labels(
    patch_labels: torch.Tensor,
    *,
    ignore_index: int,
    background_label: int,
) -> torch.Tensor:
    valid = patch_labels != int(ignore_index)
    obj = (patch_labels != int(background_label)) & valid
    out = torch.full_like(patch_labels, fill_value=int(ignore_index))
    out[valid] = obj[valid].to(torch.int64)
    return out


def _derive_boundary_from_label_grid(
    labels: torch.Tensor,
    *,
    p: int,
    q: int,
    ignore_index: int,
) -> torch.Tensor:
    grid = labels.reshape(labels.shape[0], p, q)
    valid = grid != int(ignore_index)
    boundary = torch.zeros_like(grid, dtype=torch.bool)

    diff_down = (grid[:, 1:, :] != grid[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    diff_right = (grid[:, :, 1:] != grid[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    boundary[:, 1:, :] |= diff_down
    boundary[:, :-1, :] |= diff_down
    boundary[:, :, 1:] |= diff_right
    boundary[:, :, :-1] |= diff_right

    out = torch.full_like(grid, fill_value=int(ignore_index), dtype=torch.int64)
    out[valid] = boundary[valid].to(torch.int64)
    return out.reshape(labels.shape[0], p * q)


def load_pseudo_mask_targets(
    npz_path: str | Path,
    *,
    subset_indices: Sequence[int],
    p: int,
    q: int,
    requested_metrics: Sequence[str],
    ignore_index: int = -1,
    background_label: int | None = None,
) -> PseudoMaskTargets:
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pseudo-mask NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        index_key = _resolve_index_key(data.files)
        index_array = None if index_key is None else np.asarray(data[index_key]).reshape(-1)

        patch_labels = None
        boundary_labels = None
        object_masks = None

        if "patch_labels" in data.files:
            arr = _align_first_dim(np.asarray(data["patch_labels"]), subset_indices, index_array)
            patch_labels = _to_token_grid(arr, p, q, kind="label")

        if "boundary_labels" in data.files:
            arr = _align_first_dim(np.asarray(data["boundary_labels"]), subset_indices, index_array)
            boundary_labels = _to_token_grid(arr, p, q, kind="binary")

        if "object_masks" in data.files:
            arr = _align_first_dim(np.asarray(data["object_masks"]), subset_indices, index_array)
            object_masks = _to_token_grid(arr, p, q, kind="binary")

    requested = set(requested_metrics)
    if "patch_miou" in requested and patch_labels is None:
        raise ValueError("Metric patch_miou requires `patch_labels` in the pseudo-mask NPZ.")

    if "objectness_iou" in requested and object_masks is None:
        if patch_labels is None:
            raise ValueError(
                "Metric objectness_iou requires `object_masks`, or `patch_labels` plus `--pseudo-background-label`."
            )
        if background_label is None:
            raise ValueError(
                "Metric objectness_iou cannot derive foreground masks from patch_labels without "
                "`--pseudo-background-label`."
            )
        object_masks = _derive_object_masks_from_patch_labels(
            patch_labels,
            ignore_index=ignore_index,
            background_label=background_label,
        )

    if "boundary_f1" in requested and boundary_labels is None:
        source = object_masks if object_masks is not None else patch_labels
        if source is None:
            raise ValueError(
                "Metric boundary_f1 requires `boundary_labels`, or a derivable source (`object_masks` or `patch_labels`)."
            )
        boundary_labels = _derive_boundary_from_label_grid(source, p=p, q=q, ignore_index=ignore_index)

    return PseudoMaskTargets(
        patch_labels=patch_labels,
        boundary_labels=boundary_labels,
        object_masks=object_masks,
    )
