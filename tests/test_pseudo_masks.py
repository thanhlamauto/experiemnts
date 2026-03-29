from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from sit_metrics.pseudo_masks import load_pseudo_mask_targets


def test_load_pseudo_masks_aligns_by_dataset_indices_and_derives_targets(tmp_path: Path) -> None:
    npz_path = tmp_path / "pseudo.npz"
    patch_labels = np.asarray(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int64,
    )
    dataset_indices = np.asarray([7, 11, 3], dtype=np.int64)
    np.savez(npz_path, patch_labels=patch_labels, dataset_indices=dataset_indices)

    targets = load_pseudo_mask_targets(
        npz_path,
        subset_indices=[3, 7],
        p=2,
        q=2,
        requested_metrics=["patch_miou", "boundary_f1", "objectness_iou"],
        ignore_index=-1,
        background_label=0,
    )

    assert targets.patch_labels is not None
    assert targets.boundary_labels is not None
    assert targets.object_masks is not None
    assert targets.patch_labels.shape == (2, 4)
    assert targets.object_masks.shape == (2, 4)
    assert targets.patch_labels[0].tolist() == [1, 1, 0, 0]
    assert targets.object_masks[0].tolist() == [1, 1, 0, 0]
    assert set(targets.boundary_labels[0].tolist()) <= {-1, 0, 1}


def test_load_pseudo_masks_requires_object_source_for_objectness(tmp_path: Path) -> None:
    npz_path = tmp_path / "pseudo_missing_object.npz"
    np.savez(npz_path, patch_labels=np.zeros((2, 2, 2), dtype=np.int64))

    with pytest.raises(ValueError):
        load_pseudo_mask_targets(
            npz_path,
            subset_indices=[0, 1],
            p=2,
            q=2,
            requested_metrics=["objectness_iou"],
            ignore_index=-1,
            background_label=None,
        )
