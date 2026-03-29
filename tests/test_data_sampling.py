from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("diffusers")
PIL = pytest.importorskip("PIL.Image")

from sit_metrics.data import (
    build_imagenet_per_class_loaders,
    sample_disjoint_train_val_per_class_indices,
    sample_per_class_indices,
)


def test_sample_per_class_indices_is_deterministic_and_balanced() -> None:
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    first = sample_per_class_indices(labels, samples_per_class=2, seed=7)
    second = sample_per_class_indices(labels, samples_per_class=2, seed=7)

    assert first == second
    assert len(first) == 6
    counts = Counter(labels[idx] for idx in first)
    assert counts == {0: 2, 1: 2, 2: 2}


def test_sample_per_class_indices_raises_when_a_class_is_too_small() -> None:
    labels = [0, 0, 1]

    with pytest.raises(ValueError, match="only has"):
        sample_per_class_indices(labels, samples_per_class=2, seed=0)


def test_sample_disjoint_train_val_per_class_indices_is_deterministic_balanced_and_disjoint() -> None:
    labels = [0] * 5 + [1] * 5 + [2] * 5

    train_a, val_a = sample_disjoint_train_val_per_class_indices(labels, train_samples_per_class=2, val_samples_per_class=1, seed=7)
    train_b, val_b = sample_disjoint_train_val_per_class_indices(labels, train_samples_per_class=2, val_samples_per_class=1, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert Counter(labels[idx] for idx in train_a) == {0: 2, 1: 2, 2: 2}
    assert Counter(labels[idx] for idx in val_a) == {0: 1, 1: 1, 2: 1}


def test_build_imagenet_per_class_loaders_falls_back_to_shared_train_split(tmp_path) -> None:
    Image = PIL
    root = tmp_path / "mini"
    for split, class_names in (("train", ("000", "001")), ("val", ("000",))):
        for class_name in class_names:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(3):
                Image.new("RGB", (8, 8), color=(idx * 20, 0, 0)).save(class_dir / f"{idx}.png")

    train_loader, val_loader, train_indices, val_indices, num_classes, protocol = build_imagenet_per_class_loaders(
        root=str(root),
        resolution=8,
        batch_size=2,
        num_workers=0,
        train_samples_per_class=1,
        val_samples_per_class=1,
        subset_seed=0,
    )

    assert protocol == "per_class_fixed_shared_split:train"
    assert num_classes == 2
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 2
    assert set(train_indices).isdisjoint(val_indices)
