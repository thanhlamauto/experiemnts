"""ImageNet → VAE latents and subset helpers for SiT / REPA metric scripts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def imagenet_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def encode_latents(vae: AutoencoderKL, images: torch.Tensor, scale: float = 0.18215) -> torch.Tensor:
    """images in [-1, 1]; returns latents scaled like SiT training."""
    d = vae.encode(images).latent_dist.sample()
    return d.mul_(scale)


def build_imagenet_loaders(
    root: str,
    resolution: int,
    batch_size: int,
    num_workers: int,
    max_train: Optional[int],
    max_val: Optional[int],
) -> Tuple[DataLoader, DataLoader]:
    train_dir = Path(root) / "train"
    val_dir = Path(root) / "val"
    tfm = imagenet_transform(resolution)
    train_set = datasets.ImageFolder(str(train_dir), transform=tfm)
    val_set = datasets.ImageFolder(str(val_dir), transform=tfm)
    if max_train is not None and max_train < len(train_set):
        train_set = Subset(train_set, list(range(max_train)))
    if max_val is not None and max_val < len(val_set):
        val_set = Subset(val_set, list(range(max_val)))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def sample_per_class_indices(
    labels: Sequence[int] | Iterable[int],
    samples_per_class: int,
    seed: int,
) -> List[int]:
    """Sample an equal number of examples per class, deterministically by seed."""
    labels_list = [int(label) for label in labels]
    if int(samples_per_class) <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}")

    by_class: dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels_list):
        by_class[int(label)].append(idx)

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected: List[int] = []
    for label in sorted(by_class):
        cls_indices = by_class[label]
        if len(cls_indices) < int(samples_per_class):
            raise ValueError(
                f"class {label} only has {len(cls_indices)} samples, "
                f"but {samples_per_class} are required"
            )
        perm = torch.randperm(len(cls_indices), generator=generator)[: int(samples_per_class)]
        chosen = [cls_indices[int(i)] for i in perm.tolist()]
        chosen.sort()
        selected.extend(chosen)

    selected.sort()
    return selected


def sample_disjoint_train_val_per_class_indices(
    labels: Sequence[int] | Iterable[int],
    train_samples_per_class: int,
    val_samples_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Sample deterministic, disjoint train/val subsets with fixed counts per class."""
    labels_list = [int(label) for label in labels]
    if int(train_samples_per_class) <= 0:
        raise ValueError(f"train_samples_per_class must be positive, got {train_samples_per_class}")
    if int(val_samples_per_class) <= 0:
        raise ValueError(f"val_samples_per_class must be positive, got {val_samples_per_class}")

    by_class: dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels_list):
        by_class[int(label)].append(idx)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_selected: List[int] = []
    val_selected: List[int] = []
    need_total = int(train_samples_per_class) + int(val_samples_per_class)
    for label in sorted(by_class):
        cls_indices = by_class[label]
        if len(cls_indices) < need_total:
            raise ValueError(
                f"class {label} only has {len(cls_indices)} samples, "
                f"but {need_total} are required for disjoint train/val splits"
            )
        perm = torch.randperm(len(cls_indices), generator=generator)[:need_total].tolist()
        chosen = [cls_indices[int(i)] for i in perm]
        train_chosen = sorted(chosen[: int(train_samples_per_class)])
        val_chosen = sorted(chosen[int(train_samples_per_class) :])
        train_selected.extend(train_chosen)
        val_selected.extend(val_chosen)

    train_selected.sort()
    val_selected.sort()
    return train_selected, val_selected


def sample_subset_indices(num_items: int, max_samples: Optional[int], seed: int) -> List[int]:
    if max_samples is None or max_samples >= num_items:
        return list(range(num_items))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator)[:max_samples].tolist()
    indices.sort()
    return indices


class IndexedSubset(Dataset):
    """Subset that also returns the stable subset position for deterministic noise lookup."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        return image, label, idx


def _build_subset_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_imagenet_per_class_loader(
    root: str,
    split: str,
    resolution: int,
    batch_size: int,
    num_workers: int,
    samples_per_class: int,
    subset_seed: int,
    *,
    shuffle: bool = False,
) -> Tuple[DataLoader, List[int], int]:
    """Build an ImageFolder loader with a fixed number of samples per class."""
    split_dir = Path(root) / split
    dataset = datasets.ImageFolder(str(split_dir), transform=imagenet_transform(resolution))
    indices = sample_per_class_indices(dataset.targets, samples_per_class=samples_per_class, seed=subset_seed)
    loader = _build_subset_loader(dataset, indices, batch_size, num_workers, shuffle=shuffle)
    num_classes = len({int(dataset.targets[idx]) for idx in indices})
    return loader, indices, num_classes


def build_imagenet_per_class_loaders(
    root: str,
    resolution: int,
    batch_size: int,
    num_workers: int,
    train_samples_per_class: int,
    val_samples_per_class: int,
    subset_seed: int,
) -> Tuple[DataLoader, DataLoader, List[int], List[int], int, str]:
    """Train/val loaders with matched per-class fixed-k subsets.

    If ``train/`` and ``val/`` do not share the same class space (for example Mini-ImageNet's
    disjoint 64/16/20 class splits), fall back to drawing both train and validation subsets
    from a single split with disjoint per-class sampling so NCM remains well-defined.
    """
    tfm = imagenet_transform(resolution)
    train_dataset = datasets.ImageFolder(str(Path(root) / "train"), transform=tfm)
    val_dataset = datasets.ImageFolder(str(Path(root) / "val"), transform=tfm)

    if train_dataset.classes == val_dataset.classes:
        train_indices = sample_per_class_indices(
            train_dataset.targets,
            samples_per_class=train_samples_per_class,
            seed=subset_seed,
        )
        val_indices = sample_per_class_indices(
            val_dataset.targets,
            samples_per_class=val_samples_per_class,
            seed=subset_seed,
        )
        train_loader = _build_subset_loader(train_dataset, train_indices, batch_size, num_workers, shuffle=False)
        val_loader = _build_subset_loader(val_dataset, val_indices, batch_size, num_workers, shuffle=False)
        num_classes = len({int(train_dataset.targets[idx]) for idx in train_indices})
        return train_loader, val_loader, train_indices, val_indices, num_classes, "per_class_fixed_split_pair"

    shared_split = "train" if len(train_dataset.classes) >= len(val_dataset.classes) else "val"
    shared_dataset = train_dataset if shared_split == "train" else val_dataset
    train_indices, val_indices = sample_disjoint_train_val_per_class_indices(
        shared_dataset.targets,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        seed=subset_seed,
    )
    train_loader = _build_subset_loader(shared_dataset, train_indices, batch_size, num_workers, shuffle=False)
    val_loader = _build_subset_loader(shared_dataset, val_indices, batch_size, num_workers, shuffle=False)
    num_classes = len({int(shared_dataset.targets[idx]) for idx in train_indices})
    protocol = f"per_class_fixed_shared_split:{shared_split}"
    return train_loader, val_loader, train_indices, val_indices, num_classes, protocol


def build_imagenet_indexed_loader(
    root: str,
    split: str,
    resolution: int,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    subset_seed: int,
) -> Tuple[DataLoader, List[int]]:
    split_dir = Path(root) / split
    dataset = datasets.ImageFolder(str(split_dir), transform=imagenet_transform(resolution))
    indices = sample_subset_indices(len(dataset), max_samples, subset_seed)
    indexed_dataset = IndexedSubset(dataset, indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, indices
