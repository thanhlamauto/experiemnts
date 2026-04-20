from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import load_synset_to_imagenet_idx
from .config import ProtocolConfig

_SYNSET_PATTERN = re.compile(r"^n\d{8}$")


@dataclass
class DatasetDiscovery:
    root: Path
    synset_dirs: list[Path]


def discover_dataset_root(config: ProtocolConfig) -> DatasetDiscovery:
    candidates: list[DatasetDiscovery] = []
    for root in config.iter_dataset_search_roots():
        if root.is_dir():
            synsets = [child for child in root.iterdir() if child.is_dir() and _SYNSET_PATTERN.match(child.name)]
            if len(synsets) == 100:
                candidates.append(DatasetDiscovery(root=root, synset_dirs=sorted(synsets)))
            for subdir in root.rglob("*"):
                if not subdir.is_dir():
                    continue
                if config.dataset_name_hint.lower() not in str(subdir).lower():
                    continue
                synsets = [
                    child
                    for child in subdir.iterdir()
                    if child.is_dir() and _SYNSET_PATTERN.match(child.name)
                ]
                if len(synsets) == 100:
                    candidates.append(DatasetDiscovery(root=subdir, synset_dirs=sorted(synsets)))

    if not candidates:
        raise FileNotFoundError(
            "Could not discover a miniImageNet root with exactly 100 synset folders under "
            f"{config.dataset_search_roots}."
        )

    candidates.sort(key=lambda item: (len(str(item.root)), str(item.root)))
    return candidates[0]


def _sorted_image_files(synset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in synset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _pick_preview_ids(rows: list[dict[str, object]], preview_images: int) -> set[str]:
    by_synset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["subset_role"] != "main":
            continue
        by_synset[str(row["synset"])].append(str(row["image_id"]))

    preview_ids: list[str] = []
    for synset in sorted(by_synset):
        if len(preview_ids) >= preview_images:
            break
        preview_ids.append(by_synset[synset][0])
    return set(preview_ids)


def build_manifest(config: ProtocolConfig) -> pd.DataFrame:
    discovery = discover_dataset_root(config)
    synset_to_idx = load_synset_to_imagenet_idx()
    rng = np.random.default_rng(config.seed)

    rows: list[dict[str, object]] = []
    for class_position, synset_dir in enumerate(discovery.synset_dirs):
        synset = synset_dir.name
        if synset not in synset_to_idx:
            raise KeyError(f"Missing ImageNet idx mapping for synset {synset}")
        image_files = _sorted_image_files(synset_dir)
        if len(image_files) < config.main_images_per_class + config.control_images_per_class:
            raise ValueError(
                f"Synset {synset} only has {len(image_files)} images; need "
                f"{config.main_images_per_class + config.control_images_per_class}."
            )
        shuffled_positions = rng.permutation(len(image_files))
        sampled = [image_files[int(pos)] for pos in shuffled_positions[: config.main_images_per_class + config.control_images_per_class]]

        main_paths = sampled[: config.main_images_per_class]
        control_paths = sampled[config.main_images_per_class :]

        for local_position, image_path in enumerate(main_paths):
            image_id = f"{synset}_main_{local_position:02d}"
            rows.append(
                {
                    "image_id": image_id,
                    "relative_path": str(image_path.relative_to(discovery.root)),
                    "absolute_path": str(image_path),
                    "synset": synset,
                    "imagenet_idx": synset_to_idx[synset],
                    "class_index_100": class_position,
                    "subset_role": "main",
                    "probe_split": "train" if local_position < 4 else "test",
                    "preview": False,
                }
            )

        for local_position, image_path in enumerate(control_paths):
            image_id = f"{synset}_control_{local_position:02d}"
            rows.append(
                {
                    "image_id": image_id,
                    "relative_path": str(image_path.relative_to(discovery.root)),
                    "absolute_path": str(image_path),
                    "synset": synset,
                    "imagenet_idx": synset_to_idx[synset],
                    "class_index_100": class_position,
                    "subset_role": "control",
                    "probe_split": "none",
                    "preview": False,
                }
            )

    preview_ids = _pick_preview_ids(rows, config.preview_images)
    for row in rows:
        row["preview"] = row["image_id"] in preview_ids
        row["dataset_root"] = str(discovery.root)

    manifest = pd.DataFrame(rows).sort_values(["subset_role", "class_index_100", "image_id"]).reset_index(drop=True)
    if int((manifest["subset_role"] == "main").sum()) != config.main_images_target:
        raise ValueError("Main manifest size mismatch.")
    if int((manifest["subset_role"] == "control").sum()) != config.control_images_target:
        raise ValueError("Control manifest size mismatch.")
    if int(manifest["preview"].sum()) != config.preview_images:
        raise ValueError("Preview manifest size mismatch.")
    return manifest
