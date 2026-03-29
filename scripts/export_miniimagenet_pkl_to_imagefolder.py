#!/usr/bin/env python3
"""
Download Mini-ImageNet .pkl files (learn2learn / Ravi splits), then export to ImageFolder layout:

  OUT/train/<class_id>/*.png
  OUT/val/<class_id>/*.png

Train = 64 classes, val = 16 classes (validation split). Test split ignored here.

Example:
  mkdir -p ~/data/mini && cd ~/data/mini
  wget 'https://zenodo.org/record/7978538/files/mini-imagenet-cache-train.pkl'
  wget 'https://zenodo.org/record/7978538/files/mini-imagenet-cache-validation.pkl'
  python scripts/export_miniimagenet_pkl_to_imagefolder.py --pkl-dir ~/data/mini --out-dir ~/data/mini_imagenet_folder
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def export_split(data: dict, out_root: Path, split_name: str, prefix: str = "img") -> None:
    """data has 'image_data' [N, 84, 84, 3] uint8 and 'class_dict' mapping class_name -> indices."""
    x = data["image_data"]
    class_dict = data["class_dict"]
    # map class_name -> int id 0..C-1
    names = sorted(class_dict.keys())
    name_to_id = {n: i for i, n in enumerate(names)}

    for cname, idxs in tqdm(class_dict.items(), desc=split_name):
        cid = name_to_id[cname]
        d = out_root / split_name / f"{cid:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for j, idx in enumerate(idxs):
            im = x[idx]
            Image.fromarray(im.astype(np.uint8)).save(d / f"{prefix}_{idx}_{j}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", type=Path, required=True, help="Directory containing train/validation pkl")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output root with train/ and val/")
    args = ap.parse_args()

    train_p = args.pkl_dir / "mini-imagenet-cache-train.pkl"
    val_p = args.pkl_dir / "mini-imagenet-cache-validation.pkl"
    if not train_p.is_file():
        raise FileNotFoundError(train_p)
    if not val_p.is_file():
        raise FileNotFoundError(val_p)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading train...")
    export_split(load_pkl(train_p), args.out_dir, "train")
    print("Loading val...")
    export_split(load_pkl(val_p), args.out_dir, "val")
    print(f"Done. Use --imagenet-root {args.out_dir}")


if __name__ == "__main__":
    main()
