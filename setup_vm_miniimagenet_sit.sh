#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.hf_home}"
MINIIMAGENET_DOWNLOAD_DIR="${MINIIMAGENET_DOWNLOAD_DIR:-$ROOT_DIR/data/miniimagenet_downloads}"
MINIIMAGENET_ROOT="${MINIIMAGENET_ROOT:-$ROOT_DIR/data/miniimagenet_imagefolder}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/pretrained_models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
REBUILD_MINIIMAGENET="${REBUILD_MINIIMAGENET:-0}"

export HF_HOME
export MPLBACKEND="${MPLBACKEND:-Agg}"
export ROOT_DIR
export MINIIMAGENET_DOWNLOAD_DIR
export MINIIMAGENET_ROOT
export CHECKPOINT_DIR
export REBUILD_MINIIMAGENET

mkdir -p "$HF_HOME" "$MINIIMAGENET_DOWNLOAD_DIR" "$MINIIMAGENET_ROOT" "$CHECKPOINT_DIR"

echo "[setup] repo root: $ROOT_DIR"
echo "[setup] venv: $VENV_DIR"
echo "[setup] hf cache: $HF_HOME"
echo "[setup] miniImageNet downloads: $MINIIMAGENET_DOWNLOAD_DIR"
echo "[setup] miniImageNet imagefolder: $MINIIMAGENET_ROOT"
echo "[setup] checkpoints: $CHECKPOINT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] detected GPU:"
  nvidia-smi || true
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtualenv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[setup] upgrading pip tooling"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

echo "[setup] installing PyTorch CUDA wheels"
"$VENV_DIR/bin/pip" install torch torchvision --index-url "$CUDA_INDEX_URL"

echo "[setup] installing repo/runtime dependencies"
"$VENV_DIR/bin/pip" install \
  timm \
  diffusers \
  accelerate \
  numpy \
  Pillow \
  tqdm \
  pandas \
  pyarrow \
  torchdiffeq \
  matplotlib \
  seaborn \
  scikit-learn \
  umap-learn \
  PyWavelets \
  huggingface_hub \
  gdown

echo "[setup] verifying torch + CUDA"
"$VENV_DIR/bin/python" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_count={torch.cuda.device_count()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
PY

echo "[setup] downloading SiT checkpoints and miniImageNet"
"$VENV_DIR/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import gdown
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

repo_root = Path(os.environ["ROOT_DIR"])
downloads_root = Path(os.environ["MINIIMAGENET_DOWNLOAD_DIR"])
dataset_root = Path(os.environ["MINIIMAGENET_ROOT"])
checkpoints_dir = Path(os.environ["CHECKPOINT_DIR"])
rebuild_dataset = os.environ.get("REBUILD_MINIIMAGENET", "0") == "1"

downloads_root.mkdir(parents=True, exist_ok=True)
dataset_root.mkdir(parents=True, exist_ok=True)
checkpoints_dir.mkdir(parents=True, exist_ok=True)

model_filenames = {
    "SiT-XL/2": "SiT-XL-2-256.pt",
    "SiT-B/2": "SiT-B-2-256.pt",
}
for model_name, filename in model_filenames.items():
    path = hf_hub_download(
        repo_id="nyu-visionx/SiT-collections",
        filename=filename,
        local_dir=str(checkpoints_dir),
    )
    print(f"[checkpoint] {model_name}: {path}")

pickle_files = {
    "train": ("mini-imagenet-cache-train.pkl", "1I3itTXpXxGV68olxM5roceUMG8itH9Xj"),
    "val": ("mini-imagenet-cache-val.pkl", "1KY5e491bkLFqJDp0-UWou3463Mo8AOco"),
    "test": ("mini-imagenet-cache-test.pkl", "1wpmY-hmiJUUlRBkO9ZDCXAcIpHEFdOhD"),
}
local_pickles: dict[str, Path] = {}
for split_name, (filename, file_id) in pickle_files.items():
    path = downloads_root / filename
    if not path.is_file():
        print(f"[dataset] downloading {filename}")
        gdown.download(id=file_id, output=str(path), quiet=False)
    else:
        print(f"[dataset] reusing {path}")
    local_pickles[split_name] = path

marker_path = dataset_root / "_build_complete.json"
if marker_path.is_file() and not rebuild_dataset:
    summary = json.loads(marker_path.read_text())
    print(
        "[dataset] reusing imagefolder "
        f"root={summary['dataset_root']} synsets={summary['num_synsets']} images={summary['total_images']}"
    )
    raise SystemExit(0)

if rebuild_dataset:
    for child in dataset_root.iterdir():
        if child.is_dir():
            for nested in child.iterdir():
                if nested.is_file():
                    nested.unlink()
            child.rmdir()
        else:
            child.unlink()

def load_split(path: Path) -> tuple[np.ndarray, dict[str, list[int]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    image_data = np.asarray(payload["image_data"])
    class_dict = {str(key): [int(index) for index in value] for key, value in payload["class_dict"].items()}
    return image_data, class_dict

counts_by_synset: dict[str, int] = {}
split_counts: dict[str, int] = {}
split_synsets: dict[str, int] = {}
total_images = 0
all_synsets: set[str] = set()

for split_name, pickle_path in local_pickles.items():
    image_data, class_dict = load_split(pickle_path)
    split_counts[split_name] = int(image_data.shape[0])
    split_synsets[split_name] = int(len(class_dict))
    for synset, indices in sorted(class_dict.items()):
        all_synsets.add(synset)
        synset_dir = dataset_root / synset
        synset_dir.mkdir(parents=True, exist_ok=True)
        offset = counts_by_synset.get(synset, 0)
        for local_index, image_index in enumerate(indices):
            output_path = synset_dir / f"{split_name}_{offset + local_index:04d}.png"
            if output_path.is_file():
                continue
            Image.fromarray(image_data[image_index]).save(output_path)
        counts_by_synset[synset] = offset + len(indices)
        total_images += len(indices)

summary = {
    "dataset_root": str(dataset_root.resolve()),
    "num_synsets": int(len(all_synsets)),
    "total_images": int(total_images),
    "split_counts": split_counts,
    "split_synsets": split_synsets,
}
marker_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(
    f"[dataset] built imagefolder root={summary['dataset_root']} "
    f"synsets={summary['num_synsets']} images={summary['total_images']}"
)
PY

echo "[setup] done"
