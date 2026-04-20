# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained SiT models.
"""
from pathlib import Path

import os
import torch
from torchvision.datasets.utils import download_url


MODEL_SOURCES = {
    "SiT-XL-2-256x256.pt": {
        "aliases": ("SiT-XL-2-256.pt",),
        "hf_repo": "nyu-visionx/SiT-collections",
        "hf_filename": "SiT-XL-2-256.pt",
        "direct_url": "https://huggingface.co/nyu-visionx/SiT-collections/resolve/main/SiT-XL-2-256.pt",
    }
}
pretrained_models = set(MODEL_SOURCES)


def _canonical_model_name(model_name):
    for canonical_name, meta in MODEL_SOURCES.items():
        if model_name == canonical_name or model_name in meta["aliases"]:
            return canonical_name
    return None


def _iter_local_candidates(model_name):
    requested_path = Path(model_name)
    if requested_path.is_file():
        yield requested_path

    canonical_name = _canonical_model_name(model_name)
    if not canonical_name:
        return

    model_dir = Path("pretrained_models")
    yield model_dir / canonical_name
    for alias in MODEL_SOURCES[canonical_name]["aliases"]:
        yield model_dir / alias


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    canonical_name = _canonical_model_name(model_name)
    if canonical_name:
        for candidate in _iter_local_candidates(model_name):
            if candidate.is_file():
                return _load_checkpoint(candidate)
        return download_model(canonical_name)

    assert os.path.isfile(model_name), (
        f"Could not find SiT checkpoint at {model_name}. "
        "When running offline, set cfg.checkpoint_path to a local .pt checkpoint."
    )
    return _load_checkpoint(model_name)


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    canonical_name = _canonical_model_name(model_name)
    assert canonical_name in pretrained_models

    model_dir = Path("pretrained_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for candidate in _iter_local_candidates(canonical_name):
        if candidate.is_file():
            return _load_checkpoint(candidate)

    meta = MODEL_SOURCES[canonical_name]

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=meta["hf_repo"],
            filename=meta["hf_filename"],
            local_dir=str(model_dir),
        )
        return _load_checkpoint(downloaded_path)
    except Exception:
        local_path = model_dir / canonical_name
        if not local_path.is_file():
            download_url(meta["direct_url"], str(model_dir), filename=canonical_name)
        return _load_checkpoint(local_path)
