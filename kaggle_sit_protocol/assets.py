from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DEFAULT_IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def synset_mapping_path() -> Path:
    return _asset_dir() / "synset_to_imagenet_idx.json"


def ensure_synset_mapping_json() -> Path:
    path = synset_mapping_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and len(payload) >= 100:
                return path
        except json.JSONDecodeError:
            pass

    with urllib.request.urlopen(DEFAULT_IMAGENET_CLASS_INDEX_URL) as response:
        class_index = json.load(response)
    mapping = {entry[0]: int(index) for index, entry in class_index.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
    return path


def load_synset_to_imagenet_idx() -> dict[str, int]:
    path = ensure_synset_mapping_json()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): int(value) for key, value in payload.items()}

