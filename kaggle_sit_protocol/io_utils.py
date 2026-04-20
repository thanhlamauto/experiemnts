from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def done_path(stage_dir: Path) -> Path:
    return stage_dir / "done.json"


def sanity_path(stage_dir: Path) -> Path:
    return stage_dir / "sanity.json"


def mark_done(stage_dir: Path, payload: dict[str, Any]) -> None:
    write_json(done_path(stage_dir), payload)


def mark_sanity(stage_dir: Path, payload: dict[str, Any]) -> None:
    write_json(sanity_path(stage_dir), payload)


def stage_is_complete(stage_dir: Path) -> bool:
    return done_path(stage_dir).exists() and sanity_path(stage_dir).exists()


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def estimate_bytes(*paths: Path) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            total += path.stat().st_size
            continue
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return total

