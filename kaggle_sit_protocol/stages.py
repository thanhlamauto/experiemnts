from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .config import ProtocolConfig
from .io_utils import estimate_bytes, mark_done, mark_sanity, reset_directory, stage_is_complete
from .manifest import build_manifest, discover_dataset_root


@dataclass
class AnalysisRuntime:
    config: ProtocolConfig
    manifest: pd.DataFrame
    latents: dict[str, torch.Tensor]
    noise: dict[str, torch.Tensor]
    bases: dict[str, torch.Tensor]
    device: torch.device
    model: torch.nn.Module
    index: object | None = None


def _stage_dir(config: ProtocolConfig, name: str) -> Path:
    return config.output_dir / name


def run_bootstrap_stage(config: ProtocolConfig) -> dict[str, object]:
    from .bootstrap import run_bootstrap

    config.ensure_directories()
    stage_dir = _stage_dir(config, "00_bootstrap_and_manifest")
    if stage_is_complete(stage_dir) and not config.overwrite:
        return {"status": "skipped", "stage_dir": str(stage_dir)}
    reset_directory(stage_dir)
    payload = run_bootstrap(config, stage_dir)
    mark_done(stage_dir, payload["done"])
    mark_sanity(stage_dir, payload["sanity"])
    return payload


def run_task0_stage(config: ProtocolConfig) -> dict[str, object]:
    from .task0 import run_task0

    config.ensure_directories()
    stage_dir = _stage_dir(config, "01_task0_cache")
    if stage_is_complete(stage_dir) and not config.overwrite:
        return {"status": "skipped", "stage_dir": str(stage_dir)}
    reset_directory(stage_dir)
    payload = run_task0(config, stage_dir)
    mark_done(stage_dir, payload["done"])
    mark_sanity(stage_dir, payload["sanity"])
    return payload


def create_analysis_runtime(config: ProtocolConfig) -> AnalysisRuntime:
    from .runtime import load_runtime

    return load_runtime(config)


def run_analysis_stage(config: ProtocolConfig) -> dict[str, object]:
    from .tasks import run_all_tasks

    config.ensure_directories()
    stage_dir = _stage_dir(config, "02_tasks_1_to_10")
    if stage_is_complete(stage_dir) and not config.overwrite:
        return {"status": "skipped", "stage_dir": str(stage_dir)}
    reset_directory(stage_dir)
    runtime = create_analysis_runtime(config)
    payload = run_all_tasks(config, runtime, stage_dir)
    payload["done"]["output_bytes"] = estimate_bytes(config.output_dir)
    mark_done(stage_dir, payload["done"])
    mark_sanity(stage_dir, payload["sanity"])
    return payload
