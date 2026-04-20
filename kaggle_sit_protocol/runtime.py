from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from .config import ProtocolConfig
from .modeling import load_sit_model
from .stages import AnalysisRuntime


@dataclass
class RuntimeIndex:
    main_positions: dict[str, int]
    control_positions: dict[str, int]


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_runtime_index(latents: dict[str, torch.Tensor]) -> RuntimeIndex:
    return RuntimeIndex(
        main_positions={image_id: idx for idx, image_id in enumerate(latents["main_image_ids"])},
        control_positions={image_id: idx for idx, image_id in enumerate(latents["control_image_ids"])},
    )


def load_runtime(config: ProtocolConfig) -> AnalysisRuntime:
    manifest = pd.read_parquet(config.manifest_path)
    latents = torch.load(config.cache_dir / "x1_latents_fp16.pt", map_location="cpu")
    noise = torch.load(config.cache_dir / "x0_noise_seed0_fp16.pt", map_location="cpu")
    bases = torch.load(config.cache_dir / "tsvd_v64_basis_fp16.pt", map_location="cpu")
    device = _device_from_config(config)
    model = load_sit_model(config, device)
    runtime = AnalysisRuntime(
        config=config,
        manifest=manifest,
        latents=latents,
        noise=noise,
        bases=bases,
        device=device,
        model=model,
    )
    runtime.index = build_runtime_index(latents)  # type: ignore[attr-defined]
    return runtime
