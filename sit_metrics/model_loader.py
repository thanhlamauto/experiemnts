"""Load SiT or REPA SiT-XL/2 checkpoint (same conventions as plot_sit_xl2_activation_layers.py)."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "", 1) if k.startswith("module.") else k
        out[nk] = v
    return out


def load_checkpoint_raw(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict):
        if "ema" in obj and isinstance(obj["ema"], dict):
            return obj["ema"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if any("blocks.0.attn" in k for k in obj.keys()):
            return obj
        for _k, v in obj.items():
            if isinstance(v, dict) and v and isinstance(next(iter(v.values())), torch.Tensor):
                return v
    raise ValueError(f"Could not parse checkpoint at {path}")


def load_repa_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_legacy_checkpoints(state_dict: Dict[str, torch.Tensor], encoder_depth: int) -> Dict[str, torch.Tensor]:
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "decoder_blocks" in key:
            parts = key.split(".")
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = "blocks"
            parts[1] = str(new_idx)
            key = ".".join(parts)
        new_state_dict[key] = value
    return new_state_dict


def load_model(
    backend: str,
    ckpt: str,
    device: torch.device,
    sit_root: str = "/workspace/SiT",
    repa_root: str = "/workspace/REPA",
    resolution: int = 256,
    num_classes: int = 1000,
    model_name: str = "SiT-XL/2",
    learn_sigma: bool = True,
    repa_args_json: Optional[str] = None,
    encoder_depth: Optional[int] = None,
    projector_embed_dims: Optional[str] = None,
    use_cfg: Optional[bool] = None,
    fused_attn: Optional[bool] = None,
    qk_norm: Optional[bool] = None,
    legacy: bool = False,
) -> torch.nn.Module:
    latent_size = resolution // 8

    if backend == "sit":
        sit_root_path = Path(sit_root)
        download_mod = load_module_from_path("sit_download_local", sit_root_path / "download.py")
        sit_models_mod = load_module_from_path("sit_models_local", sit_root_path / "models.py")
        find_model = download_mod.find_model
        SiT_models = sit_models_mod.SiT_models

        m = SiT_models[model_name](
            input_size=latent_size,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        ).to(device)
        sd = find_model(ckpt)
        m.load_state_dict(sd, strict=False)
        return m

    repa_root_path = Path(repa_root)
    repa_models_mod = load_module_from_path("repa_models_sit_local", repa_root_path / "models" / "sit.py")
    SiT_models = repa_models_mod.SiT_models

    j = load_repa_json(repa_args_json)
    enc_d = encoder_depth if encoder_depth is not None else int(j.get("encoder_depth", 8))
    fused = fused_attn if fused_attn is not None else bool(j.get("fused_attn", True))
    qk = qk_norm if qk_norm is not None else bool(j.get("qk_norm", False))
    res = int(j.get("resolution", resolution))
    latent_size = res // 8
    cfg_prob = float(j.get("cfg_prob", 0.1))
    use_cfg_r = use_cfg if use_cfg is not None else (cfg_prob > 0)

    if projector_embed_dims:
        z_dims = [int(x) for x in projector_embed_dims.split(",") if x.strip()]
    else:
        z_dims = [768]

    block_kwargs = {"fused_attn": fused, "qk_norm": qk}
    m = SiT_models[model_name](
        input_size=latent_size,
        num_classes=int(j.get("num_classes", num_classes)),
        use_cfg=use_cfg_r,
        z_dims=z_dims,
        encoder_depth=enc_d,
        **block_kwargs,
    ).to(device)

    sd = load_checkpoint_raw(ckpt, map_location="cpu")
    sd = strip_module_prefix(sd)
    if legacy:
        sd = load_legacy_checkpoints(sd, enc_d)
    m.load_state_dict(sd, strict=False)
    return m
