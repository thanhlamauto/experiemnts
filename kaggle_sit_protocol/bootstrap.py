from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .config import ProtocolConfig
from .io_utils import estimate_bytes
from .manifest import build_manifest
from .modeling import (
    block_shuffle,
    encode_images_to_latents,
    forward_features,
    linear_path_xt,
    load_sit_model,
    load_vae,
    patch_shuffle,
    preprocess_pil_image,
)


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess_pil_image(image, image_size=image_size)


def _encode_batch(vae: torch.nn.Module, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        latents = encode_images_to_latents(vae, batch)
    return latents.detach().cpu().to(torch.float16)


def _smoke_test(
    config: ProtocolConfig,
    device: torch.device,
    main_latents: torch.Tensor,
    main_noise: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    model = load_sit_model(config, device)
    smoke_count = min(config.smoke_images, main_latents.shape[0])
    timestep_positions = list(config.smoke_timestep_positions)
    time_values = [config.time_values[pos] for pos in timestep_positions]

    x1 = main_latents[:smoke_count].to(device=device, dtype=torch.float32)
    x0 = main_noise[:smoke_count].to(device=device, dtype=torch.float32)
    y = labels[:smoke_count].to(device=device, dtype=torch.long)

    xt = linear_path_xt(x1, x0, time_values)
    xt = xt.permute(1, 0, 2, 3, 4).reshape(smoke_count * len(time_values), *x1.shape[1:])
    t = torch.tensor(time_values, device=device).repeat(smoke_count)
    y_rep = y.repeat_interleave(len(time_values))
    patch_tokens, block_tokens = forward_features(model, xt, t, y_rep)

    return {
        "smoke_images": smoke_count,
        "smoke_timesteps": len(time_values),
        "patch_tokens_shape": list(patch_tokens.shape),
        "block_tokens_shape": list(block_tokens.shape),
    }


def run_bootstrap(config: ProtocolConfig, stage_dir: Path) -> dict[str, object]:
    device = _device_from_config(config)
    manifest = build_manifest(config)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    manifest.to_csv(config.cache_dir / "manifest.csv", index=False)

    vae = load_vae(config, device)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    main_rows = manifest[manifest["subset_role"] == "main"].reset_index(drop=True)
    control_rows = manifest[manifest["subset_role"] == "control"].reset_index(drop=True)

    main_images: list[torch.Tensor] = []
    main_ids: list[str] = []
    for _, row in main_rows.iterrows():
        main_images.append(_load_image_tensor(str(row["absolute_path"]), config.image_size))
        main_ids.append(str(row["image_id"]))
    control_images: list[torch.Tensor] = []
    control_patch: list[torch.Tensor] = []
    control_block: list[torch.Tensor] = []
    control_ids: list[str] = []
    for control_index, (_, row) in enumerate(control_rows.iterrows()):
        tensor = _load_image_tensor(str(row["absolute_path"]), config.image_size)
        control_images.append(tensor)
        control_patch.append(patch_shuffle(tensor, patch_size=16, seed=config.seed + control_index))
        control_block.append(block_shuffle(tensor, block_grid=4, seed=config.seed + control_index))
        control_ids.append(str(row["image_id"]))

    main_latents = []
    for start in range(0, len(main_images), config.latent_batch_size):
        batch = main_images[start : start + config.latent_batch_size]
        main_latents.append(_encode_batch(vae, batch, device))
    control_latents = []
    control_patch_latents = []
    control_block_latents = []
    for start in range(0, len(control_images), config.latent_batch_size):
        control_latents.append(_encode_batch(vae, control_images[start : start + config.latent_batch_size], device))
        control_patch_latents.append(
            _encode_batch(vae, control_patch[start : start + config.latent_batch_size], device)
        )
        control_block_latents.append(
            _encode_batch(vae, control_block[start : start + config.latent_batch_size], device)
        )

    main_latents_t = torch.cat(main_latents, dim=0)
    control_latents_t = torch.cat(control_latents, dim=0)
    control_patch_t = torch.cat(control_patch_latents, dim=0)
    control_block_t = torch.cat(control_block_latents, dim=0)

    main_noise = torch.randn(main_latents_t.shape, generator=generator, dtype=torch.float32).to(torch.float16)
    control_noise = torch.randn(control_latents_t.shape, generator=generator, dtype=torch.float32).to(torch.float16)

    torch.save(
        {
            "main_image_ids": main_ids,
            "control_image_ids": control_ids,
            "main_latents": main_latents_t,
            "control_latents": control_latents_t,
            "control_patchshuffle_latents": control_patch_t,
            "control_blockshuffle_latents": control_block_t,
        },
        config.cache_dir / "x1_latents_fp16.pt",
    )
    torch.save(
        {
            "main_image_ids": main_ids,
            "control_image_ids": control_ids,
            "main_noise": main_noise,
            "control_noise": control_noise,
        },
        config.cache_dir / "x0_noise_seed0_fp16.pt",
    )

    smoke = _smoke_test(
        config,
        device,
        main_latents_t,
        main_noise,
        torch.as_tensor(main_rows["imagenet_idx"].to_numpy(), dtype=torch.long),
    )

    output_bytes = estimate_bytes(config.cache_dir)
    done = {
        "stage": "bootstrap",
        "manifest_path": str(manifest_path),
        "cache_dir": str(config.cache_dir),
        "dataset_root": str(manifest["dataset_root"].iloc[0]),
        "output_bytes": int(output_bytes),
    }
    sanity = {
        "num_main_images": int(len(main_rows)),
        "num_control_images": int(len(control_rows)),
        "num_preview_images": int(manifest["preview"].sum()),
        "num_synsets": int(manifest["synset"].nunique()),
        "smoke": smoke,
    }
    pd.DataFrame(
        [
            {"artifact": "manifest.parquet", "bytes": manifest_path.stat().st_size},
            {"artifact": "x1_latents_fp16.pt", "bytes": (config.cache_dir / "x1_latents_fp16.pt").stat().st_size},
            {"artifact": "x0_noise_seed0_fp16.pt", "bytes": (config.cache_dir / "x0_noise_seed0_fp16.pt").stat().st_size},
        ]
    ).to_csv(stage_dir / "bootstrap_artifacts.csv", index=False)
    return {"done": done, "sanity": sanity}
