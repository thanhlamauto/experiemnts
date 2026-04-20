from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch

from .config import ProtocolConfig


def _ensure_sit_on_path(config: ProtocolConfig) -> None:
    sit_root = Path(config.sit_root).resolve()
    if str(sit_root) not in sys.path:
        sys.path.insert(0, str(sit_root))
    repo_root = sit_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _resolve_hf_token(config: ProtocolConfig) -> str | None:
    if config.hf_token:
        return str(config.hf_token)
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def _configure_hf_auth(config: ProtocolConfig) -> str | None:
    token = _resolve_hf_token(config)
    if not token:
        return None
    os.environ.setdefault("HF_TOKEN", token)
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False, skip_if_logged_in=True)
    except Exception:
        pass
    return token


def load_sit_model(config: ProtocolConfig, device: torch.device) -> torch.nn.Module:
    _ensure_sit_on_path(config)
    _configure_hf_auth(config)
    from SiT.download import find_model
    from SiT.models import SiT_models

    model = SiT_models[config.model_name](
        input_size=config.latent_size,
        num_classes=config.num_classes,
        learn_sigma=config.learn_sigma,
    ).to(device)
    ckpt = config.checkpoint_path or "SiT-XL-2-256x256.pt"
    state_dict = find_model(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_vae(config: ProtocolConfig, device: torch.device) -> torch.nn.Module:
    from diffusers.models import AutoencoderKL

    token = _configure_hf_auth(config)
    if token:
        try:
            vae = AutoencoderKL.from_pretrained(config.vae_model, token=token).to(device)
        except TypeError:
            vae = AutoencoderKL.from_pretrained(config.vae_model, use_auth_token=token).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(config.vae_model).to(device)
    vae.eval()
    return vae


def preprocess_pil_image(image, image_size: int):
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")

    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)

    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)
    width, height = image.size
    crop_x = (width - image_size) // 2
    crop_y = (height - image_size) // 2
    image = image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))
    arr = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    arr = arr * 2.0 - 1.0
    return arr


@torch.no_grad()
def encode_images_to_latents(
    vae: torch.nn.Module,
    image_batch: torch.Tensor,
    scale: float = 0.18215,
) -> torch.Tensor:
    latents = vae.encode(image_batch).latent_dist.sample()
    return latents.mul_(scale)


def patch_shuffle(image: torch.Tensor, patch_size: int = 16, seed: int = 0) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {tuple(image.shape)}")
    channels, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size for patch shuffle.")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = (
        image.view(channels, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
        .view(grid_h * grid_w, channels, patch_size, patch_size)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(grid_h * grid_w, generator=generator)
    shuffled = patches[perm]
    return (
        shuffled.view(grid_h, grid_w, channels, patch_size, patch_size)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(channels, height, width)
    )


def block_shuffle(image: torch.Tensor, block_grid: int = 4, seed: int = 0) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {tuple(image.shape)}")
    channels, height, width = image.shape
    if height % block_grid != 0 or width % block_grid != 0:
        raise ValueError("Image dimensions must be divisible by block_grid for block shuffle.")
    block_h = height // block_grid
    block_w = width // block_grid
    blocks = (
        image.view(channels, block_grid, block_h, block_grid, block_w)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
        .view(block_grid * block_grid, channels, block_h, block_w)
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(block_grid * block_grid, generator=generator)
    shuffled = blocks[perm]
    return (
        shuffled.view(block_grid, block_grid, channels, block_h, block_w)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(channels, height, width)
    )


@torch.no_grad()
def compute_patch_tokens(model: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    return model.x_embedder(latents) + model.pos_embed


@torch.no_grad()
def forward_features(
    model: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.x_embedder(latents) + model.pos_embed
    patch_tokens = x.clone()
    t = model.t_embedder(timesteps)
    y = model.y_embedder(labels, train=False)
    c = t + y
    blocks: list[torch.Tensor] = []
    for block in model.blocks:
        x = block(x, c)
        blocks.append(x.clone())
    return patch_tokens, torch.stack(blocks, dim=0)


def linear_path_xt(x1: torch.Tensor, x0: torch.Tensor, timesteps: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(timesteps, torch.Tensor):
        t = timesteps.to(device=x1.device, dtype=x1.dtype)
    else:
        t = torch.tensor(list(timesteps), device=x1.device, dtype=x1.dtype)
    expand = (t.shape[0],) + (1,) * x1.ndim
    t = t.view(*expand)
    return t * x1.unsqueeze(0) + (1.0 - t) * x0.unsqueeze(0)


def time_index_to_float(index: int, total: int) -> float:
    return float(index) / float(max(total - 1, 1))


def token_grid_size(num_tokens: int) -> tuple[int, int]:
    grid = int(math.sqrt(num_tokens))
    if grid * grid != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square grid.")
    return grid, grid
