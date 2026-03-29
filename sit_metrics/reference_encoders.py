"""Helpers for pretrained reference encoders used in representation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def dinov2_input_resolution(image_resolution: int) -> int:
    return 224 * max(1, image_resolution // 256)


@torch.no_grad()
def load_dinov2_model(
    model_name: str,
    device: torch.device,
    image_resolution: int,
) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    input_resolution = dinov2_input_resolution(image_resolution)
    if input_resolution != 224:
        import timm

        patch_resolution = input_resolution // 14
        model.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            model.pos_embed.data,
            [patch_resolution, patch_resolution],
        )
    model = model.to(device)
    model.eval()
    return model


def _prepare_dinov2_images(images: torch.Tensor, image_resolution: int) -> torch.Tensor:
    x = images.float()
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    input_resolution = dinov2_input_resolution(image_resolution)
    if x.shape[-1] != input_resolution or x.shape[-2] != input_resolution:
        x = F.interpolate(x, size=(input_resolution, input_resolution), mode="bicubic", align_corners=False)
    return x


@torch.no_grad()
def dinov2_patch_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_resolution: int,
) -> torch.Tensor:
    """
    images: [B, 3, H, W] in [-1, 1].
    Returns normalized patch tokens [B, T, D].
    """
    x = _prepare_dinov2_images(images, image_resolution)
    feats = model.forward_features(x)
    if not isinstance(feats, dict) or "x_norm_patchtokens" not in feats:
        raise ValueError("DINOv2 forward_features did not return patch tokens")
    return feats["x_norm_patchtokens"]


@torch.no_grad()
def dinov2_global_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_resolution: int,
    feature_kind: str = "mean_patch",
) -> torch.Tensor:
    """
    images: [B, 3, H, W] in [-1, 1].
    Returns one global vector per image from DINOv2.
    """
    x = _prepare_dinov2_images(images, image_resolution)
    feats = model.forward_features(x)
    if not isinstance(feats, dict):
        return feats
    if feature_kind == "cls":
        return feats["x_norm_clstoken"]
    if feature_kind == "mean_patch":
        return feats["x_norm_patchtokens"].mean(dim=1)
    raise ValueError(f"Unsupported DINO feature kind: {feature_kind}")
