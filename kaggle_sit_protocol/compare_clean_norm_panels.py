from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.decomposition import spatial_normalize_tokens
    from kaggle_sit_protocol.model_specs import apply_sit_model_spec
    from kaggle_sit_protocol.modeling import compute_patch_tokens, load_sit_model, load_vae
    from kaggle_sit_protocol.progress import progress
else:
    from .config import ProtocolConfig
    from .decomposition import spatial_normalize_tokens
    from .model_specs import apply_sit_model_spec
    from .modeling import compute_patch_tokens, load_sit_model, load_vae
    from .progress import progress


DEFAULT_METHODS = (
    "identity",
    "spatial_norm",
    "local_contrast",
    "group_norm",
    "patchwise_norm",
    "high_pass",
    "dog",
    "laplacian",
)
ALL_METHODS = DEFAULT_METHODS + (
    "instance_norm",
    "pca_whiten",
)
METHOD_LABELS = {
    "identity": "identity",
    "spatial_norm": "spatial norm",
    "instance_norm": "instance norm",
    "local_contrast": "local contrast",
    "group_norm": "group norm",
    "patchwise_norm": "patch-wise norm",
    "high_pass": "high-pass",
    "dog": "DoG",
    "laplacian": "laplacian",
    "pca_whiten": "PCA whiten",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render clean-panel PCA sweeps for blurred images under multiple normalization "
            "or filtering schemes on VAE latent32 and SiT patchify0 tokens."
        )
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Protocol output root containing cache/manifest from bootstrap/task0.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional explicit manifest override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for rendered outputs. Defaults to <output_root>/analysis/clean_norm_panels.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional ProtocolConfig.model_name override.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit SiT checkpoint path override.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Override ProtocolConfig.device.",
    )
    parser.add_argument(
        "--subset-role",
        choices=("main", "control"),
        default="main",
        help="Which manifest subset to select images from.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Restrict selection to manifest rows with preview=True.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of selected images when --image-ids is not provided.",
    )
    parser.add_argument(
        "--image-ids",
        type=str,
        default=None,
        help="Comma-separated explicit image_id list. Overrides --max-images and --preview-only ordering.",
    )
    parser.add_argument(
        "--blur-radii",
        type=str,
        default="0,0.5,1,2,4,8",
        help="Comma-separated Gaussian blur radii applied to the RGB input after resize/crop.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="default",
        help=(
            "Comma-separated methods, or one of {default,all}. "
            "Note: spatial_norm and instance_norm are equivalent here up to numerical precision."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="latent32,patchify0",
        help="Comma-separated stages to render from {latent32,patchify0}.",
    )
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Use posterior.sample() instead of posterior.mode() for the VAE latent.",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=5,
        help="Odd kernel size for local contrast normalization.",
    )
    parser.add_argument(
        "--group-count",
        type=int,
        default=32,
        help="Requested max group count for GroupNorm. Uses the largest divisor <= this value.",
    )
    parser.add_argument(
        "--patchwise-size",
        type=int,
        default=4,
        help="Patch size for non-overlapping patch-wise normalization.",
    )
    parser.add_argument(
        "--highpass-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for high-pass filtering x - blur(x).",
    )
    parser.add_argument(
        "--dog-sigma-small",
        type=float,
        default=1.0,
        help="Smaller Gaussian sigma for DoG.",
    )
    parser.add_argument(
        "--dog-sigma-large",
        type=float,
        default=2.0,
        help="Larger Gaussian sigma for DoG.",
    )
    parser.add_argument(
        "--whiten-max-components",
        type=int,
        default=128,
        help="Max PCA components for the optional pca_whiten method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for PCA whitening and image ordering where applicable.",
    )
    return parser.parse_args()


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_csv_strings(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _resolve_methods(methods_arg: str) -> list[str]:
    token = str(methods_arg).strip().lower()
    if token == "default":
        return list(DEFAULT_METHODS)
    if token == "all":
        return list(ALL_METHODS)
    methods = _parse_csv_strings(methods_arg)
    if not methods:
        raise ValueError("At least one method must be requested.")
    unknown = sorted(set(methods).difference(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {list(ALL_METHODS)}")
    return methods


def _resolve_stages(stages_arg: str) -> list[str]:
    stages = _parse_csv_strings(stages_arg)
    allowed = {"latent32", "patchify0"}
    unknown = sorted(set(stages).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Supported: {sorted(allowed)}")
    if not stages:
        raise ValueError("At least one stage must be requested.")
    return stages


def _apply_overrides(config: ProtocolConfig, args: argparse.Namespace) -> None:
    if args.output_root is not None:
        config.output_root = str(args.output_root)
        config.manifest_path = f"{config.output_root}/cache/manifest.parquet"
    if args.manifest_path is not None:
        config.manifest_path = str(args.manifest_path)
    if args.model_name is not None:
        config.model_name = str(args.model_name)
    if args.checkpoint_path is not None:
        config.checkpoint_path = str(args.checkpoint_path)
    if args.device is not None:
        config.device = str(args.device)
    apply_sit_model_spec(config)
    config.ensure_directories()


def _prepare_pil_image(image: Image.Image, image_size: int) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)
    width, height = image.size
    crop_x = (width - image_size) // 2
    crop_y = (height - image_size) // 2
    return image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    return array * 2.0 - 1.0


def _select_rows(config: ProtocolConfig, args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_parquet(config.manifest_path)
    rows = manifest[manifest["subset_role"] == args.subset_role].copy()
    explicit_ids = _parse_csv_strings(args.image_ids)

    if explicit_ids:
        rows = rows[rows["image_id"].isin(explicit_ids)].copy()
        if rows.empty:
            raise ValueError(f"No rows matched requested image_ids={explicit_ids}")
        rows["image_id"] = pd.Categorical(rows["image_id"], categories=explicit_ids, ordered=True)
        rows = rows.sort_values("image_id").reset_index(drop=True)
        if len(rows) != len(explicit_ids):
            found = rows["image_id"].astype(str).tolist()
            missing = [image_id for image_id in explicit_ids if image_id not in found]
            raise ValueError(f"Missing requested image_ids in manifest: {missing}")
        return rows

    if args.preview_only:
        rows = rows[rows["preview"]].copy()
    rows = rows.sort_values(["class_index_100", "image_id"]).reset_index(drop=True)
    max_images = max(int(args.max_images), 1)
    return rows.head(max_images).reset_index(drop=True)


def _tensor_grid_size(tokens: torch.Tensor) -> int:
    side = int(round(math.sqrt(int(tokens.shape[0]))))
    if side * side != int(tokens.shape[0]):
        raise ValueError(f"Token count {int(tokens.shape[0])} is not a square grid.")
    return side


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected [C,H,W] feature map, got {tuple(feature_map.shape)}")
    channels, height, width = feature_map.shape
    return feature_map.permute(1, 2, 0).reshape(height * width, channels)


def _tokens_to_map(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected [N,D] token matrix, got {tuple(tokens.shape)}")
    if grid_size * grid_size != int(tokens.shape[0]):
        raise ValueError(f"Grid size {grid_size} incompatible with {int(tokens.shape[0])} tokens.")
    return tokens.reshape(grid_size, grid_size, tokens.shape[1]).permute(2, 0, 1).contiguous()


def _depthwise_gaussian_blur(feature_map: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return feature_map
    radius = max(int(round(3.0 * sigma)), 1)
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, device=feature_map.device, dtype=feature_map.dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(feature_map.shape[0], 1, 1, 1)
    pad_mode = "reflect"
    if radius >= min(int(feature_map.shape[-2]), int(feature_map.shape[-1])):
        pad_mode = "replicate"
    padded = F.pad(
        feature_map.unsqueeze(0),
        (radius, radius, radius, radius),
        mode=pad_mode,
    )
    blurred = F.conv2d(padded, kernel, groups=feature_map.shape[0])
    return blurred.squeeze(0)


def _largest_divisor_at_most(value: int, limit: int) -> int:
    limit = max(min(int(limit), int(value)), 1)
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _resolve_group_count(channels: int, requested_groups: int) -> int:
    limit = int(requested_groups)
    if channels <= 8:
        limit = min(limit, max(channels // 2, 1))
    return _largest_divisor_at_most(channels, limit)


def _local_contrast_normalize(feature_map: torch.Tensor, kernel_size: int, eps: float) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    batch = feature_map.unsqueeze(0)
    mean = F.avg_pool2d(batch, kernel_size=kernel_size, stride=1, padding=padding)
    second_moment = F.avg_pool2d(batch * batch, kernel_size=kernel_size, stride=1, padding=padding)
    variance = (second_moment - mean * mean).clamp_min(0.0)
    normalized = (batch - mean) / torch.sqrt(variance + eps)
    return normalized.squeeze(0)


def _patchwise_normalize(feature_map: torch.Tensor, patch_size: int, eps: float) -> torch.Tensor:
    channels, height, width = feature_map.shape
    patch_size = max(min(int(patch_size), height, width), 1)
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Patch size {patch_size} must divide stage grid {height}x{width} for patch-wise normalization."
        )
    patches = (
        feature_map.view(channels, height // patch_size, patch_size, width // patch_size, patch_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    mean = patches.mean(dim=(-1, -2), keepdim=True)
    variance = patches.var(dim=(-1, -2), keepdim=True, unbiased=False)
    normalized = (patches - mean) / torch.sqrt(variance + eps)
    return (
        normalized.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(channels, height, width)
    )


def _laplacian_filter(feature_map: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    weight = kernel.view(1, 1, 3, 3).repeat(feature_map.shape[0], 1, 1, 1)
    padded = F.pad(feature_map.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, weight, groups=feature_map.shape[0]).squeeze(0)


def _pca_whiten_tokens(tokens: torch.Tensor, max_components: int, seed: int) -> torch.Tensor:
    matrix = tokens.detach().cpu().numpy().astype(np.float32, copy=False)
    components = min(int(max_components), matrix.shape[0], matrix.shape[1])
    if components < 3:
        return tokens.detach().cpu().float()
    max_rank = min(matrix.shape[0], matrix.shape[1])
    solver = "full" if components == max_rank else "randomized"
    whitened = PCA(
        n_components=components,
        whiten=True,
        svd_solver=solver,
        random_state=seed,
    ).fit_transform(matrix)
    return torch.from_numpy(whitened.astype(np.float32, copy=False))


def _method_tokens(
    *,
    feature_map: torch.Tensor,
    method: str,
    config: ProtocolConfig,
    args: argparse.Namespace,
) -> torch.Tensor:
    tokens = _map_to_tokens(feature_map)
    if method in {"spatial_norm", "instance_norm"}:
        return spatial_normalize_tokens(
            tokens,
            gamma=config.spatial_norm_gamma,
            eps=config.spatial_norm_eps,
        ).detach().cpu().float()
    if method == "identity":
        return tokens.detach().cpu().float()
    if method == "pca_whiten":
        return _pca_whiten_tokens(tokens, max_components=args.whiten_max_components, seed=args.seed)

    if method == "local_contrast":
        mapped = _local_contrast_normalize(feature_map, kernel_size=args.local_window, eps=config.spatial_norm_eps)
    elif method == "group_norm":
        groups = _resolve_group_count(int(feature_map.shape[0]), args.group_count)
        mapped = F.group_norm(feature_map.unsqueeze(0), num_groups=groups, eps=config.spatial_norm_eps).squeeze(0)
    elif method == "patchwise_norm":
        mapped = _patchwise_normalize(feature_map, patch_size=args.patchwise_size, eps=config.spatial_norm_eps)
    elif method == "high_pass":
        mapped = feature_map - _depthwise_gaussian_blur(feature_map, sigma=float(args.highpass_sigma))
    elif method == "dog":
        mapped = _depthwise_gaussian_blur(feature_map, sigma=float(args.dog_sigma_small)) - _depthwise_gaussian_blur(
            feature_map,
            sigma=float(args.dog_sigma_large),
        )
    elif method == "laplacian":
        mapped = _laplacian_filter(feature_map)
    else:
        raise ValueError(f"Unsupported method={method!r}")
    return _map_to_tokens(mapped).detach().cpu().float()


def _local_pca_rgb(tokens: torch.Tensor, grid_size: int) -> np.ndarray:
    matrix = tokens.detach().cpu().numpy().astype(np.float32, copy=False)
    components = PCA(n_components=3).fit_transform(matrix)
    scaled = components.astype(np.float32, copy=True)
    for channel_index in range(3):
        channel = scaled[:, channel_index]
        min_value = float(channel.min())
        max_value = float(channel.max())
        if max_value - min_value <= 1e-6:
            channel.fill(0.0)
        else:
            channel -= min_value
            channel /= max_value - min_value
    return scaled.reshape(grid_size, grid_size, 3)


def _render_stage_page(
    *,
    pdf: PdfPages,
    image_id: str,
    stage_name: str,
    input_panels: list[np.ndarray],
    blur_radii: list[float],
    method_panels: dict[str, list[np.ndarray]],
    methods: list[str],
) -> None:
    row_labels = ["input blur"] + [METHOD_LABELS[method] for method in methods]
    nrows = len(row_labels)
    ncols = len(blur_radii)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.25 * ncols, 1.9 * nrows),
        dpi=150,
    )
    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 1:
        if nrows == 1:
            axes_arr = axes_arr.reshape(1, ncols)
        else:
            axes_arr = axes_arr.reshape(nrows, 1)

    for col_index, radius in enumerate(blur_radii):
        ax = axes_arr[0, col_index]
        ax.imshow(input_panels[col_index])
        ax.axis("off")
        ax.set_title(f"blur={radius:g}", fontsize=10)

    for row_index, method in enumerate(methods, start=1):
        panels = method_panels[method]
        for col_index in range(ncols):
            ax = axes_arr[row_index, col_index]
            ax.imshow(panels[col_index])
            ax.axis("off")

    for row_index, label in enumerate(row_labels):
        axes_arr[row_index, 0].text(
            -0.18,
            0.5,
            label,
            transform=axes_arr[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        f"{image_id} | {stage_name} | local PCA-RGB over blur sweep",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def _compute_stage_maps(
    *,
    model: torch.nn.Module,
    vae: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    sample_latent: bool,
) -> dict[str, torch.Tensor]:
    batch = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    posterior = vae.encode(batch).latent_dist
    latent = posterior.sample() if sample_latent else posterior.mode()
    latent = latent.mul(0.18215)

    patch_tokens = compute_patch_tokens(model, latent).squeeze(0).float()
    patch_grid = _tensor_grid_size(patch_tokens)

    return {
        "latent32": latent.squeeze(0).float(),
        "patchify0": _tokens_to_map(patch_tokens, patch_grid).float(),
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    config = ProtocolConfig.from_kaggle_defaults()
    _apply_overrides(config, args)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.analysis_dir / "clean_norm_panels"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(config, args)
    if rows.empty:
        raise ValueError("No images selected for rendering.")

    methods = _resolve_methods(args.methods)
    stages = _resolve_stages(args.stages)
    blur_radii = _parse_csv_floats(args.blur_radii)
    if not blur_radii:
        raise ValueError("At least one blur radius is required.")

    device = _device_from_config(config)
    print(f"Using device: {device}")
    print(f"Selected images: {len(rows)}")
    print(f"Stages: {stages}")
    print(f"Methods: {methods}")

    vae = load_vae(config, device)
    model = load_sit_model(config, device)

    stem = f"clean_norm_panels_{args.subset_role}_{len(rows)}img"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}_selected_images.csv"
    json_path = output_dir / f"{stem}_config.json"

    rows[["image_id", "absolute_path", "subset_role", "preview"]].to_csv(csv_path, index=False)
    json_payload = {
        "output_root": str(config.output_root),
        "manifest_path": str(config.manifest_path),
        "model_name": str(config.model_name),
        "checkpoint_path": config.checkpoint_path,
        "device": str(device),
        "subset_role": str(args.subset_role),
        "preview_only": bool(args.preview_only),
        "max_images": int(args.max_images),
        "image_ids": _parse_csv_strings(args.image_ids),
        "blur_radii": blur_radii,
        "stages": stages,
        "methods": methods,
        "sample_latent": bool(args.sample_latent),
        "local_window": int(args.local_window),
        "group_count": int(args.group_count),
        "patchwise_size": int(args.patchwise_size),
        "highpass_sigma": float(args.highpass_sigma),
        "dog_sigma_small": float(args.dog_sigma_small),
        "dog_sigma_large": float(args.dog_sigma_large),
        "whiten_max_components": int(args.whiten_max_components),
        "seed": int(args.seed),
        "notes": {
            "instance_norm_equivalence": (
                "spatial_norm and instance_norm are mathematically equivalent here because "
                "both normalize each channel over spatial positions for a single image."
            )
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)

    with PdfPages(pdf_path) as pdf:
        for _, row in progress(rows.iterrows(), desc="Render norm panels", total=len(rows)):
            input_panels: list[np.ndarray] = []
            stage_method_panels = {
                stage_name: {method: [] for method in methods}
                for stage_name in stages
            }

            for radius in blur_radii:
                with Image.open(str(row["absolute_path"])) as image:
                    image = _prepare_pil_image(image, config.image_size)
                    if float(radius) > 0:
                        image = image.filter(ImageFilter.GaussianBlur(radius=float(radius)))
                    input_panels.append(np.asarray(image))
                    image_tensor = _pil_to_tensor(image)

                stage_maps = _compute_stage_maps(
                    model=model,
                    vae=vae,
                    image_tensor=image_tensor,
                    device=device,
                    sample_latent=bool(args.sample_latent),
                )
                for stage_name in stages:
                    feature_map = stage_maps[stage_name]
                    grid_size = int(feature_map.shape[-1])
                    for method in methods:
                        transformed_tokens = _method_tokens(
                            feature_map=feature_map,
                            method=method,
                            config=config,
                            args=args,
                        )
                        stage_method_panels[stage_name][method].append(
                            _local_pca_rgb(transformed_tokens, grid_size=grid_size)
                        )

            for stage_name in stages:
                _render_stage_page(
                    pdf=pdf,
                    image_id=str(row["image_id"]),
                    stage_name=stage_name,
                    input_panels=input_panels,
                    blur_radii=blur_radii,
                    method_panels=stage_method_panels[stage_name],
                    methods=methods,
                )

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved image CSV: {csv_path}")
    print(f"Saved config JSON: {json_path}")


if __name__ == "__main__":
    main()
