#!/usr/bin/env python3
"""Visualize DINOv2 patch-token similarity before and after spatial normalization.

This script is intended for Kaggle usage with either:

  - a Hugging Face repo id such as `facebook/dinov2-base`, or
  - a local Hugging Face-style Flax checkpoint directory, for example:

  /kaggle/input/datasets/bangchi/dinov2-vitb14-flax

It loads the DINOv2 checkpoint with `transformers`, falls back to a local `.pkl`
Flax pytree when needed, extracts patch tokens, and reproduces the paper-style
visualization of cosine-similarity maps with and without spatial normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import shutil
import ssl
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

try:
    import pywt
except ImportError:
    pywt = None


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_ANCHORS = ((0.28, 0.30), (0.68, 0.33), (0.32, 0.66), (0.77, 0.71))


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def model_spec_to_local_path(model_spec: str) -> Path | None:
    path = Path(model_spec)
    if path.exists():
        return path
    return None


def infer_dinov2_config_kwargs_from_name(name: str) -> dict[str, float | int]:
    lower = name.lower()
    common = {
        "image_size": 224,
        "patch_size": 14,
        "num_channels": 3,
        "layer_norm_eps": 1e-6,
    }
    if "vits14" in lower or "small" in lower:
        return {
            **common,
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
        }
    if "vitl14" in lower or "large" in lower:
        return {
            **common,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        }
    if "vitg14" in lower or "giant" in lower:
        return {
            **common,
            "hidden_size": 1536,
            "num_hidden_layers": 40,
            "num_attention_heads": 24,
        }
    return {
        **common,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
    }


def find_local_flax_pickle(model_root: Path | None) -> Path | None:
    if model_root is None:
        return None
    if model_root.is_file() and model_root.suffix.lower() == ".pkl":
        return model_root
    if not model_root.is_dir():
        return None
    pkl_files = sorted(model_root.glob("*.pkl"))
    if len(pkl_files) == 1:
        return pkl_files[0]
    if len(pkl_files) > 1:
        for candidate in pkl_files:
            if "dinov2" in candidate.name.lower():
                return candidate
        return pkl_files[0]
    return None


def unwrap_pickled_flax_params(obj):
    try:
        from flax.core import FrozenDict
        from flax.core.frozen_dict import unfreeze
        from flax.traverse_util import unflatten_dict
    except ImportError:
        FrozenDict = None
        unfreeze = None
        unflatten_dict = None

    if FrozenDict is not None and isinstance(obj, FrozenDict):
        obj = unfreeze(obj)

    if hasattr(obj, "params"):
        obj = obj.params

    if isinstance(obj, dict):
        for key in ("params", "state_dict", "model"):
            if key in obj and len(obj[key]) > 0:
                obj = obj[key]
                break

    if isinstance(obj, dict) and unflatten_dict is not None:
        if obj and all(isinstance(key, tuple) for key in obj):
            obj = unflatten_dict(obj)
        elif obj and all(isinstance(key, str) and "/" in key for key in obj):
            flat = {tuple(key.split("/")): value for key, value in obj.items()}
            obj = unflatten_dict(flat)

    return obj


def load_pickled_flax_dinov2_model(pkl_path: Path):
    try:
        import jax
        import jax.numpy as jnp
        from transformers import Dinov2Config, FlaxDinov2Model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Loading a .pkl Flax checkpoint requires transformers, jax, and flax."
        ) from exc

    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    params = unwrap_pickled_flax_params(payload)
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value,
        params,
    )

    config = Dinov2Config(**infer_dinov2_config_kwargs_from_name(pkl_path.name))
    model = FlaxDinov2Model(config, _do_init=False)
    backend = jax.default_backend()
    print(f"[info] Loaded pickled Flax DINOv2 checkpoint from {pkl_path} (jax backend: {backend})")
    return SimpleNamespace(
        backend="flax_pkl",
        model=model,
        processor=None,
        params=params,
        config=config,
        source_path=pkl_path,
    )


def load_local_dinov2_model(model_spec: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, Dinov2Model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required. On Kaggle run: pip install -q transformers"
        ) from exc

    model_root = model_spec_to_local_path(model_spec)
    flax_pkl = find_local_flax_pickle(model_root)
    processor = None
    try:
        if model_root is not None and model_root.is_dir():
            processor = AutoImageProcessor.from_pretrained(str(model_root), local_files_only=True)
        else:
            processor = AutoImageProcessor.from_pretrained(model_spec)
    except Exception as exc:
        print(f"[warn] Failed to load AutoImageProcessor from {model_spec}: {exc}")

    last_error: Exception | None = None

    if model_root is not None and model_root.is_dir():
        for from_flax in (True, False):
            try:
                model = Dinov2Model.from_pretrained(
                    str(model_root),
                    from_flax=from_flax,
                    local_files_only=True,
                )
                model.eval().to(device)
                load_mode = "flax" if from_flax else "pytorch"
                print(f"[info] Loaded DINOv2 checkpoint from {model_root} ({load_mode} weights)")
                return SimpleNamespace(
                    backend="torch",
                    model=model,
                    processor=processor,
                    params=None,
                    config=model.config,
                    source_path=model_root,
                )
            except Exception as exc:
                last_error = exc
    elif model_root is None:
        try:
            model = Dinov2Model.from_pretrained(model_spec)
            model.eval().to(device)
            print(f"[info] Loaded DINOv2 checkpoint from Hugging Face Hub: {model_spec}")
            return SimpleNamespace(
                backend="torch",
                model=model,
                processor=processor,
                params=None,
                config=model.config,
                source_path=model_spec,
            )
        except Exception as exc:
            last_error = exc

    if flax_pkl is not None:
        return load_pickled_flax_dinov2_model(flax_pkl)

    raise SystemExit(f"Failed to load DINOv2 checkpoint from {model_spec}: {last_error}")


def is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _looks_like_ssl_verification_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _download_url_to_path(image_url: str, local_path: Path, ssl_context=None) -> None:
    request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, context=ssl_context) as response:
        with local_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def download_image_url(image_url: str, download_dir: Path, allow_insecure_download: bool) -> Path:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    local_path = download_dir / f"download_{digest}{suffix}"
    if local_path.exists():
        print(f"[info] Reusing downloaded image: {local_path}")
        return local_path

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading image from {image_url}")
    try:
        _download_url_to_path(image_url, local_path)
    except Exception as exc:
        if allow_insecure_download and _looks_like_ssl_verification_error(exc):
            print("[warn] SSL verification failed; retrying download without certificate verification")
            try:
                _download_url_to_path(
                    image_url,
                    local_path,
                    ssl_context=ssl._create_unverified_context(),
                )
                return local_path
            except Exception as retry_exc:
                raise SystemExit(
                    f"Failed to download image from {image_url} even with insecure SSL retry: {retry_exc}"
                ) from retry_exc
        if _looks_like_ssl_verification_error(exc):
            raise SystemExit(
                f"Failed to download image from {image_url}: {exc}\n"
                "The remote host has an invalid SSL certificate. "
                "If you trust this image source, rerun with --allow-insecure-download."
            ) from exc
        raise SystemExit(f"Failed to download image from {image_url}: {exc}") from exc
    return local_path


def materialize_image_specs(
    image_specs: list[str],
    download_dir: Path,
    allow_insecure_download: bool,
) -> list[Path]:
    image_paths: list[Path] = []
    for image_spec in image_specs:
        if is_http_url(image_spec):
            image_paths.append(
                download_image_url(
                    image_spec,
                    download_dir,
                    allow_insecure_download=allow_insecure_download,
                )
            )
        else:
            image_paths.append(Path(image_spec))
    return image_paths


def infer_image_size(processor, config, override_size: int | None) -> int:
    if override_size is not None:
        return int(override_size)

    for candidate in (
        getattr(config, "image_size", None),
        getattr(processor, "crop_size", None),
        getattr(processor, "size", None),
    ):
        size = extract_scalar_size(candidate)
        if size is not None:
            return size

    return 224


def extract_scalar_size(size_like) -> int | None:
    if size_like is None:
        return None
    if isinstance(size_like, int):
        return int(size_like)
    if isinstance(size_like, (tuple, list)) and size_like:
        return int(size_like[0])
    if isinstance(size_like, dict):
        for key in ("height", "width", "shortest_edge"):
            if key in size_like:
                return int(size_like[key])
    return None


def infer_mean_std(processor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    image_mean = getattr(processor, "image_mean", None)
    image_std = getattr(processor, "image_std", None)
    if image_mean is None or image_std is None:
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    return tuple(float(v) for v in image_mean), tuple(float(v) for v in image_std)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess_pil_image(
    image: Image.Image,
    image_size: int,
    blur_radius: float = 0.0,
    wavelet_mode: str | None = None,
    wavelet_name: str = "haar",
) -> Image.Image:
    image = center_crop_square(image).resize((image_size, image_size), resample=Image.BICUBIC)
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if wavelet_mode in ("low", "high"):
        if pywt is None:
            raise ImportError("PyWavelets is required for wavelet filtering. Run: pip install PyWavelets")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        channels = []
        for i in range(3):
            c = arr[:, :, i]
            # Use 1-level DWT to split low/high frequencies
            coeffs = pywt.wavedec2(c, wavelet_name, level=1)
            cA, (cH, cV, cD) = coeffs
            if wavelet_mode == "low":
                # Keep only LL (approximation)
                new_coeffs = [cA, (np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD))]
            else:  # high
                # Keep only Details
                new_coeffs = [np.zeros_like(cA), (cH, cV, cD)]

            recon = pywt.waverec2(new_coeffs, wavelet_name)
            # Ensure output shape matches in case of padding/odd dimensions
            recon = recon[: c.shape[0], : c.shape[1]]
            channels.append(recon)

        res = np.stack(channels, axis=-1)
        if wavelet_mode == "high":
            # Shift high-frequency detail for better visualization in [0, 1] range
            res = (res + 0.5).clip(0, 1)
        else:
            res = res.clip(0, 1)
        image = Image.fromarray((res * 255).astype(np.uint8))

    return image


def load_and_prepare_image(
    image_path: Path,
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    _device: torch.device,
    blur_radius: float = 0.0,
    wavelet_mode: str | None = None,
    wavelet_name: str = "haar",
) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image = preprocess_pil_image(
        image,
        image_size=image_size,
        blur_radius=blur_radius,
        wavelet_mode=wavelet_mode,
        wavelet_name=wavelet_name,
    )

    display = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(display).permute(2, 0, 1).contiguous()

    mean_t = torch.tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
    pixel_values = ((tensor - mean_t) / std_t).unsqueeze(0).cpu().numpy()
    return display, pixel_values


@torch.no_grad()
def extract_patch_tokens(model_bundle, pixel_values: np.ndarray, device: torch.device) -> np.ndarray:
    if model_bundle.backend == "torch":
        pixel_values_t = torch.from_numpy(pixel_values).to(device)
        outputs = model_bundle.model(pixel_values=pixel_values_t)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError(f"Unexpected DINOv2 hidden state shape: {tuple(hidden.shape)}")
        return hidden[:, 1:, :].detach().cpu().numpy()

    if model_bundle.backend == "flax_pkl":
        import jax.numpy as jnp

        outputs = model_bundle.model(
            pixel_values=jnp.asarray(pixel_values),
            params=model_bundle.params,
            train=False,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        hidden = np.asarray(hidden)
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError(f"Unexpected Flax DINOv2 hidden state shape: {tuple(hidden.shape)}")
        return hidden[:, 1:, :]

    raise ValueError(f"Unsupported model backend: {model_bundle.backend}")


def grid_size_from_tokens(num_tokens: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Expected a square patch grid, got {num_tokens} tokens")
    return side, side


def spatial_normalize_tokens(tokens: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    tokens = tokens - gamma * tokens.mean(dim=1, keepdim=True)
    tokens = tokens / (tokens.std(dim=1, keepdim=True) + eps)
    return tokens


def parse_anchor_points(spec: str | None) -> list[tuple[float, float]]:
    if spec is None or not spec.strip():
        return [(float(x), float(y)) for x, y in DEFAULT_ANCHORS]

    anchors: list[tuple[float, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = [piece.strip() for piece in chunk.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid anchor specification: {chunk!r}")
        x, y = float(pieces[0]), float(pieces[1])
        anchors.append((x, y))
    if not anchors:
        raise ValueError("At least one anchor is required")
    return anchors


def resolve_anchor_specs(anchor_specs: list[str] | None, num_images: int) -> list[list[tuple[float, float]]]:
    specs = list(anchor_specs or [])
    if not specs:
        return [parse_anchor_points(None) for _ in range(num_images)]
    if len(specs) == 1:
        anchors = parse_anchor_points(specs[0])
        return [anchors for _ in range(num_images)]
    if len(specs) != num_images:
        raise ValueError(
            f"Expected either 1 --anchors spec or {num_images} specs, got {len(specs)}"
        )
    return [parse_anchor_points(spec) for spec in specs]


def normalized_anchor_to_grid(
    anchor_xy: tuple[float, float],
    grid_width: int,
    grid_height: int,
) -> tuple[int, int]:
    x = min(max(anchor_xy[0], 0.0), 1.0)
    y = min(max(anchor_xy[1], 0.0), 1.0)
    col = int(round(x * (grid_width - 1)))
    row = int(round(y * (grid_height - 1)))
    return row, col


def anchor_center_in_pixels(
    anchor_rc: tuple[int, int],
    image_size: int,
    grid_width: int,
    grid_height: int,
) -> tuple[float, float]:
    row, col = anchor_rc
    x = (col + 0.5) * image_size / grid_width
    y = (row + 0.5) * image_size / grid_height
    return x, y


def compute_similarity_maps(
    tokens: torch.Tensor,
    anchors_rc: list[tuple[int, int]],
    grid_width: int,
    grid_height: int,
) -> list[np.ndarray]:
    if tokens.ndim != 2:
        raise ValueError(f"Expected [T, D] tokens, got {tuple(tokens.shape)}")

    normalized = F.normalize(tokens, dim=-1)
    similarity_maps: list[np.ndarray] = []
    for row, col in anchors_rc:
        flat_index = row * grid_width + col
        anchor = normalized[flat_index]
        scores = normalized @ anchor
        similarity_maps.append(scores.view(grid_height, grid_width).detach().cpu().numpy())
    return similarity_maps


def build_condition_record(
    *,
    title: str,
    model_bundle,
    image_path: Path,
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    device: torch.device,
    anchors_rc: list[tuple[int, int]] | None = None,
    anchor_points_xy: list[tuple[float, float]] | None = None,
    gamma: float = 1.0,
    blur_radius: float = 0.0,
    wavelet_mode: str | None = None,
    wavelet_name: str = "haar",
    apply_spatial_norm: bool = False,
) -> dict[str, object]:
    display_image, pixel_values = load_and_prepare_image(
        image_path,
        image_size,
        mean,
        std,
        device,
        blur_radius=blur_radius,
        wavelet_mode=wavelet_mode,
        wavelet_name=wavelet_name,
    )
    tokens = torch.from_numpy(extract_patch_tokens(model_bundle, pixel_values, device)[0]).float()
    grid_height, grid_width = grid_size_from_tokens(tokens.shape[0])

    if anchors_rc is None:
        if anchor_points_xy is None:
            raise ValueError("Either anchors_rc or anchor_points_xy must be provided")
        anchors_rc = [normalized_anchor_to_grid(anchor, grid_width, grid_height) for anchor in anchor_points_xy]

    if apply_spatial_norm:
        tokens = spatial_normalize_tokens(tokens.unsqueeze(0), gamma=gamma)[0]

    return {
        "title": title,
        "display_image": display_image,
        "grid_shape": (grid_height, grid_width),
        "anchors_rc": anchors_rc,
        "anchors_px": [
            anchor_center_in_pixels(anchor_rc, image_size, grid_width, grid_height)
            for anchor_rc in anchors_rc
        ],
        "sim_maps": compute_similarity_maps(tokens, anchors_rc, grid_width, grid_height),
    }


def build_visualization_record(
    model_bundle,
    image_path: Path,
    anchor_points_xy: list[tuple[float, float]],
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    gamma: float,
    device: torch.device,
    compare_gaussian_blur: bool,
    blur_radius: float,
    compare_wavelet: bool = False,
    wavelet_name: str = "haar",
) -> dict[str, object]:
    baseline = build_condition_record(
        title="Original Input",
        model_bundle=model_bundle,
        image_path=image_path,
        image_size=image_size,
        mean=mean,
        std=std,
        device=device,
        anchor_points_xy=anchor_points_xy,
    )

    conditions = [
        baseline,
        build_condition_record(
            title="Spatial Normalization",
            model_bundle=model_bundle,
            image_path=image_path,
            image_size=image_size,
            mean=mean,
            std=std,
            device=device,
            anchors_rc=baseline["anchors_rc"],
            gamma=gamma,
            apply_spatial_norm=True,
        ),
    ]
    if compare_gaussian_blur:
        conditions.append(
            build_condition_record(
                title=f"Gaussian Blur Input (r={blur_radius:g})",
                model_bundle=model_bundle,
                image_path=image_path,
                image_size=image_size,
                mean=mean,
                std=std,
                device=device,
                anchors_rc=baseline["anchors_rc"],
                blur_radius=blur_radius,
            )
        )

    if compare_wavelet:
        for mode in ("low", "high"):
            label = "Low Freq" if mode == "low" else "High Freq"
            conditions.append(
                build_condition_record(
                    title=f"DWT {label} ({wavelet_name})",
                    model_bundle=model_bundle,
                    image_path=image_path,
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    device=device,
                    anchors_rc=baseline["anchors_rc"],
                    wavelet_mode=mode,
                    wavelet_name=wavelet_name,
                )
            )

    record = {
        "image_path": image_path,
        "grid_shape": baseline["grid_shape"],
        "anchors_rc": baseline["anchors_rc"],
        "conditions": conditions,
    }
    return record


def collect_color_limits(records: list[dict[str, object]]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for record in records:
        for condition in record["conditions"]:
            values.extend(condition["sim_maps"])
    stacked = np.concatenate([arr.reshape(-1) for arr in values], axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def add_anchor_overlay_to_image(ax, image: np.ndarray, anchors_px: list[tuple[float, float]], star_size: float) -> None:
    ax.imshow(image)
    for x, y in anchors_px:
        ax.scatter(
            x,
            y,
            marker="*",
            s=star_size,
            c="red",
            edgecolors="yellow",
            linewidths=0.6,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def add_anchor_overlay_to_heatmap(ax, heatmap: np.ndarray, anchor_rc: tuple[int, int], star_size: float) -> None:
    row, col = anchor_rc
    ax.scatter(
        col,
        row,
        marker="*",
        s=star_size,
        c="red",
        edgecolors="yellow",
        linewidths=0.6,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_similarity_figure(
    records: list[dict[str, object]],
    output_path: Path,
    cmap: str,
    dpi: int,
    title_size: float,
    star_size: float,
    hide_titles: bool,
) -> None:
    if not records:
        raise ValueError("No records to plot")

    num_rows = len(records)
    num_anchors = len(records[0]["conditions"][0]["anchors_rc"])
    num_groups = len(records[0]["conditions"])
    group_cols = 1 + num_anchors
    spacer_width = 0.18
    width_ratios: list[float] = []
    for group_idx in range(num_groups):
        width_ratios.extend([1.0] * group_cols)
        if group_idx != num_groups - 1:
            width_ratios.append(spacer_width)
    width_ratios.append(0.08)
    total_cols = len(width_ratios)

    fig = plt.figure(figsize=(2.35 * total_cols, 2.55 * num_rows), constrained_layout=False)
    gs = fig.add_gridspec(num_rows, total_cols, width_ratios=width_ratios, wspace=0.05, hspace=0.08)

    cbar_ax = fig.add_subplot(gs[:, -1])

    if not hide_titles:
        width_total = float(sum(width_ratios))
        current_col = 0
        group_titles = [condition["title"] for condition in records[0]["conditions"]]
        for group_idx, group_title in enumerate(group_titles):
            group_width = sum(width_ratios[current_col : current_col + group_cols]) / width_total
            group_offset = sum(width_ratios[:current_col]) / width_total
            group_center = group_offset + group_width / 2.0
            fig.text(
                group_center,
                0.995,
                group_title,
                ha="center",
                va="top",
                fontsize=title_size,
            )
            current_col += group_cols
            if group_idx != num_groups - 1:
                current_col += 1

    vmin, vmax = collect_color_limits(records)
    last_im = None

    for row_index, record in enumerate(records):
        current_col = 0
        for group_idx, condition in enumerate(record["conditions"]):
            image = condition["display_image"]
            anchors_px = condition["anchors_px"]
            anchors_rc = condition["anchors_rc"]
            sim_maps = condition["sim_maps"]

            ax = fig.add_subplot(gs[row_index, current_col])
            add_anchor_overlay_to_image(ax, image, anchors_px, star_size)

            for anchor_idx, heatmap in enumerate(sim_maps, start=1):
                ax = fig.add_subplot(gs[row_index, current_col + anchor_idx])
                last_im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
                add_anchor_overlay_to_heatmap(ax, heatmap, anchors_rc[anchor_idx - 1], star_size)

            current_col += group_cols
            if group_idx != num_groups - 1:
                spacer_ax = fig.add_subplot(gs[row_index, current_col])
                spacer_ax.axis("off")
                current_col += 1

    if last_im is None:
        raise ValueError("Failed to create similarity maps")

    colorbar = fig.colorbar(last_im, cax=cbar_ax)
    colorbar.set_label("Cosine Similarity", fontsize=max(title_size - 1, 10))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DINOv2 patch-token cosine similarity before/after spatial normalization."
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face repo id or local checkpoint path for DINOv2.",
    )
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        required=True,
        help="Input image path or HTTP/HTTPS URL. Repeat this flag to visualize multiple images.",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        action="append",
        default=[],
        help=(
            "Semicolon-separated normalized x,y anchor pairs, e.g. "
            "'0.28,0.30;0.68,0.33;0.32,0.66;0.77,0.71'. "
            "Provide once to reuse across all images, or once per image."
        ),
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Spatial normalization strength.")
    parser.add_argument(
        "--compare-gaussian-blur",
        action="store_true",
        help="Add a third comparison group with Gaussian blur applied to the input image before DINOv2.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=4.0,
        help="Gaussian blur radius used when --compare-gaussian-blur is enabled.",
    )
    parser.add_argument(
        "--compare-wavelet",
        action="store_true",
        help="Add comparison groups for wavelet low-frequency and high-frequency filtering.",
    )
    parser.add_argument(
        "--wavelet-name",
        type=str,
        default="haar",
        help="Wavelet name for --compare-wavelet (e.g., 'haar', 'db1', 'sym2').",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Square input resolution after crop/resize.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dinov2_spatial_norm/dinov2_spatial_norm.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--title-size", type=float, default=17.0)
    parser.add_argument("--star-size", type=float, default=95.0)
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help="Retry URL downloads without SSL certificate verification if the remote host is misconfigured.",
    )
    parser.add_argument("--hide-titles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    image_paths = materialize_image_specs(
        args.image,
        output_path.parent / "_downloads",
        allow_insecure_download=args.allow_insecure_download,
    )
    for image_path in image_paths:
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

    anchor_sets = resolve_anchor_specs(args.anchors, len(image_paths))
    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")

    model_bundle = load_local_dinov2_model(args.model_root, device)
    image_size = infer_image_size(model_bundle.processor, model_bundle.config, args.image_size)
    mean, std = infer_mean_std(model_bundle.processor)
    print(f"[info] Image size: {image_size} | mean: {mean} | std: {std}")

    records = []
    for image_path, anchors in zip(image_paths, anchor_sets):
        print(f"[info] Processing {image_path}")
        records.append(
            build_visualization_record(
                model_bundle=model_bundle,
                image_path=image_path,
                anchor_points_xy=anchors,
                image_size=image_size,
                mean=mean,
                std=std,
                gamma=args.gamma,
                device=device,
                compare_gaussian_blur=args.compare_gaussian_blur,
                blur_radius=args.blur_radius,
                compare_wavelet=args.compare_wavelet,
                wavelet_name=args.wavelet_name,
            )
        )

    plot_similarity_figure(
        records=records,
        output_path=output_path,
        cmap=args.cmap,
        dpi=args.dpi,
        title_size=args.title_size,
        star_size=args.star_size,
        hide_titles=args.hide_titles,
    )
    print(f"[done] Saved figure to {output_path}")


if __name__ == "__main__":
    main()
