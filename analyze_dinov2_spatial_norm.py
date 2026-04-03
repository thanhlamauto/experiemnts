#!/usr/bin/env python3
"""Visualize DINOv2 patch-token similarity before and after spatial normalization.

This script is intended for Kaggle usage with a local Hugging Face-style Flax checkpoint,
for example:

  /kaggle/input/datasets/bangchi/dinov2-vitb14-flax

It loads the local DINOv2 checkpoint with `transformers`, converts Flax weights to a
PyTorch `Dinov2Model` when needed, extracts patch tokens, and reproduces the paper-style
visualization of cosine-similarity maps with and without spatial normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def load_local_dinov2_model(model_root: Path, device: torch.device):
    try:
        from transformers import AutoImageProcessor, Dinov2Model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required. On Kaggle run: pip install -q transformers"
        ) from exc

    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(str(model_root), local_files_only=True)
    except Exception as exc:
        print(f"[warn] Failed to load AutoImageProcessor from {model_root}: {exc}")

    last_error: Exception | None = None
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
            return model, processor
        except Exception as exc:
            last_error = exc

    raise SystemExit(f"Failed to load DINOv2 checkpoint from {model_root}: {last_error}")


def is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def download_image_url(image_url: str, download_dir: Path) -> Path:
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
        urlretrieve(image_url, local_path)
    except Exception as exc:
        raise SystemExit(f"Failed to download image from {image_url}: {exc}") from exc
    return local_path


def materialize_image_specs(image_specs: list[str], download_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for image_spec in image_specs:
        if is_http_url(image_spec):
            image_paths.append(download_image_url(image_spec, download_dir))
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


def load_and_prepare_image(
    image_path: Path,
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    image = center_crop_square(image).resize((image_size, image_size), resample=Image.BICUBIC)

    display = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(display).permute(2, 0, 1).contiguous()

    mean_t = torch.tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
    pixel_values = ((tensor - mean_t) / std_t).unsqueeze(0).to(device)
    return display, pixel_values


@torch.no_grad()
def extract_patch_tokens(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError(f"Unexpected DINOv2 hidden state shape: {tuple(hidden.shape)}")
    return hidden[:, 1:, :]


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


def build_visualization_record(
    model: torch.nn.Module,
    image_path: Path,
    anchor_points_xy: list[tuple[float, float]],
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
    gamma: float,
    device: torch.device,
) -> dict[str, object]:
    display_image, pixel_values = load_and_prepare_image(image_path, image_size, mean, std, device)
    tokens = extract_patch_tokens(model, pixel_values)[0].detach().cpu()
    grid_height, grid_width = grid_size_from_tokens(tokens.shape[0])
    anchors_rc = [normalized_anchor_to_grid(anchor, grid_width, grid_height) for anchor in anchor_points_xy]

    record = {
        "image_path": image_path,
        "display_image": display_image,
        "grid_shape": (grid_height, grid_width),
        "anchors_rc": anchors_rc,
        "anchors_px": [
            anchor_center_in_pixels(anchor_rc, image_size, grid_width, grid_height)
            for anchor_rc in anchors_rc
        ],
        "sim_wo": compute_similarity_maps(tokens, anchors_rc, grid_width, grid_height),
        "sim_w": compute_similarity_maps(
            spatial_normalize_tokens(tokens.unsqueeze(0), gamma=gamma)[0],
            anchors_rc,
            grid_width,
            grid_height,
        ),
    }
    return record


def collect_color_limits(records: list[dict[str, object]]) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for record in records:
        values.extend(record["sim_wo"])
        values.extend(record["sim_w"])
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
    num_anchors = len(records[0]["anchors_rc"])
    total_cols = 2 * (1 + num_anchors) + 2
    width_ratios = [1.0] * (1 + num_anchors) + [0.18] + [1.0] * (1 + num_anchors) + [0.08]

    fig = plt.figure(figsize=(2.35 * total_cols, 2.55 * num_rows), constrained_layout=False)
    gs = fig.add_gridspec(num_rows, total_cols, width_ratios=width_ratios, wspace=0.05, hspace=0.08)

    left_group_cols = 1 + num_anchors
    right_group_start = left_group_cols + 1
    cbar_ax = fig.add_subplot(gs[:, -1])

    if not hide_titles:
        width_total = float(sum(width_ratios))
        left_center = sum(width_ratios[:left_group_cols]) / width_total / 2.0
        right_offset = sum(width_ratios[:right_group_start]) / width_total
        right_width = sum(width_ratios[right_group_start : right_group_start + left_group_cols]) / width_total
        right_center = right_offset + right_width / 2.0
        fig.text(
            left_center,
            0.995,
            "w/o Spatial Normalization Layer",
            ha="center",
            va="top",
            fontsize=title_size,
        )
        fig.text(
            right_center,
            0.995,
            "w/ Spatial Normalization Layer",
            ha="center",
            va="top",
            fontsize=title_size,
        )

    vmin, vmax = collect_color_limits(records)
    last_im = None

    for row_index, record in enumerate(records):
        image = record["display_image"]
        anchors_px = record["anchors_px"]
        anchors_rc = record["anchors_rc"]
        sim_wo = record["sim_wo"]
        sim_w = record["sim_w"]

        ax = fig.add_subplot(gs[row_index, 0])
        add_anchor_overlay_to_image(ax, image, anchors_px, star_size)

        for anchor_idx, heatmap in enumerate(sim_wo, start=1):
            ax = fig.add_subplot(gs[row_index, anchor_idx])
            last_im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
            add_anchor_overlay_to_heatmap(ax, heatmap, anchors_rc[anchor_idx - 1], star_size)

        spacer_ax = fig.add_subplot(gs[row_index, left_group_cols])
        spacer_ax.axis("off")

        ax = fig.add_subplot(gs[row_index, right_group_start])
        add_anchor_overlay_to_image(ax, image, anchors_px, star_size)

        for anchor_idx, heatmap in enumerate(sim_w, start=1):
            ax = fig.add_subplot(gs[row_index, right_group_start + anchor_idx])
            last_im = ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax)
            add_anchor_overlay_to_heatmap(ax, heatmap, anchors_rc[anchor_idx - 1], star_size)

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
        default="/kaggle/input/datasets/bangchi/dinov2-vitb14-flax",
        help="Local Hugging Face checkpoint directory for DINOv2 ViT-B/14 Flax weights.",
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
    parser.add_argument("--hide-titles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_root = Path(args.model_root)
    output_path = Path(args.output)
    image_paths = materialize_image_specs(args.image, output_path.parent / "_downloads")
    for image_path in image_paths:
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")

    anchor_sets = resolve_anchor_specs(args.anchors, len(image_paths))
    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")

    model, processor = load_local_dinov2_model(model_root, device)
    image_size = infer_image_size(processor, model.config, args.image_size)
    mean, std = infer_mean_std(processor)
    print(f"[info] Image size: {image_size} | mean: {mean} | std: {std}")

    records = []
    for image_path, anchors in zip(image_paths, anchor_sets):
        print(f"[info] Processing {image_path}")
        records.append(
            build_visualization_record(
                model=model,
                image_path=image_path,
                anchor_points_xy=anchors,
                image_size=image_size,
                mean=mean,
                std=std,
                gamma=args.gamma,
                device=device,
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
