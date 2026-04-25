from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

try:
    from kaggle_sit_protocol.assets import load_synset_to_imagenet_idx
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.manifest import discover_dataset_root
    from kaggle_sit_protocol.modeling import load_sit_model, load_vae, preprocess_pil_image, token_grid_size
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from kaggle_sit_protocol.assets import load_synset_to_imagenet_idx
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.manifest import discover_dataset_root
    from kaggle_sit_protocol.modeling import load_sit_model, load_vae, preprocess_pil_image, token_grid_size


@dataclass(frozen=True)
class ImageRow:
    image_id: str
    path: Path
    synset: str
    imagenet_idx: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode miniImageNet images with the SD VAE, pass clean latents through "
            "SiT patch embed and every transformer layer, then render sra_local PCA-RGB panels."
        )
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="miniImageNet root. Defaults to auto-discovery under /kaggle/input.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Local SiT-XL/2 checkpoint path, e.g. /kaggle/input/.../SiT-XL-2-256.pt.")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="Diffusers VAE id or local VAE directory.")
    parser.add_argument("--sit-root", type=str, default="SiT", help="Path to the vendored SiT repo.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Defaults to outputs/kaggle_protocol/analysis/all_layer_sra_local_pca.")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to process.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when shuffling images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset images before taking --num-images.")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size for SiT-XL/2 256 checkpoints.")
    parser.add_argument("--timestep", type=float, default=1.0, help="Conditioning timestep passed to SiT blocks for the clean latent.")
    parser.add_argument("--sample-latent", action="store_true", help="Use posterior.sample(); default uses posterior.mode() for deterministic VAE latents.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model/VAE dtype. auto uses float32 for numerically stable PCA extraction.",
    )
    parser.add_argument("--save-tokens", action="store_true", help="Also save raw latent, patchify0, and layer token tensors as .pt files.")
    parser.add_argument("--panel-dpi", type=int, default=160)
    return parser.parse_args()


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float32
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _iter_image_files(synset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in synset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _select_images(config: ProtocolConfig, *, num_images: int, seed: int, shuffle: bool) -> list[ImageRow]:
    discovery = discover_dataset_root(config)
    synset_to_idx = load_synset_to_imagenet_idx()
    rows: list[ImageRow] = []
    for synset_dir in discovery.synset_dirs:
        synset = synset_dir.name
        if synset not in synset_to_idx:
            continue
        for image_path in _iter_image_files(synset_dir):
            rows.append(
                ImageRow(
                    image_id=f"{synset}_{image_path.stem}",
                    path=image_path,
                    synset=synset,
                    imagenet_idx=int(synset_to_idx[synset]),
                )
            )
    if not rows:
        raise FileNotFoundError(f"No miniImageNet image files found under {discovery.root}")
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]
    return rows[:num_images]


def _local_pca_rgb(tokens: torch.Tensor, grid_h: int, grid_w: int, *, stage_name: str) -> np.ndarray:
    matrix = tokens.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if matrix.ndim != 2:
        raise ValueError(f"Expected [tokens, dim], got {matrix.shape}")
    if matrix.shape[0] != grid_h * grid_w:
        raise ValueError(f"Grid {grid_h}x{grid_w} does not match {matrix.shape[0]} tokens")
    if min(matrix.shape) < 3:
        raise ValueError(f"Need at least 3 rows/features for PCA, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        nonfinite_count = int((~np.isfinite(matrix)).sum())
        raise ValueError(
            f"{stage_name} contains {nonfinite_count} non-finite PCA inputs. "
            "Rerun with --dtype float32, or leave --dtype unset so auto uses float32."
        )

    components = PCA(n_components=3).fit_transform(matrix).astype(np.float32, copy=True)
    for channel_index in range(3):
        channel = components[:, channel_index]
        low = float(channel.min())
        high = float(channel.max())
        if high - low <= 1e-6:
            channel.fill(0.0)
        else:
            channel -= low
            channel /= high - low
    return components.reshape(grid_h, grid_w, 3)


def _save_rgb(rgb: np.ndarray, path: Path, *, scale: int) -> None:
    array = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
    image.save(path)


@torch.inference_mode()
def _collect_tokens(
    *,
    model: torch.nn.Module,
    vae: torch.nn.Module,
    image_tensor: torch.Tensor,
    label: int,
    timestep: float,
    device: torch.device,
    dtype: torch.dtype,
    sample_latent: bool,
) -> dict[str, torch.Tensor]:
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        batch = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        posterior = vae.encode(batch).latent_dist
        latent = posterior.sample() if sample_latent else posterior.mode()
        latent = latent.mul(0.18215)

        latent_cpu = latent.squeeze(0).detach().float().cpu()
        layer0 = model.x_embedder(latent) + model.pos_embed
        stages: dict[str, torch.Tensor] = {
            "vae32": latent_cpu.permute(1, 2, 0).reshape(-1, latent_cpu.shape[0]),
            "layer0_patch_embed": layer0.squeeze(0).detach().float().cpu(),
        }

        t = torch.full((1,), float(timestep), device=device, dtype=dtype)
        y = torch.tensor([int(label)], device=device, dtype=torch.long)
        c = model.t_embedder(t) + model.y_embedder(y, train=False)

        for layer_index, block in enumerate(model.blocks, start=1):
            layer_output = block(layer0, c)
            stages[f"layer{layer_index:02d}"] = layer_output.squeeze(0).detach().float().cpu()

    return stages


def _render_contact_sheet(
    *,
    source_image: Image.Image,
    pca_images: dict[str, np.ndarray],
    image_id: str,
    output_path: Path,
    dpi: int,
) -> None:
    names = list(pca_images)
    columns = 6
    panel_count = 1 + len(names)
    rows = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.15), dpi=dpi)
    axes_arr = np.asarray(axes, dtype=object).reshape(rows, columns)

    for ax in axes_arr.ravel():
        ax.axis("off")

    axes_arr[0, 0].imshow(source_image.resize((256, 256)))
    axes_arr[0, 0].set_title("input", fontsize=8)

    for index, name in enumerate(names, start=1):
        ax = axes_arr[index // columns, index % columns]
        ax.imshow(pca_images[name])
        ax.set_title(name, fontsize=8)

    fig.suptitle(f"{image_id} | VAE latent -> patch embed -> SiT layers | sra_local PCA-RGB", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype}")

    config = ProtocolConfig.from_kaggle_defaults()
    config.image_size = int(args.image_size)
    config.latent_size = int(args.image_size) // 8
    config.patch_grid_size = config.latent_size // 2
    config.checkpoint_path = args.checkpoint_path
    config.vae_model = args.vae_model
    config.sit_root = args.sit_root
    if args.dataset_root:
        config.explicit_dataset_root = args.dataset_root
    config.ensure_directories()

    output_dir = Path(args.output_dir) if args.output_dir else config.analysis_dir / "all_layer_sra_local_pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_images(config, num_images=int(args.num_images), seed=int(args.seed), shuffle=bool(args.shuffle))

    model = load_sit_model(config, device).eval().to(device=device, dtype=dtype)
    vae = load_vae(config, device).eval().to(device=device, dtype=dtype)

    metadata: list[dict[str, object]] = []
    for row in tqdm(rows, desc="all-layer sra_local PCA"):
        image_dir = output_dir / row.image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(row.path) as image:
            source = image.convert("RGB")
            image_tensor = preprocess_pil_image(source, image_size=config.image_size)

        stages = _collect_tokens(
            model=model,
            vae=vae,
            image_tensor=image_tensor,
            label=row.imagenet_idx,
            timestep=float(args.timestep),
            device=device,
            dtype=dtype,
            sample_latent=bool(args.sample_latent),
        )

        pca_images: dict[str, np.ndarray] = {}
        for stage_name, tokens in stages.items():
            if stage_name == "vae32":
                grid_h = grid_w = config.latent_size
                scale = 8
            else:
                grid_h, grid_w = token_grid_size(int(tokens.shape[0]))
                scale = 16
            rgb = _local_pca_rgb(tokens, grid_h, grid_w, stage_name=stage_name)
            pca_images[stage_name] = rgb
            _save_rgb(rgb, image_dir / f"{stage_name}_sra_local_pca.png", scale=scale)

        _render_contact_sheet(
            source_image=source,
            pca_images=pca_images,
            image_id=row.image_id,
            output_path=image_dir / "contact_sheet_sra_local_pca.png",
            dpi=int(args.panel_dpi),
        )

        if args.save_tokens:
            torch.save(stages, image_dir / "tokens.pt")

        metadata.append(
            {
                "image_id": row.image_id,
                "path": str(row.path),
                "synset": row.synset,
                "imagenet_idx": row.imagenet_idx,
                "timestep": float(args.timestep),
                "sample_latent": bool(args.sample_latent),
                "num_stages": len(stages),
                "contact_sheet": str(image_dir / "contact_sheet_sra_local_pca.png"),
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {len(rows)} image folders to {output_dir}")


if __name__ == "__main__":
    main()
