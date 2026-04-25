from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm.auto import tqdm

try:
    from kaggle_sit_protocol.assets import load_synset_to_imagenet_idx
    from kaggle_sit_protocol.all_layer_sra_local_pca import (
        _local_pca_rgb,
        _render_contact_sheet,
        _resolve_dtype,
        _save_rgb,
        _select_images,
    )
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.decomposition import spatial_normalize_tokens
    from kaggle_sit_protocol.modeling import load_sit_model, load_vae, preprocess_pil_image, token_grid_size
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from kaggle_sit_protocol.assets import load_synset_to_imagenet_idx
    from kaggle_sit_protocol.all_layer_sra_local_pca import (
        _local_pca_rgb,
        _render_contact_sheet,
        _resolve_dtype,
        _save_rgb,
        _select_images,
    )
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.decomposition import spatial_normalize_tokens
    from kaggle_sit_protocol.modeling import load_sit_model, load_vae, preprocess_pil_image, token_grid_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick one miniImageNet image, mix its VAE latent with Gaussian noise at "
            "several noise fractions, and render sra_local PCA-RGB for patch embed "
            "plus each independent SiT block."
        )
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="miniImageNet root.")
    parser.add_argument("--image-path", type=str, default=None, help="Optional direct image path. If set, dataset selection is skipped.")
    parser.add_argument("--imagenet-idx", type=int, default=None, help="Required with --image-path unless --label is not important.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Local SiT-XL/2 checkpoint path.")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema", help="Diffusers VAE id or local VAE directory.")
    parser.add_argument("--sit-root", type=str, default="SiT", help="Path to the vendored SiT repo.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--pdf-path", type=str, default=None, help="Optional path for the combined PDF.")
    parser.add_argument("--image-index", type=int, default=0, help="Image index after optional shuffle when selecting from dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset images before selecting --image-index.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for image shuffle and Gaussian latent noise.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated noise fractions. 0.25 means 75 percent clean latent plus 25 percent noise.",
    )
    parser.add_argument("--sample-latent", action="store_true", help="Use posterior.sample(); default uses posterior.mode().")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model/VAE dtype. auto uses float32 for numerically stable PCA extraction.",
    )
    parser.add_argument("--save-tokens", action="store_true", help="Save per-noise-level token tensors.")
    parser.add_argument("--save-latents", action="store_true", help="Save clean/noise/mixed latent tensors.")
    parser.add_argument("--panel-dpi", type=int, default=160)
    return parser.parse_args()


def _parse_noise_levels(value: str) -> list[float]:
    levels = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not levels:
        raise ValueError("--noise-levels must contain at least one value")
    for level in levels:
        if level < 0.0 or level > 1.0:
            raise ValueError(f"Noise level must be in [0, 1], got {level}")
    return levels


def _image_id_from_path(path: Path) -> str:
    parent = path.parent.name
    return f"{parent}_{path.stem}" if parent else path.stem


def _select_one_image(config: ProtocolConfig, args: argparse.Namespace):
    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        if args.imagenet_idx is not None:
            label = int(args.imagenet_idx)
        else:
            label = int(load_synset_to_imagenet_idx().get(image_path.parent.name, 0))
        return {
            "image_id": _image_id_from_path(image_path),
            "path": image_path,
            "synset": image_path.parent.name,
            "imagenet_idx": label,
        }

    rows = _select_images(
        config,
        num_images=int(args.image_index) + 1,
        seed=int(args.seed),
        shuffle=bool(args.shuffle),
    )
    row = rows[int(args.image_index)]
    return {
        "image_id": row.image_id,
        "path": row.path,
        "synset": row.synset,
        "imagenet_idx": row.imagenet_idx,
    }


@torch.inference_mode()
def _encode_clean_latent(
    *,
    vae: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sample_latent: bool,
) -> torch.Tensor:
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        batch = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        posterior = vae.encode(batch).latent_dist
        latent = posterior.sample() if sample_latent else posterior.mode()
        return latent.mul(0.18215)


@torch.inference_mode()
def _collect_tokens_from_latent(
    *,
    model: torch.nn.Module,
    latent: torch.Tensor,
    label: int,
    timestep: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        latent = latent.to(device=device, dtype=dtype)
        latent_cpu = latent.squeeze(0).detach().float().cpu()
        layer0 = model.x_embedder(latent) + model.pos_embed
        stages: dict[str, torch.Tensor] = {
            "latent32_xt": latent_cpu.permute(1, 2, 0).reshape(-1, latent_cpu.shape[0]),
            "layer0_patch_embed": layer0.squeeze(0).detach().float().cpu(),
        }

        t = torch.full((1,), float(timestep), device=device, dtype=dtype)
        y = torch.tensor([int(label)], device=device, dtype=torch.long)
        c = model.t_embedder(t) + model.y_embedder(y, train=False)

        for layer_index, block in enumerate(model.blocks, start=1):
            layer_output = block(layer0, c)
            stages[f"layer{layer_index:02d}"] = layer_output.squeeze(0).detach().float().cpu()
    return stages


def _render_stage_set(
    *,
    stages: dict[str, torch.Tensor],
    source: Image.Image,
    image_id: str,
    level_dir: Path,
    variant_name: str,
    config: ProtocolConfig,
    pdf: PdfPages,
    panel_dpi: int,
    apply_spatial_norm: bool,
) -> str:
    variant_dir = level_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    pca_images: dict[str, np.ndarray] = {}
    for stage_name, tokens in stages.items():
        if stage_name == "latent32_xt":
            grid_h = grid_w = config.latent_size
            scale = 8
        else:
            grid_h, grid_w = token_grid_size(int(tokens.shape[0]))
            scale = 16
        stage_tokens = (
            spatial_normalize_tokens(
                tokens,
                gamma=config.spatial_norm_gamma,
                eps=config.spatial_norm_eps,
            )
            if apply_spatial_norm
            else tokens
        )
        rgb = _local_pca_rgb(stage_tokens, grid_h, grid_w, stage_name=f"{image_id}/{variant_name}/{stage_name}")
        pca_images[stage_name] = rgb
        _save_rgb(rgb, variant_dir / f"{stage_name}_sra_local_pca.png", scale=scale)

    contact_sheet = variant_dir / "contact_sheet_sra_local_pca.png"
    _render_contact_sheet(
        source_image=source,
        pca_images=pca_images,
        image_id=f"{image_id} | {variant_name}",
        output_path=contact_sheet,
        pdf=pdf,
        dpi=panel_dpi,
    )
    return str(contact_sheet)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    noise_levels = _parse_noise_levels(args.noise_levels)
    print(f"Using device={device}, dtype={dtype}, noise_levels={noise_levels}")

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

    output_dir = Path(args.output_dir) if args.output_dir else config.analysis_dir / "one_image_noise_sra_local_pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(args.pdf_path) if args.pdf_path else output_dir / "one_image_noise_sra_local_pca.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    row = _select_one_image(config, args)
    image_path = Path(row["path"])
    with Image.open(image_path) as image:
        source = image.convert("RGB")
        image_tensor = preprocess_pil_image(source, image_size=config.image_size)

    model = load_sit_model(config, device).eval().to(device=device, dtype=dtype)
    vae = load_vae(config, device).eval().to(device=device, dtype=dtype)

    clean_latent = _encode_clean_latent(
        vae=vae,
        image_tensor=image_tensor,
        device=device,
        dtype=dtype,
        sample_latent=bool(args.sample_latent),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    noise_latent = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)

    metadata: list[dict[str, object]] = []
    pdf = PdfPages(pdf_path)
    for noise_level in tqdm(noise_levels, desc="one-image noise sra_local PCA"):
        timestep = 1.0 - float(noise_level)
        mixed_latent = timestep * clean_latent + float(noise_level) * noise_latent
        level_name = f"noise_{noise_level:.2f}".replace(".", "p")
        image_id = f"{row['image_id']}_{level_name}_t_{timestep:.2f}".replace(".", "p")
        level_dir = output_dir / image_id
        level_dir.mkdir(parents=True, exist_ok=True)

        stages = _collect_tokens_from_latent(
            model=model,
            latent=mixed_latent,
            label=int(row["imagenet_idx"]),
            timestep=timestep,
            device=device,
            dtype=dtype,
        )

        sheet_title = f"{row['image_id']} | noise={noise_level:.2f} | t={timestep:.2f}"
        raw_contact_sheet = _render_stage_set(
            stages=stages,
            source=source,
            image_id=sheet_title,
            level_dir=level_dir,
            variant_name="raw",
            config=config,
            pdf=pdf,
            panel_dpi=int(args.panel_dpi),
            apply_spatial_norm=False,
        )
        spatial_norm_contact_sheet = _render_stage_set(
            stages=stages,
            source=source,
            image_id=sheet_title,
            level_dir=level_dir,
            variant_name="spatial_norm",
            config=config,
            pdf=pdf,
            panel_dpi=int(args.panel_dpi),
            apply_spatial_norm=True,
        )

        if args.save_tokens:
            torch.save(stages, level_dir / "tokens.pt")
        if args.save_latents:
            torch.save(
                {
                    "clean_latent": clean_latent.detach().float().cpu(),
                    "noise_latent": noise_latent.detach().float().cpu(),
                    "mixed_latent": mixed_latent.detach().float().cpu(),
                    "noise_level": float(noise_level),
                    "timestep": float(timestep),
                },
                level_dir / "latents.pt",
            )

        metadata.append(
            {
                "image_id": str(row["image_id"]),
                "path": str(image_path),
                "synset": str(row["synset"]),
                "imagenet_idx": int(row["imagenet_idx"]),
                "noise_level": float(noise_level),
                "timestep": float(timestep),
                "formula": "x_t = (1 - noise_level) * clean_latent + noise_level * gaussian_noise",
                "level_dir": str(level_dir),
                "raw_contact_sheet": raw_contact_sheet,
                "spatial_norm_contact_sheet": spatial_norm_contact_sheet,
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
    pdf.close()

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote outputs to {output_dir}")
    print(f"Wrote combined PDF to {pdf_path}")


if __name__ == "__main__":
    main()
