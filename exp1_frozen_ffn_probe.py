#!/usr/bin/env python3
"""
Experiment 1: Frozen FFN Probing
=================================
Dùng final_layer (frozen weight từ ckpt) để decode hidden state tại từng block
của SiT XL/2 (vanilla) và REPA SiT XL/2, rồi decode qua VAE ra ảnh dự đoán.

Mục đích: khảo sát xem layer nào chứa nhiều thông tin dành cho denoise nhất.

Colab usage:
    # 0. Clone repos (chạy 1 lần)
    !git clone https://github.com/willisma/SiT /content/SiT
    !git clone https://github.com/sihyun-yu/REPA /content/REPA
    !pip install diffusers timm accelerate datasets Pillow tqdm -q

    # 1. Chạy với random latents (không cần data)
    !python exp1_frozen_ffn_probe.py

    # 2. Hoặc với mini-imagenet thật
    !python exp1_frozen_ffn_probe.py --data-dir /path/to/imagenet/val --batch-size 256

Local usage (có GPU):
    python exp1_frozen_ffn_probe.py --sit-root ./SiT --repa-root ./REPA \\
        --timesteps 0.1,0.3,0.5,0.7,0.9 --batch-size 64 --outdir ./outputs/exp1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# Directory of this script — used as base for all relative paths
_SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    """Download a file if it doesn't exist yet."""
    if os.path.isfile(dest):
        print(f"[skip] already exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[download] {url}\n        -> {dest}")
    try:
        from torchvision.datasets.utils import download_url as tv_dl
        tv_dl(url, os.path.dirname(dest), filename=os.path.basename(dest))
    except Exception as e:
        print(f"[warn] torchvision download failed ({e}), trying requests...")
        import requests, shutil
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def download_sit_ckpt(dest: str = "pretrained_models/SiT-XL-2-256x256.pt") -> str:
    url = (
        "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/"
        "SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"
    )
    _download(url, dest)
    return dest


def download_repa_ckpt(dest: str = "pretrained_models/repa-last.pt") -> str:
    url = (
        "https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/"
        "last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0"
    )
    _download(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Model loading (adapted from sit_metrics/model_loader.py)
# ---------------------------------------------------------------------------

def _load_module(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sit_model(ckpt_path: str, sit_root: str, device: torch.device) -> nn.Module:
    root = Path(sit_root)
    dl_mod = _load_module("sit_dl", root / "download.py")
    sit_mod = _load_module("sit_models", root / "models.py")

    model = sit_mod.SiT_models["SiT-XL/2"](
        input_size=32, num_classes=1000, learn_sigma=True
    ).to(device)

    ckpt = dl_mod.find_model(ckpt_path)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"[sit]  loaded from {ckpt_path}")
    return model


def load_repa_model(ckpt_path: str, repa_root: str, device: torch.device) -> nn.Module:
    root = Path(repa_root)
    repa_mod = _load_module("repa_sit", root / "models" / "sit.py")

    model = repa_mod.SiT_models["SiT-XL/2"](
        input_size=32,
        num_classes=1000,
        use_cfg=True,
        z_dims=[768],
        encoder_depth=8,
        fused_attn=False,
        qk_norm=False,
    ).to(device)

    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("ema", raw.get("model", raw))
    # strip module. prefix
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[repa] loaded from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class RandomLatentDataset(Dataset):
    """Synthetic dataset: random latents + random labels for quick demo."""
    def __init__(self, n: int = 256, latent_size: int = 32, channels: int = 4, seed: int = 0):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.x = torch.randn(n, channels, latent_size, latent_size, generator=rng)
        self.y = torch.randint(0, 1000, (n,), generator=rng)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_imagenet_loader(data_dir: str, batch_size: int, n_workers: int, n_samples: int = 256):
    """Load real ImageNet val images and encode to latents on-the-fly."""
    from torchvision.datasets import ImageFolder
    tfm = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImageFolder(data_dir, transform=tfm)
    # subsample
    indices = torch.randperm(len(ds))[:n_samples].tolist()
    sub = torch.utils.data.Subset(ds, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)


@torch.no_grad()
def encode_batch(vae: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode images (B,3,H,W) to latents (B,4,H/8,W/8)."""
    posterior = vae.encode(images).latent_dist
    z = posterior.sample() * 0.18215
    return z


# ---------------------------------------------------------------------------
# Hook-based hidden state extraction
# ---------------------------------------------------------------------------

def get_hidden_states_sit(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run SiT forward, collect hidden state after every block. Also return c."""
    hidden_states: List[Optional[torch.Tensor]] = [None] * len(model.blocks)

    def make_hook(idx):
        def _hook(_mod, _inp, out):
            hidden_states[idx] = out.detach()
        return _hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.blocks)]
    with torch.no_grad():
        x_emb = model.x_embedder(x) + model.pos_embed
        t_emb = model.t_embedder(t)
        y_emb = model.y_embedder(y, False)
        c = t_emb + y_emb
        h = x_emb
        for blk in model.blocks:
            h = blk(h, c)
    for hh in handles:
        hh.remove()
    return hidden_states, c


def get_hidden_states_repa(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run REPA forward, collect hidden state after every block. Also return c."""
    hidden_states: List[Optional[torch.Tensor]] = [None] * len(model.blocks)

    def make_hook(idx):
        def _hook(_mod, _inp, out):
            hidden_states[idx] = out.detach()
        return _hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.blocks)]
    with torch.no_grad():
        x_emb = model.x_embedder(x) + model.pos_embed
        t_emb = model.t_embedder(t)
        y_emb = model.y_embedder(y, False)
        c = t_emb + y_emb
        h = x_emb
        for blk in model.blocks:
            h = blk(h, c)
    for hh in handles:
        hh.remove()
    return hidden_states, c


# ---------------------------------------------------------------------------
# Probe one batch: compute per-layer MSE loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_batch_frozen_ffn(
    model: nn.Module,
    backend: str,           # "sit" or "repa"
    x_latent: torch.Tensor, # (B, 4, 32, 32) — noisy latents
    t: torch.Tensor,        # (B,) timestep in [0,1]
    y: torch.Tensor,        # (B,) class labels
) -> Dict[int, float]:
    """
    For each block ℓ:
      H_ell → final_layer (frozen) → unpatchify → take noise channels
    Returns MSE(pred_noise, noisy_input) per layer.
    (proxy: how well this hidden state can reconstruct the noisy signal)
    """
    if backend == "sit":
        hidden_states, c = get_hidden_states_sit(model, x_latent, t, y)
    else:
        hidden_states, c = get_hidden_states_repa(model, x_latent, t, y)

    losses: Dict[int, float] = {}
    n_blocks = len(model.blocks)

    for ell in range(n_blocks):
        H = hidden_states[ell]  # (B, T, D)

        # Pass through the frozen final_layer from ckpt
        pred_patches = model.final_layer(H, c)  # (B, T, p*p*C_out)

        # Unpatchify
        if backend == "sit":
            pred_img = model.unpatchify(pred_patches)  # (B, 8, 32, 32)   (learn_sigma=True → 8ch)
            pred_noise = pred_img[:, :4]               # take first 4 channels
        else:
            pred_img = model.unpatchify(pred_patches)  # (B, 4, 32, 32)
            pred_noise = pred_img

        # Proxy MSE: how close the prediction is to the noisy input
        mse = float(((pred_noise - x_latent) ** 2).mean().item())
        losses[ell] = mse

    return losses


# ---------------------------------------------------------------------------
# VAE decode to image grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latent_to_pil(vae: nn.Module, latent: torch.Tensor):
    """latent: (B,4,32,32) [unscaled] → PIL image grid."""
    from PIL import Image
    latent_scaled = latent / 0.18215
    imgs = vae.decode(latent_scaled).sample  # (B,3,256,256) in [-1,1]
    imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()  # (B,3,H,W)
    imgs = imgs.transpose(0, 2, 3, 1)  # (B,H,W,3)
    n = imgs.shape[0]
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    h, w = imgs.shape[1], imgs.shape[2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return Image.fromarray(canvas)


@torch.no_grad()
def visualize_layer_outputs(
    model: nn.Module,
    backend: str,
    vae: nn.Module,
    x_latent: torch.Tensor,
    t_val: float,
    y: torch.Tensor,
    layers_to_show: List[int],
    outdir: Path,
    tag: str,
) -> None:
    """Decode hidden state at selected layers → save PIL image grid."""
    B = x_latent.shape[0]
    t = torch.full((B,), t_val, device=x_latent.device, dtype=x_latent.dtype)

    if backend == "sit":
        hidden_states, c = get_hidden_states_sit(model, x_latent, t, y)
    else:
        hidden_states, c = get_hidden_states_repa(model, x_latent, t, y)

    outdir.mkdir(parents=True, exist_ok=True)
    for ell in layers_to_show:
        H = hidden_states[ell]
        pred_patches = model.final_layer(H, c)
        if backend == "sit":
            pred_img = model.unpatchify(pred_patches)
            pred_noise = pred_img[:, :4]
        else:
            pred_img = model.unpatchify(pred_patches)
            pred_noise = pred_img

        # Show first 4 images in the batch
        pil = decode_latent_to_pil(vae, pred_noise[:4].clamp(-5, 5))
        fname = outdir / f"{tag}_layer{ell:02d}_t{t_val:.1f}.png"
        pil.save(str(fname))
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 1: Frozen FFN probing of SiT hidden states"
    )
    # Paths
    parser.add_argument("--sit-root", default=None,
                        help="Path to SiT repo (default: ./SiT relative to this script)")
    parser.add_argument("--repa-root", default=None,
                        help="Path to REPA repo (default: ./REPA relative to this script)")
    parser.add_argument("--sit-ckpt", default=None,
                        help="Local path for SiT ckpt (default: <script_dir>/pretrained_models/SiT-XL-2-256x256.pt)")
    parser.add_argument("--repa-ckpt", default=None,
                        help="Local path for REPA ckpt (default: <script_dir>/pretrained_models/repa-last.pt)")
    parser.add_argument("--outdir", default=None,
                        help="Directory to save loss tables and images (default: <script_dir>/outputs/exp1)")
    # Data
    parser.add_argument("--data-dir", default=None,
                        help="ImageNet val dir (optional; uses random latents if not given)")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of images/latents to use (batch for loss)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per forward pass (accumulate over n-samples)")
    parser.add_argument("--n-workers", type=int, default=2)
    # Experiment
    parser.add_argument("--timesteps", default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated timesteps in (0,1)")
    parser.add_argument("--backends", default="sit,repa",
                        help="Which models to run: sit, repa, or sit,repa")
    parser.add_argument("--vae", default="mse", choices=["ema", "mse"])
    # Viz
    parser.add_argument("--viz-layers", default="0,6,13,20,27",
                        help="Layer indices to visualize (comma-sep)")
    parser.add_argument("--viz-timestep", type=float, default=0.5,
                        help="Timestep to use for visualization")
    parser.add_argument("--n-viz-images", type=int, default=4,
                        help="Number of images to decode for visualization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve all paths relative to the script's own directory
    sit_root  = Path(args.sit_root)  if args.sit_root  else _SCRIPT_DIR / "SiT"
    repa_root = Path(args.repa_root) if args.repa_root else _SCRIPT_DIR / "REPA"
    sit_ckpt  = args.sit_ckpt  if args.sit_ckpt  else str(_SCRIPT_DIR / "pretrained_models" / "SiT-XL-2-256x256.pt")
    repa_ckpt = args.repa_ckpt if args.repa_ckpt else str(_SCRIPT_DIR / "pretrained_models" / "repa-last.pt")
    outdir_p  = Path(args.outdir) if args.outdir else _SCRIPT_DIR / "outputs" / "exp1"

    # Override args so downstream code uses resolved paths
    args.sit_root  = str(sit_root)
    args.repa_root = str(repa_root)
    args.sit_ckpt  = sit_ckpt
    args.repa_ckpt = repa_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[paths] script_dir={_SCRIPT_DIR}")
    print(f"[paths] sit_root={args.sit_root}")
    print(f"[paths] repa_root={args.repa_root}")

    outdir = outdir_p
    outdir.mkdir(parents=True, exist_ok=True)

    timesteps = [float(t.strip()) for t in args.timesteps.split(",") if t.strip()]
    viz_layers = [int(l) for l in args.viz_layers.split(",") if l.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    # ---- Download checkpoints -------------------------------------------
    if "sit" in backends:
        if not os.path.isfile(args.sit_ckpt):
            download_sit_ckpt(args.sit_ckpt)
        else:
            print(f"[sit]  ckpt found: {args.sit_ckpt}")

    if "repa" in backends:
        if not os.path.isfile(args.repa_ckpt):
            download_repa_ckpt(args.repa_ckpt)
        else:
            print(f"[repa] ckpt found: {args.repa_ckpt}")

    # ---- Load VAE ----------------------------------------------------------
    print(f"[vae] loading stabilityai/sd-vae-ft-{args.vae}...")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ---- Build data --------------------------------------------------------
    if args.data_dir:
        print(f"[data] loading ImageNet from {args.data_dir}...")
        loader = build_imagenet_loader(args.data_dir, args.batch_size, args.n_workers, args.n_samples)
        use_real = True
    else:
        print(f"[data] no --data-dir given, using {args.n_samples} random latents")
        ds = RandomLatentDataset(n=args.n_samples, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        use_real = False

    # Collect all latents + labels upfront (they're reused across timesteps/models)
    all_latents: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    print("[data] encoding batch...")
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            if use_real:
                z = encode_batch(vae, imgs)
            else:
                z = imgs  # already latents
            all_latents.append(z.cpu())
            all_labels.append(labels.cpu())
            if sum(x.shape[0] for x in all_latents) >= args.n_samples:
                break

    all_latents = torch.cat(all_latents, dim=0)[:args.n_samples]
    all_labels = torch.cat(all_labels, dim=0)[:args.n_samples]
    print(f"[data] collected {all_latents.shape[0]} latents, shape={tuple(all_latents.shape)}")

    # ---- Load models and run probing -------------------------------------------
    results: Dict[str, Dict[float, Dict[int, float]]] = {}  # backend -> t -> layer -> mse

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*60}")

        if backend == "sit":
            model = load_sit_model(args.sit_ckpt, args.sit_root, device)
        else:
            model = load_repa_model(args.repa_ckpt, args.repa_root, device)

        n_blocks = len(model.blocks)
        print(f"  # blocks: {n_blocks}")

        results[backend] = {}

        for t_val in timesteps:
            print(f"\n  --- timestep t={t_val} ---")
            layer_losses_sum: Dict[int, float] = {i: 0.0 for i in range(n_blocks)}
            n_batches = 0

            # Accumulate over mini-batches
            for b_start in range(0, all_latents.shape[0], args.batch_size):
                x_b = all_latents[b_start:b_start + args.batch_size].to(device)
                y_b = all_labels[b_start:b_start + args.batch_size].to(device)
                B = x_b.shape[0]

                # Add flow-matching noise: x_t = (1-t)*x0 + t*eps
                t_tensor = torch.full((B,), t_val, device=device, dtype=x_b.dtype)
                eps = torch.randn_like(x_b)
                x_noisy = (1 - t_val) * x_b + t_val * eps

                batch_losses = probe_batch_frozen_ffn(model, backend, x_noisy, t_tensor, y_b)

                for ell, loss in batch_losses.items():
                    layer_losses_sum[ell] += loss
                n_batches += 1

            # Average
            layer_losses_avg = {ell: layer_losses_sum[ell] / n_batches for ell in range(n_blocks)}
            results[backend][t_val] = layer_losses_avg

            # Print table row
            for ell in range(n_blocks):
                print(f"    layer {ell:2d}: MSE = {layer_losses_avg[ell]:.4f}")

        # ---- Save numerical results -----------------------------------------
        loss_csv = outdir / f"{backend}_frozen_ffn_loss.csv"
        with open(loss_csv, "w") as f:
            header = "layer," + ",".join(f"t={t}" for t in timesteps)
            f.write(header + "\n")
            for ell in range(n_blocks):
                row = str(ell) + "," + ",".join(
                    f"{results[backend][t][ell]:.6f}" for t in timesteps
                )
                f.write(row + "\n")
        print(f"\n  [saved] loss table: {loss_csv}")

        # ---- Visualization: decode images at selected layers ----------------
        print(f"\n  [viz] decoding images at layers {viz_layers}, t={args.viz_timestep}")
        viz_latents = all_latents[:args.n_viz_images].to(device)
        viz_labels = all_labels[:args.n_viz_images].to(device)
        viz_dir = outdir / "images"
        visualize_layer_outputs(
            model, backend, vae,
            viz_latents, args.viz_timestep, viz_labels,
            viz_layers, viz_dir, tag=backend
        )

        # Also save the original (clean) decoded images for reference
        with torch.no_grad():
            orig_pil = decode_latent_to_pil(vae, viz_latents[:4])
            orig_pil.save(str(viz_dir / f"{backend}_original_clean.png"))
            print(f"  saved reference: {viz_dir / f'{backend}_original_clean.png'}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Summary print -------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY: MSE per layer per timestep")
    print(f"{'='*60}")
    for backend in backends:
        print(f"\n  === {backend.upper()} ===")
        n_blocks = len(results[backend][timesteps[0]])
        header = f"  {'layer':>6}" + "".join(f"  t={t:.1f}" for t in timesteps)
        print(header)
        for ell in range(n_blocks):
            row = f"  {ell:>6}" + "".join(
                f"  {results[backend][t][ell]:6.4f}" for t in timesteps
            )
            print(row)

    print(f"\n[done] all outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
