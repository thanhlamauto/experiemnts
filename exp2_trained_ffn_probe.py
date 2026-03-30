#!/usr/bin/env python3
"""
Experiment 2: Trained FFN Probing per Layer
=============================================
Train một FinalLayer FFN riêng cho từng block của SiT XL/2 (và REPA),
để tối ưu dự đoán noise từ hidden state tại layer đó.
Mục đích: xem layer nào học được "denoise signal" mạnh nhất khi có supervised training.

Dataset: tiny-imagenet từ HuggingFace (200 classes, ~100k ảnh, ~100MB)
Target: flow-matching velocity  v = eps - x0  (linear path: x_t = (1-t)*x0 + t*eps)

Colab usage:
    # 0. Clone repos (1 lần)
    !git clone https://github.com/willisma/SiT /content/SiT
    !git clone https://github.com/sihyun-yu/REPA /content/REPA
    !pip install diffusers timm accelerate datasets Pillow tqdm -q

    # 1. Train on tiny-imagenet (auto-downloaded), probe all 28 layers
    !python exp2_trained_ffn_probe.py

    # 2. Or with custom imagenet
    !python exp2_trained_ffn_probe.py --data-dir /path/to/imagenet/train \\
        --epochs 5 --batch-size 64 --backends sit,repa

Local usage:
    python exp2_trained_ffn_probe.py --sit-root ./SiT --repa-root ./REPA \\
        --epochs 3 --batch-size 32 --outdir ./outputs/exp2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Directory of this script — used as base for all relative paths
_SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# Re-use download helpers from exp1
# ============================================================

def _download(url: str, dest: str) -> None:
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
    _download(
        "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/"
        "SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0",
        dest,
    )
    return dest


def download_repa_ckpt(dest: str = "pretrained_models/repa-last.pt") -> str:
    _download(
        "https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/"
        "last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0",
        dest,
    )
    return dest


# ============================================================
# Model loading
# ============================================================

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
    dl_mod = _load_module("sit_dl2", root / "download.py")
    sit_mod = _load_module("sit_models2", root / "models.py")
    model = sit_mod.SiT_models["SiT-XL/2"](
        input_size=32, num_classes=1000, learn_sigma=True
    ).to(device)
    ckpt = dl_mod.find_model(ckpt_path)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[sit]  loaded & frozen from {ckpt_path}")
    return model


def load_repa_model(ckpt_path: str, repa_root: str, device: torch.device) -> nn.Module:
    root = Path(repa_root)
    repa_mod = _load_module("repa_sit2", root / "models" / "sit.py")
    model = repa_mod.SiT_models["SiT-XL/2"](
        input_size=32, num_classes=1000, use_cfg=True,
        z_dims=[768], encoder_depth=8, fused_attn=False, qk_norm=False,
    ).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("ema", raw.get("model", raw))
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[repa] loaded & frozen from {ckpt_path}")
    return model


# ============================================================
# FinalLayer probe head (same structure as SiT/REPA FinalLayer)
# ============================================================

class FinalLayerProbe(nn.Module):
    """
    Lightweight probe head — same architecture as SiT FinalLayer:
    AdaLN modulation + linear projection → patch space.
    Khởi tạo ngẫu nhiên (không copy weight từ ckpt).
    """
    def __init__(self, hidden_size: int = 1152, patch_size: int = 2, out_channels: int = 4):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), c: (B, D)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = (1 + scale.unsqueeze(1)) * self.norm_final(x) + shift.unsqueeze(1)
        return self.linear(x)


class LayerProbeBank(nn.Module):
    """28 independent FinalLayer probes, one per DiT block."""
    def __init__(self, n_layers: int = 28, hidden_size: int = 1152,
                 patch_size: int = 2, out_channels: int = 4):
        super().__init__()
        self.probes = nn.ModuleList([
            FinalLayerProbe(hidden_size, patch_size, out_channels)
            for _ in range(n_layers)
        ])

    def forward_layer(self, layer_idx: int, H: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.probes[layer_idx](H, c)


# ============================================================
# Data: tiny-imagenet via HuggingFace datasets
# ============================================================

def build_tiny_imagenet_loader(batch_size: int, n_workers: int, split: str = "train"):
    """Load tiny-imagenet (200 cls, 64x64, resize to 256). Auto-downloaded."""
    print("[data] loading tiny-imagenet from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds_hf = load_dataset("zh-plus/tiny-imagenet", split=split)

    tfm = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    class _TinyImageNetWrapper(Dataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = item["label"]
            return tfm(img), label

    wrapped = _TinyImageNetWrapper(ds_hf)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=True,
                      num_workers=n_workers, pin_memory=True, drop_last=True)


def build_imagenet_loader(data_dir: str, batch_size: int, n_workers: int):
    from torchvision.datasets import ImageFolder
    tfm = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImageFolder(data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=n_workers, pin_memory=True, drop_last=True)


@torch.no_grad()
def encode_batch(vae: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return vae.encode(images).latent_dist.sample() * 0.18215


# ============================================================
# Collect hidden states for a batch (all layers at once)
# ============================================================

@torch.no_grad()
def collect_hidden_states(
    model: nn.Module,
    backend: str,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns:
        hidden_states: list of (B, T, D) tensors, one per block
        c: (B, D) conditioning embedding
    """
    n_blocks = len(model.blocks)
    hidden_states: List[Optional[torch.Tensor]] = [None] * n_blocks

    def make_hook(idx):
        def _hook(_mod, _inp, out):
            # clone and store on CPU to save GPU VRAM during training
            hidden_states[idx] = out.detach()
        return _hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.blocks)]

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


# ============================================================
# Unpatchify (same as SiT/REPA)
# ============================================================

def unpatchify(x_patches: torch.Tensor, patch_size: int = 2, out_channels: int = 4) -> torch.Tensor:
    """x_patches: (B, T, p*p*C) → (B, C, H, W)"""
    B, T, _ = x_patches.shape
    h = w = int(math.sqrt(T))
    p = patch_size
    c = out_channels
    x = x_patches.reshape(B, h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(B, c, h * p, w * p)
    return imgs


# ============================================================
# Visualization
# ============================================================

@torch.no_grad()
def decode_and_save_grid(
    vae: nn.Module,
    latents: torch.Tensor,
    path: Path,
    title: str = "",
) -> None:
    from PIL import Image, ImageDraw
    latents_scaled = latents.clamp(-5, 5) / 0.18215
    imgs = vae.decode(latents_scaled).sample  # (B,3,H,W) in [-1,1]
    imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
    B, H, W, _ = imgs.shape
    cols = min(B, 4)
    rows = math.ceil(B / cols)
    canvas = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = img
    pil = Image.fromarray(canvas)
    if title:
        draw = ImageDraw.Draw(pil)
        draw.text((4, 4), title, fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(path))


@torch.no_grad()
def visualize_all_layers_all_timesteps_probe(
    model: nn.Module,
    backend: str,
    probe_bank: LayerProbeBank,
    vae: nn.Module,
    x0_latents: torch.Tensor,
    all_labels: torch.Tensor,
    timesteps: List[float],
    outdir: Path,
    tag: str,
    device: torch.device,
    n_imgs: int = 4,
) -> None:
    from PIL import Image, ImageDraw

    n_blocks = len(model.blocks)
    imgs_dir = outdir / tag
    imgs_dir.mkdir(parents=True, exist_ok=True)

    x0 = x0_latents[:n_imgs].to(device)
    y  = all_labels[:n_imgs].to(device)
    B  = x0.shape[0]

    # Save clean reference
    ref_path = imgs_dir / "00_reference_clean_x0.png"
    if not ref_path.exists():
        decode_and_save_grid(vae, x0, ref_path)

    for t_val in timesteps:
        t_tensor = torch.full((B,), t_val, device=device, dtype=x0.dtype)
        eps      = torch.randn_like(x0)
        x_noisy  = (1 - t_val) * x0 + t_val * eps

        hidden_states, c = collect_hidden_states(model, backend, x_noisy, t_tensor, y)

        t_dir = imgs_dir / f"t{t_val:.1f}"
        t_dir.mkdir(parents=True, exist_ok=True)

        row_images: List[np.ndarray] = []

        for ell in range(n_blocks):
            H = hidden_states[ell]
            pred_patches = probe_bank.forward_layer(ell, H, c)
            pred_v = unpatchify(pred_patches)

            # Decode to PIL
            latents_scaled = pred_v.clamp(-5, 5) / 0.18215
            imgs = vae.decode(latents_scaled).sample
            imgs = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
            H_img, W_img = imgs.shape[1:3]
            cols = min(B, 4)
            rows = math.ceil(B / cols)
            canvas = np.zeros((rows * H_img, cols * W_img, 3), dtype=np.uint8)
            for i, img in enumerate(imgs):
                r, c_idx = divmod(i, cols)
                canvas[r * H_img:(r + 1) * H_img, c_idx * W_img:(c_idx + 1) * W_img] = img
            pil = Image.fromarray(canvas)

            fname = t_dir / f"layer{ell:02d}.png"
            pil.save(str(fname))
            row_images.append(canvas)

        # Summary grid
        summary = np.vstack(row_images)
        summary_pil = Image.fromarray(summary)
        draw = ImageDraw.Draw(summary_pil)
        h_per_row = row_images[0].shape[0]
        for ell in range(n_blocks):
            y_pos = ell * h_per_row + 4
            draw.text((4, y_pos), f"L{ell:02d}", fill=(255, 220, 0))

        summary_path = imgs_dir / f"summary_t{t_val:.1f}.png"
        summary_pil.save(str(summary_path))

    # Cross-timestep summary per layer
    for ell in range(n_blocks):
        t_strips: List[np.ndarray] = []
        for t_val in timesteps:
            p = imgs_dir / f"t{t_val:.1f}" / f"layer{ell:02d}.png"
            if p.exists():
                t_strips.append(np.array(Image.open(str(p))))
        if t_strips:
            combo = np.hstack(t_strips)
            combo_pil = Image.fromarray(combo)
            draw = ImageDraw.Draw(combo_pil)
            for ti, t_val in enumerate(timesteps):
                draw.text((4 + ti * t_strips[0].shape[1], 4), f"t={t_val:.1f}", fill=(255, 220, 0))
            cp = imgs_dir / f"layer{ell:02d}_all_timesteps.png"
            combo_pil.save(str(cp))



# ============================================================
# Training loop
# ============================================================

def train_one_model(
    model: nn.Module,
    backend: str,
    probe_bank: LayerProbeBank,
    loader: DataLoader,
    vae: nn.Module,
    device: torch.device,
    args,
    outdir: Path,
    tag: str,
) -> Dict[int, List[float]]:
    """
    Train all 28 probe heads jointly.
    Returns per_layer_epoch_loss: {layer_idx: [loss_epoch0, loss_epoch1, ...]}
    """
    n_blocks = len(model.blocks)
    probe_bank = probe_bank.to(device)
    optimizer = AdamW(probe_bank.parameters(), lr=args.lr, weight_decay=1e-4)

    timesteps = [float(t.strip()) for t in args.timesteps.split(",") if t.strip()]

    per_layer_epoch_loss: Dict[int, List[float]] = {i: [] for i in range(n_blocks)}
    global_epoch_loss: List[float] = []

    # A small fixed batch for visualization (grabbed before training begins)
    viz_batch = None

    print(f"\n[{tag}] Starting training: {args.epochs} epochs, "
          f"batch_size={args.batch_size}, lr={args.lr}")

    for epoch in range(args.epochs):
        probe_bank.train()
        epoch_layer_sums = {i: 0.0 for i in range(n_blocks)}
        epoch_total = 0.0
        n_iter = 0

        pbar = tqdm(loader, desc=f"[{tag}] epoch {epoch+1}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Encode to latent
            with torch.no_grad():
                x0 = encode_batch(vae, images)   # (B, 4, 32, 32)

            # Save viz batch on first iteration
            if viz_batch is None:
                viz_batch = (x0[:8].cpu(), labels[:8].cpu())

            # Sample random timestep for each image
            B = x0.shape[0]
            t_vals = torch.rand(B, device=device)                  # (B,) uniform in [0,1]
            eps = torch.randn_like(x0)
            x_noisy = (1 - t_vals.view(-1, 1, 1, 1)) * x0 + t_vals.view(-1, 1, 1, 1) * eps
            # flow matching velocity target: v = eps - x0
            v_target = eps - x0  # (B, 4, 32, 32)

            # Collect hidden states (frozen model, no grad)
            with torch.no_grad():
                hidden_states, c = collect_hidden_states(model, backend, x_noisy, t_vals, labels)

            # Train probe heads
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            layer_losses_batch: Dict[int, float] = {}

            for ell in range(n_blocks):
                H = hidden_states[ell]   # (B, T, D) — on device
                pred_patches = probe_bank.forward_layer(ell, H, c)  # (B, T, p*p*4)
                pred_img = unpatchify(pred_patches)                   # (B, 4, H, W)
                loss_ell = ((pred_img - v_target) ** 2).mean()
                total_loss = total_loss + loss_ell
                layer_losses_batch[ell] = float(loss_ell.detach().item())

            total_loss.backward()
            nn.utils.clip_grad_norm_(probe_bank.parameters(), 1.0)
            optimizer.step()

            for ell in range(n_blocks):
                epoch_layer_sums[ell] += layer_losses_batch[ell]
            epoch_total += float(total_loss.detach().item())
            n_iter += 1

            pbar.set_postfix({"total_loss": f"{epoch_total / n_iter:.4f}"})

        # Epoch summary
        avg_layer = {ell: epoch_layer_sums[ell] / max(n_iter, 1) for ell in range(n_blocks)}
        avg_total = epoch_total / max(n_iter, 1)
        for ell in range(n_blocks):
            per_layer_epoch_loss[ell].append(avg_layer[ell])
        global_epoch_loss.append(avg_total)

        print(f"\n[{tag}] epoch {epoch+1}/{args.epochs}  total_avg_loss={avg_total:.4f}")
        for ell in range(n_blocks):
            print(f"  layer {ell:2d}: {avg_layer[ell]:.4f}")

        # Visualize after every epoch (all layers, all timesteps)
        probe_bank.eval()
        if viz_batch is not None and not args.skip_viz:
            visualize_all_layers_all_timesteps_probe(
                model, backend, probe_bank, vae,
                viz_batch[0], viz_batch[1],
                timesteps=timesteps,
                outdir=outdir / "images",
                tag=f"{tag}_ep{epoch+1:02d}",
                device=device,
                n_imgs=min(args.n_viz_images, 4),
            )


    return per_layer_epoch_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 2: Trained FFN probe on SiT/REPA hidden states"
    )
    # Paths
    parser.add_argument("--sit-root", default=None,
                        help="Path to SiT repo (default: ./SiT relative to this script)")
    parser.add_argument("--repa-root", default=None,
                        help="Path to REPA repo (default: ./REPA relative to this script)")
    parser.add_argument("--sit-ckpt", default=None)
    parser.add_argument("--repa-ckpt", default=None)
    parser.add_argument("--outdir", default=None)
    # Data
    parser.add_argument("--data-dir", default=None,
                        help="ImageNet train dir (uses tiny-imagenet if not given)")
    parser.add_argument("--n-workers", type=int, default=2)
    # Training
    parser.add_argument("--backends", default="sit,repa",
                        help="Which models to train probes for: sit, repa, or sit,repa")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size; reduce if OOM")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", default="0.1,0.3,0.5,0.7,0.9",
                        help="Timesteps for eval-time loss table (not used during training)")
    parser.add_argument("--vae", default="mse", choices=["ema", "mse"])
    # Visualization
    parser.add_argument("--n-viz-images", type=int, default=4,
                        help="Number of images to decode per layer (max 4)")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip generating image grids after each epoch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve all paths relative to the script's own directory
    sit_root  = Path(args.sit_root)  if args.sit_root  else _SCRIPT_DIR / "SiT"
    repa_root = Path(args.repa_root) if args.repa_root else _SCRIPT_DIR / "REPA"
    sit_ckpt  = args.sit_ckpt  if args.sit_ckpt  else str(_SCRIPT_DIR / "pretrained_models" / "SiT-XL-2-256x256.pt")
    repa_ckpt = args.repa_ckpt if args.repa_ckpt else str(_SCRIPT_DIR / "pretrained_models" / "repa-last.pt")
    outdir_p  = Path(args.outdir) if args.outdir else _SCRIPT_DIR / "outputs" / "exp2"

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

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    eval_timesteps = [float(t.strip()) for t in args.timesteps.split(",") if t.strip()]

    # ---- Download checkpoints -----------------------------------------------
    if "sit" in backends and not os.path.isfile(args.sit_ckpt):
        download_sit_ckpt(args.sit_ckpt)
    if "repa" in backends and not os.path.isfile(args.repa_ckpt):
        download_repa_ckpt(args.repa_ckpt)

    # ---- Load VAE ----------------------------------------------------------
    print(f"[vae] loading stabilityai/sd-vae-ft-{args.vae}...")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ---- Build data loader -------------------------------------------------
    if args.data_dir:
        print(f"[data] using ImageNet from {args.data_dir}")
        loader = build_imagenet_loader(args.data_dir, args.batch_size, args.n_workers)
    else:
        loader = build_tiny_imagenet_loader(args.batch_size, args.n_workers)
    print(f"[data] dataset size: {len(loader.dataset)} images")

    # ---- Main loop over backends -------------------------------------------
    all_results: Dict[str, Dict[int, List[float]]] = {}

    for backend in backends:
        print(f"\n{'='*65}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*65}")

        # Load frozen model
        if backend == "sit":
            model = load_sit_model(args.sit_ckpt, args.sit_root, device)
        else:
            model = load_repa_model(args.repa_ckpt, args.repa_root, device)

        n_blocks = len(model.blocks)

        # Build probe bank (random init)
        probe_bank = LayerProbeBank(
            n_layers=n_blocks,
            hidden_size=1152,
            patch_size=2,
            out_channels=4,
        )

        # Train
        per_layer_history = train_one_model(
            model, backend, probe_bank, loader, vae, device, args, outdir, tag=backend
        )
        all_results[backend] = per_layer_history

        # ---- Eval-time loss table at fixed timesteps using last epoch ------
        print(f"\n[{backend}] Eval loss table (final probe, fixed timesteps)...")
        probe_bank.eval()
        probe_bank_dev = probe_bank.to(device)

        eval_sums: Dict[float, Dict[int, float]] = {t: {i: 0.0 for i in range(n_blocks)}
                                                     for t in eval_timesteps}
        n_eval = 0
        for images, labels in tqdm(loader, desc="eval", leave=False):
            if n_eval >= 10:  # ~10 batches for quick eval
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                x0 = encode_batch(vae, images)
            B = x0.shape[0]
            for t_val in eval_timesteps:
                t_tensor = torch.full((B,), t_val, device=device, dtype=x0.dtype)
                eps = torch.randn_like(x0)
                x_noisy = (1 - t_val) * x0 + t_val * eps
                v_target = eps - x0
                with torch.no_grad():
                    hidden_states, c = collect_hidden_states(model, backend, x_noisy, t_tensor, labels)
                    for ell in range(n_blocks):
                        H = hidden_states[ell]
                        pred = unpatchify(probe_bank_dev.forward_layer(ell, H, c))
                        eval_sums[t_val][ell] += float(((pred - v_target) ** 2).mean().item())
            n_eval += 1

        # Print eval table
        print(f"\n  === {backend.upper()} — Eval MSE (trained probes, velocity target) ===")
        header = f"  {'layer':>6}" + "".join(f"   t={t:.1f}" for t in eval_timesteps)
        print(header)
        for ell in range(n_blocks):
            row = f"  {ell:>6}" + "".join(
                f"  {eval_sums[t][ell] / max(n_eval, 1):7.4f}" for t in eval_timesteps
            )
            print(row)

        # Save CSV
        csv_path = outdir / f"{backend}_trained_probe_eval_loss.csv"
        with open(csv_path, "w") as f:
            f.write("layer," + ",".join(f"t={t}" for t in eval_timesteps) + "\n")
            for ell in range(n_blocks):
                f.write(str(ell) + "," + ",".join(
                    f"{eval_sums[t][ell] / max(n_eval, 1):.6f}" for t in eval_timesteps
                ) + "\n")
        print(f"\n  [saved] {csv_path}")

        # Save training history CSV
        hist_path = outdir / f"{backend}_trained_probe_train_history.csv"
        with open(hist_path, "w") as f:
            f.write("layer," + ",".join(f"epoch{e+1}" for e in range(args.epochs)) + "\n")
            for ell in range(n_blocks):
                f.write(str(ell) + "," + ",".join(
                    f"{per_layer_history[ell][e]:.6f}" for e in range(args.epochs)
                ) + "\n")
        print(f"  [saved] {hist_path}")

        # Save probes checkpoint
        probe_ckpt = outdir / f"{backend}_probe_bank.pt"
        torch.save(probe_bank.state_dict(), str(probe_ckpt))
        print(f"  [saved] probe checkpoint: {probe_ckpt}")

        del model, probe_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Final comparison summary ------------------------------------------
    print(f"\n{'='*65}")
    print("FINAL SUMMARY — per-layer training loss (last epoch)")
    print(f"{'='*65}")
    for backend in backends:
        print(f"\n  {backend.upper()}")
        for ell in range(len(all_results[backend])):
            last = all_results[backend][ell][-1] if all_results[backend][ell] else float("nan")
            print(f"    layer {ell:2d}: {last:.4f}")

    print(f"\n[done] all outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
