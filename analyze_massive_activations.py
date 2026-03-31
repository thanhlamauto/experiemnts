#!/usr/bin/env python3
"""
Visualize Massive Activations (Outlier Dimensions) in SiT and REPA.
Inspired by the SD3.5 Massive Activations analysis.
Produces 3D plots of [Tokens x Dimensions] for specific layers.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add local paths for SiT and REPA
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_SCRIPT_DIR))

from sit_metrics.model_loader import load_model

@torch.no_grad()
def get_activations(model, backend, device, layer_idx):
    """Run one forward pass and grab activation of a specific layer."""
    model.eval()
    
    # Simple random input
    z = torch.randn(1, 4, 32, 32).to(device)
    t = torch.tensor([0.5]).to(device) # mid-noise
    y = torch.randint(0, 1000, (1,)).to(device)
    
    activation = None
    def hook(module, input, output):
        nonlocal activation
        # output is [B, T, D]
        activation = output.detach().cpu()

    # Register hook on the specific block
    handle = model.blocks[layer_idx].register_forward_hook(hook)
    
    if backend == 'sit':
        model(z, t, y)
    else:
        model(z, t, y) # REPA model forward
        
    handle.remove()
    return activation[0] # [T, D]

def plot_3d_activations(activations, title, save_path, top_k_dims=40):
    """
    activations: [T, D]
    We visualize a subset of tokens and dimensions to match the paper style.
    """
    T, D = activations.shape
    abs_act = activations.abs().numpy()
    
    # Identify the top-k dimensions with highest maximum activation across any token
    max_per_dim = abs_act.max(axis=0)
    top_dims = np.argsort(-max_per_dim)[:top_k_dims]
    top_dims = sorted(top_dims) # keep index order
    
    # Filter activations to only these dimensions
    filtered_act = abs_act[:, top_dims]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')

    # X: tokens, Y: dimensions
    x_pos, y_pos = np.meshgrid(np.arange(T), np.arange(len(top_dims)))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    
    dx = dy = 0.8 # width of bars
    dz = filtered_act.T.flatten() # heights
    
    # Color mapping based on height
    norm = plt.Normalize(dz.min(), dz.max())
    colors = plt.cm.viridis(norm(dz))

    # To keep it fast, we only plot bars that have meaningful magnitude
    mask = dz > (dz.max() * 0.05)
    
    ax.bar3d(x_pos[mask], y_pos[mask], z_pos[mask], dx, dy, dz[mask], color=colors[mask], alpha=0.8, shade=True)
    
    ax.set_xlabel('Patch tokens', labelpad=15, fontweight='bold')
    ax.set_ylabel('Top Dimensions (Ranked)', labelpad=15, fontweight='bold')
    ax.set_zlabel('Magnitude |h|', labelpad=15, fontweight='bold')
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    
    # Label the axes with actual dimension indices for the top few
    peak_idx = np.argmax(max_per_dim)
    ax.text2D(0.05, 0.95, f"Global Peak Dim: {peak_idx}\nMax Magnitude: {max_per_dim[peak_idx]:.2f}", 
              transform=ax.transAxes, color='#3498db', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#000000')
    plt.close()
    print(f"Saved 3D plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    # Support relative paths
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--sit-ckpt", type=str, default=str(base_dir / "SiT/pretrained_models/SiT-XL-2-256x256.pt"))
    parser.add_argument("--repa-ckpt", type=str, default=str(base_dir / "REPA/pretrained_models/last.pt"))
    parser.add_argument("--outdir", type=str, default="outputs/massive_activations")
    parser.add_argument("--layers", type=str, default="2,12,25")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    layer_indices = [int(x) for x in args.layers.split(",")]
    
    # Verify ckpts
    sit_ckpt = Path(args.sit_ckpt)
    repa_ckpt = Path(args.repa_ckpt)
    
    configs = []
    if sit_ckpt.exists():
        configs.append(('sit', str(sit_ckpt), "SiT Vanilla"))
    else:
        print(f"[skip] SiT ckpt not found at {sit_ckpt}")
        
    if repa_ckpt.exists():
        configs.append(('repa', str(repa_ckpt), "REPA"))
    else:
        print(f"[skip] REPA ckpt not found at {repa_ckpt}")
    
    for backend, ckpt, name in configs:
        print(f"\n{'='*40}\nProcessing {name}...\n{'='*40}")
        try:
            model = load_model(
                backend=backend,
                ckpt=ckpt,
                sit_root=str(base_dir / "SiT"),
                repa_root=str(base_dir / "REPA"),
                device=device,
                encoder_depth=8,
                projector_embed_dims="768"
            )
            
            for l_idx in layer_indices:
                if l_idx >= len(model.blocks): 
                    print(f"  [skip] Layer {l_idx} invalid for this model.")
                    continue
                print(f"  Extracting Layer {l_idx}...")
                act = get_activations(model, backend, device, l_idx)
                save_path = outdir / f"massive_{backend}_L{l_idx:02d}.png"
                plot_3d_activations(act, f"{name}: Layer {l_idx} Massive Activations", save_path)
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[error] Failed to process {name}: {e}")

if __name__ == "__main__":
    main()
