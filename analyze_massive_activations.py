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
def get_activations(model, backend, device, layer_idx, t_val=0.5):
    """Run one forward pass and grab activation of a specific layer at fixed t."""
    model.eval()
    
    # Simple random input
    z = torch.randn(1, 4, 32, 32).to(device)
    t = torch.tensor([t_val]).to(device) 
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

def plot_3d_activations(activations, title, save_path, top_k_dims=1152):
    """
    activations: [T, D]
    Visualizes dimensions. If D is large, use surface plot for performance.
    """
    T, D = activations.shape
    abs_act = activations.abs().numpy()
    
    # Identify top-k dimensions if requested, otherwise take all
    if top_k_dims < D:
        max_per_dim = abs_act.max(axis=0)
        top_dims = np.argsort(-max_per_dim)[:top_k_dims]
        top_dims = sorted(top_dims)
        filtered_act = abs_act[:, top_dims]
    else:
        top_dims = np.arange(D)
        filtered_act = abs_act

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')

    # Create coordinate grids
    x_tokens = np.arange(T)
    y_dims = np.arange(len(top_dims))
    X, Y = np.meshgrid(x_tokens, y_dims)
    Z = filtered_act.T # [len(top_dims), T]
    
    # Selection of plot type based on density
    if len(top_dims) <= 64:
        # Use bars for sparse visualization
        x_pos = X.flatten()
        y_pos = Y.flatten()
        z_pos = np.zeros_like(x_pos)
        dz = Z.flatten()
        dx = dy = 0.8
        norm = plt.Normalize(dz.min(), dz.max())
        colors = plt.cm.viridis(norm(dz))
        mask = dz > (dz.max() * 0.02) # Hide floor noise for speed
        ax.bar3d(x_pos[mask], y_pos[mask], z_pos[mask], dx, dy, dz[mask], color=colors[mask], alpha=0.8, shade=True)
    else:
        # Use surface for dense visualization (1152 dims)
        # Antialiasing=False makes it faster and cleaner for spiky data
        surf = ax.plot_surface(X, Y, Z, cmap='magma', antialiased=False, linewidth=0, alpha=0.9, rstride=1, cstride=1)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Magnitude |h|')

    ax.set_xlabel('Patch tokens (T)', labelpad=15, fontweight='bold')
    ax.set_ylabel('Dimensions (D)', labelpad=15, fontweight='bold')
    ax.set_zlabel('Magnitude |h|', labelpad=15, fontweight='bold')
    ax.set_title(title + f" (D={len(top_dims)})", pad=20, fontsize=18, fontweight='bold')
    
    # Peak info
    peak_dim = np.argmax(abs_act.max(axis=0))
    peak_val = abs_act[:, peak_dim].max()
    ax.text2D(0.05, 0.95, f"Massive Spike @ Dim: {peak_dim}\nPeak Magnitude: {peak_val:.2f}", 
              transform=ax.transAxes, color='#3498db', fontsize=14, fontweight='bold')

    # Dynamic camera angle to see spikes better
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    # Remove bbox_inches='tight' to ensure uniform pixel dimensions for all plots
    plt.savefig(save_path, dpi=250, facecolor='#000000')
    plt.close()
    print(f"Saved 3D plot (D={len(top_dims)}) to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    # Support relative paths
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--sit-ckpt", type=str, default=str(base_dir / "SiT/pretrained_models/SiT-XL-2-256x256.pt"))
    parser.add_argument("--repa-ckpt", type=str, default=str(base_dir / "REPA/pretrained_models/last.pt"))
    parser.add_argument("--outdir", type=str, default="outputs/massive_activations")
    parser.add_argument("--layers", type=str, default="2,12,25")
    parser.add_argument("--timesteps", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--top-k-dims", type=int, default=1152) 
    parser.add_argument("--backends", type=str, default="sit,repa", help="sit, repa, or sit,repa")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    layer_indices = [int(x) for x in args.layers.split(",")]
    t_vals = [float(x) for x in args.timesteps.split(",")]
    selected_backends = [b.strip().lower() for b in args.backends.split(",")]
    
    # Verify ckpts
    sit_ckpt = Path(args.sit_ckpt)
    repa_ckpt = Path(args.repa_ckpt)
    
    configs = []
    if 'sit' in selected_backends:
        if sit_ckpt.exists():
            configs.append(('sit', str(sit_ckpt), "SiT Vanilla"))
        else:
            print(f"[skip] SiT ckpt not found at {sit_ckpt}")
        
    if 'repa' in selected_backends:
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
                if l_idx >= len(model.blocks): continue
                for tv in t_vals:
                    print(f"  Extracting Layer {l_idx} at t={tv}...")
                    act = get_activations(model, backend, device, l_idx, t_val=tv)
                    t_str = f"{tv:.1f}".replace(".", "p")
                    save_path = outdir / f"massive_{backend}_L{l_idx:02d}_t{t_str}.png"
                    plot_3d_activations(act, f"{name}: Layer {l_idx} (t={tv}) Massive Activations", save_path, top_k_dims=args.top_k_dims)
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[error] Failed to process {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
