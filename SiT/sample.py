# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
import os
from time import time
import numpy as np

# Fix matplotlib backend issue in Jupyter/Colab
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


#################################################################################
#               Block-wise Cosine Similarity Functions                          #
#################################################################################

def compute_block_cosine_matrix(block_tokens: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between all blocks.

    Args:
        block_tokens: List of hidden tokens from each block.
                     Each element has shape [B, N, D] where:
                     - B: batch size
                     - N: number of patches/tokens
                     - D: hidden dimension

    Returns:
        sim_mat: Cosine similarity matrix of shape [L, L] where L is number of blocks.
                 sim_mat[a, b] = average cosine similarity between block a and block b.
    """
    if not block_tokens:
        return None

    # Stack all block outputs: [L, B, N, D]
    H = torch.stack(block_tokens, dim=0)
    L, B, N, D = H.shape

    # Normalize along the hidden dimension
    H_norm = F.normalize(H, p=2, dim=-1)  # [L, B, N, D]

    # Compute pairwise cosine similarity: [L, L, B, N]
    sim = torch.einsum('ibnd,jbnd->ijbn', H_norm, H_norm)

    # Average over batch and patches: [L, L]
    sim_mat = sim.mean(dim=(-1, -2))

    return sim_mat


def visualize_similarity_matrices(all_sim_mats, timesteps, output_path, model_depth):
    """Visualize block-wise cosine similarity matrices across timesteps."""
    num_timesteps = len(timesteps)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx in range(min(6, num_timesteps)):
        if idx >= len(all_sim_mats):
            axes[idx].axis('off')
            continue

        sim_mat = all_sim_mats[idx]
        sns.heatmap(
            sim_mat, ax=axes[idx], cmap='RdYlBu_r', vmin=0.0, vmax=1.0, square=True,
            cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=range(model_depth), yticklabels=range(model_depth)
        )
        axes[idx].set_title(f'Timestep {timesteps[idx]}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Block Index', fontsize=12)
        axes[idx].set_ylabel('Block Index', fontsize=12)

    for idx in range(num_timesteps, 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved similarity matrix visualization to {output_path}")
    plt.close()


def visualize_spatial_heatmap(block_tokens, block_a, block_b, grid_size, output_path):
    """Visualize spatial heatmap of cosine similarity for a pair of blocks."""
    h_a = block_tokens[block_a]
    h_b = block_tokens[block_b]

    h_a_norm = F.normalize(h_a, p=2, dim=-1)
    h_b_norm = F.normalize(h_b, p=2, dim=-1)

    patch_sim = torch.sum(h_a_norm * h_b_norm, dim=-1)
    patch_sim_avg = patch_sim.mean(dim=0)
    patch_map = patch_sim_avg.reshape(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(8, 7))
    sns.heatmap(patch_map, cmap='RdYlBu_r', vmin=0.0, vmax=1.0, square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f'Spatial Similarity: Block {block_a} vs Block {block_b}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Patch X', fontsize=12)
    plt.ylabel('Patch Y', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spatial heatmap to {output_path}")
    plt.close()


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        # For custom checkpoints, try to auto-detect learn_sigma from checkpoint
        # Most 256x256 models use learn_sigma=True
        learn_sigma = True if args.image_size == 256 else False

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)

    # Block analysis mode
    if hasattr(args, 'analyze_blocks') and args.analyze_blocks:
        print(f"\n{'='*60}")
        print(f"BLOCK ANALYSIS MODE ENABLED")

        # Parse noise levels
        noise_levels = [float(x) for x in args.noise_levels.split(',')]
        print(f"Noise levels: {noise_levels}")

        model_depth = len(model.blocks)
        grid_size = int(model.x_embedder.num_patches ** 0.5)
        print(f"Model depth: {model_depth} blocks")
        print(f"Track at mid-point of sampling")
        print(f"{'='*60}\n")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_by_noise = {}

        # Loop through each noise level
        for noise_scale in noise_levels:
            print(f"\n{'='*60}")
            print(f"Processing noise level: {noise_scale:.2f}")
            print(f"{'='*60}")

            # Create noise with specific scale
            z = torch.randn(n, 4, latent_size, latent_size, device=device) * noise_scale
            y = torch.tensor(class_labels, device=device)

            # Setup CFG
            z_cfg = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y_cfg = torch.cat([y, y_null], 0)

            # Sample normally first
            start_time = time()
            if mode == "SDE":
                sample_fn = sampler.sample_sde(
                    sampling_method=args.sampling_method,
                    diffusion_form=args.diffusion_form,
                    diffusion_norm=args.diffusion_norm,
                    last_step=args.last_step,
                    last_step_size=args.last_step_size,
                    num_steps=args.num_sampling_steps,
                )
                samples = sample_fn(z_cfg, model.forward_with_cfg, y=y_cfg, cfg_scale=args.cfg_scale)[-1]
            else:
                sample_fn = sampler.sample_ode(
                    sampling_method=args.sampling_method,
                    num_steps=args.num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                    reverse=args.reverse
                )
                samples = sample_fn(z_cfg, model.forward_with_cfg, y=y_cfg, cfg_scale=args.cfg_scale)[-1]

            print(f"Sampling took {time() - start_time:.2f} seconds.")

            # Now run one forward pass at mid-point to get block tokens
            with torch.no_grad():
                # Recreate the state at mid-point for analysis
                z_mid = torch.randn(n, 4, latent_size, latent_size, device=device) * noise_scale
                z_mid_cfg = torch.cat([z_mid, z_mid], 0)

                # Use middle timestep
                t_mid = 0.5  # Middle of [0, 1]
                t_batch = torch.ones(z_mid_cfg.size(0), device=device) * t_mid

                # Get block tokens
                _, block_tokens = model.forward_with_cfg(
                    z_mid_cfg, t_batch, y_cfg, args.cfg_scale, return_block_tokens=True
                )
                similarity_matrix = compute_block_cosine_matrix(block_tokens)
                print(f"Collected similarity at t={t_mid:.2f} (separate forward pass)")

            # Extract conditional samples
            samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample

            # Save results for this noise level
            noise_dir = output_dir / f"noise_{noise_scale:.2f}"
            noise_dir.mkdir(exist_ok=True)

            # Save images
            save_image(samples, noise_dir / "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
            for i, img in enumerate(samples):
                save_image(img, noise_dir / f"sample_{i:02d}.png", normalize=True, value_range=(-1, 1))

            # Save similarity matrix
            if similarity_matrix is not None:
                sim_mat_np = similarity_matrix.cpu().numpy()
                results_by_noise[noise_scale] = sim_mat_np
                np.save(noise_dir / "similarity_matrix.npy", sim_mat_np)

                # Plot individual heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(sim_mat_np, annot=False, cmap='viridis', vmin=0, vmax=1, square=True)
                plt.title(f'Block Similarity - Noise={noise_scale:.2f}')
                plt.xlabel('Block Index')
                plt.ylabel('Block Index')
                plt.tight_layout()
                plt.savefig(noise_dir / "similarity_matrix.png", dpi=150)
                plt.close()

        # Create comparison visualization
        if len(results_by_noise) > 1:
            print(f"\n{'='*60}")
            print("Creating noise level comparison...")
            print(f"{'='*60}")

            n_levels = len(noise_levels)
            ncols = min(5, n_levels)
            nrows = (n_levels + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3))
            if n_levels == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, noise_level in enumerate(sorted(results_by_noise.keys())):
                sim_mat = results_by_noise[noise_level]
                sns.heatmap(sim_mat, ax=axes[idx], annot=False, cmap='viridis',
                           vmin=0, vmax=1, square=True, cbar=True)
                axes[idx].set_title(f'Noise={noise_level:.2f}')
                axes[idx].set_xlabel('Block')
                axes[idx].set_ylabel('Block')

            for idx in range(n_levels, len(axes)):
                axes[idx].axis('off')

            plt.suptitle('Block-wise Cosine Similarity Across Noise Levels')
            plt.tight_layout()
            plt.savefig(output_dir / "noise_comparison.png", dpi=150)
            plt.close()

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}\n")

    else:
        # Standard sampling (no block analysis)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        start_time = time()
        samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
        samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / 0.18215).sample
        print(f"Sampling took {time() - start_time:.2f} seconds.")
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--analyze-blocks", action="store_true",
                        help="Enable block-wise cosine similarity analysis")
    parser.add_argument("--noise-levels", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated noise levels for analysis (default: 0.1,0.2,...,1.0)")
    parser.add_argument("--output-dir", type=str, default="./similarity_results",
                        help="Output directory for similarity analysis results")


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
