from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.metrics import pairwise_cos_batchmean_nd
    from kaggle_sit_protocol.modeling import forward_features, linear_path_xt, load_sit_model
    from kaggle_sit_protocol.progress import progress
else:
    from .config import ProtocolConfig
    from .metrics import pairwise_cos_batchmean_nd
    from .modeling import forward_features, linear_path_xt, load_sit_model
    from .progress import progress


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarities for layer deltas L_i - L_{i-1} on SiT block outputs."
    )
    parser.add_argument(
        "--subset-role",
        choices=("main", "control"),
        default="main",
        help="Which manifest subset to evaluate.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of images from the chosen subset.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=4,
        help="How many images to process together when recomputing block outputs.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Override ProtocolConfig.device.",
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
        help="Optional explicit manifest path override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV/NPZ outputs. Defaults under cfg.analysis_dir.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used by the cosine normalizations.",
    )
    return parser.parse_args()


def _subset_arrays(
    subset_role: str,
    latents: dict[str, torch.Tensor],
    noise: dict[str, torch.Tensor],
) -> tuple[str, str, dict[str, int]]:
    if subset_role == "main":
        latent_key = "main_latents"
        noise_key = "main_noise"
        ids = latents["main_image_ids"]
    elif subset_role == "control":
        latent_key = "control_latents"
        noise_key = "control_noise"
        ids = latents["control_image_ids"]
    else:
        raise ValueError(f"Unsupported subset_role={subset_role!r}")
    return latent_key, noise_key, {str(image_id): idx for idx, image_id in enumerate(ids)}


@torch.no_grad()
def _extract_block_batch(
    model: torch.nn.Module,
    config: ProtocolConfig,
    device: torch.device,
    x1: torch.Tensor,
    x0: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    x1 = x1.to(device=device, dtype=torch.float32)
    x0 = x0.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    timesteps = torch.tensor(config.time_values, device=device, dtype=torch.float32)
    xt = linear_path_xt(x1, x0, timesteps)
    batch_size = int(x1.shape[0])
    num_times = int(timesteps.shape[0])
    flat_xt = xt.permute(1, 0, 2, 3, 4).reshape(batch_size * num_times, *x1.shape[1:])
    flat_t = timesteps.repeat(batch_size)
    flat_labels = labels.repeat_interleave(num_times)
    _, blocks = forward_features(model, flat_xt, flat_t, flat_labels)
    return blocks.reshape(config.num_layers, batch_size, num_times, blocks.shape[-2], blocks.shape[-1]).detach().cpu()


def _streaming_flat_gram(tensors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tlbnd,tibnd->tli", tensors, tensors)


def _streaming_token_sum(tensors: torch.Tensor, eps: float) -> torch.Tensor:
    x = F.normalize(tensors, dim=-1, eps=eps)
    return torch.einsum("tlbnd,tibnd->tli", x, x)


def _finalize_flatcos(gram: torch.Tensor, eps: float) -> torch.Tensor:
    diag = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(eps).sqrt()
    denom = diag.unsqueeze(-1) * diag.unsqueeze(-2)
    return gram / denom.clamp_min(eps)


def _finalize_batchmean(sum_over_batch: torch.Tensor, batch_count: int, eps: float) -> torch.Tensor:
    mean_tensor = (sum_over_batch / float(max(batch_count, 1))).unsqueeze(2)
    per_time = []
    for time_pos in range(mean_tensor.shape[0]):
        per_time.append(pairwise_cos_batchmean_nd(mean_tensor[time_pos], eps=eps))
    return torch.stack(per_time, dim=0)


def _longform_rows(
    metric_name: str,
    matrix_by_time: np.ndarray,
    time_indices: tuple[int, ...],
    time_values: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_deltas = int(matrix_by_time.shape[1])
    for time_pos, (time_index, time_value) in enumerate(zip(time_indices, time_values)):
        matrix = matrix_by_time[time_pos]
        for i in range(num_deltas):
            for j in range(num_deltas):
                rows.append(
                    {
                        "metric": metric_name,
                        "time_position": time_pos,
                        "time_index": int(time_index),
                        "time_value": float(time_value),
                        "delta_i": i + 2,
                        "delta_j": j + 2,
                        "delta_i_label": f"L{i + 2}-L{i + 1}",
                        "delta_j_label": f"L{j + 2}-L{j + 1}",
                        "value": float(matrix[i, j]),
                    }
                )
    return rows


def main() -> None:
    args = _parse_args()
    config = ProtocolConfig.from_kaggle_defaults()
    if args.output_root is not None:
        config.output_root = str(args.output_root)
        config.manifest_path = f"{config.output_root}/cache/manifest.parquet"
    if args.manifest_path is not None:
        config.manifest_path = str(args.manifest_path)
    if args.device is not None:
        config.device = str(args.device)
    config.ensure_directories()

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.analysis_dir / f"residual_delta_cosine_{args.subset_role}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(config.manifest_path)
    latents = torch.load(config.cache_dir / "x1_latents_fp16.pt", map_location="cpu")
    noise = torch.load(config.cache_dir / "x0_noise_seed0_fp16.pt", map_location="cpu")
    device = _device_from_config(config)
    model = load_sit_model(config, device)

    subset_rows = manifest[manifest["subset_role"] == args.subset_role].reset_index(drop=True)
    if args.max_images is not None:
        subset_rows = subset_rows.head(int(args.max_images)).reset_index(drop=True)
    if subset_rows.empty:
        raise ValueError(f"No rows found for subset_role={args.subset_role!r}")

    latent_key, noise_key, position_lookup = _subset_arrays(args.subset_role, latents, noise)
    num_times = len(config.time_values)
    num_delta_layers = config.num_layers - 1
    num_tokens = config.patch_grid_size * config.patch_grid_size

    flat_gram = torch.zeros((num_times, num_delta_layers, num_delta_layers), dtype=torch.float64)
    token_sum = torch.zeros((num_times, num_delta_layers, num_delta_layers), dtype=torch.float64)
    batchmean_sum = torch.zeros(
        (num_times, num_delta_layers, num_tokens, config.hidden_dim),
        dtype=torch.float32,
    )
    image_count = 0

    for start in progress(
        range(0, len(subset_rows), args.image_batch_size),
        desc="Residual delta cosine",
        total=(len(subset_rows) + args.image_batch_size - 1) // args.image_batch_size,
    ):
        batch_rows = subset_rows.iloc[start : start + args.image_batch_size]
        positions = [position_lookup[str(image_id)] for image_id in batch_rows["image_id"].tolist()]
        x1 = latents[latent_key][positions]
        x0 = noise[noise_key][positions]
        labels = torch.as_tensor(batch_rows["imagenet_idx"].to_numpy(), dtype=torch.long)

        blocks = _extract_block_batch(model, config, device, x1, x0, labels).float()
        delta = (blocks[1:] - blocks[:-1]).permute(2, 0, 1, 3, 4).contiguous()

        flat_gram += _streaming_flat_gram(delta).to(torch.float64)
        token_sum += _streaming_token_sum(delta, eps=args.eps).to(torch.float64)
        batchmean_sum += delta.sum(dim=2)
        image_count += len(batch_rows)

    flatcos = _finalize_flatcos(flat_gram, eps=args.eps).cpu()
    tokenwise = (token_sum / float(max(image_count * num_tokens, 1))).cpu()
    batchmean = _finalize_batchmean(batchmean_sum, batch_count=image_count, eps=args.eps).cpu()

    np.savez(
        outdir / "residual_delta_cosine_matrices.npz",
        subset_role=np.array(args.subset_role),
        image_count=np.array(image_count, dtype=np.int64),
        time_positions=np.arange(num_times, dtype=np.int64),
        time_indices=np.array(config.time_grid_indices, dtype=np.int64),
        time_values=np.array(config.time_values, dtype=np.float32),
        delta_layers=np.arange(2, config.num_layers + 1, dtype=np.int64),
        flat_bnd=flatcos.numpy().astype(np.float32),
        tokenwise=tokenwise.numpy().astype(np.float32),
        batchmean_nd=batchmean.numpy().astype(np.float32),
        flat_bnd_mean_over_time=flatcos.mean(dim=0).numpy().astype(np.float32),
        tokenwise_mean_over_time=tokenwise.mean(dim=0).numpy().astype(np.float32),
        batchmean_nd_mean_over_time=batchmean.mean(dim=0).numpy().astype(np.float32),
    )

    rows: list[dict[str, object]] = []
    rows.extend(_longform_rows("flat_bnd", flatcos.numpy(), config.time_grid_indices, config.time_values))
    rows.extend(_longform_rows("tokenwise", tokenwise.numpy(), config.time_grid_indices, config.time_values))
    rows.extend(_longform_rows("batchmean_nd", batchmean.numpy(), config.time_grid_indices, config.time_values))
    pd.DataFrame(rows).to_csv(outdir / "residual_delta_cosine_long.csv", index=False)

    mean_over_time_rows = []
    for metric_name, matrix in (
        ("flat_bnd", flatcos.mean(dim=0).numpy()),
        ("tokenwise", tokenwise.mean(dim=0).numpy()),
        ("batchmean_nd", batchmean.mean(dim=0).numpy()),
    ):
        for i in range(num_delta_layers):
            for j in range(num_delta_layers):
                mean_over_time_rows.append(
                    {
                        "metric": metric_name,
                        "delta_i": i + 2,
                        "delta_j": j + 2,
                        "delta_i_label": f"L{i + 2}-L{i + 1}",
                        "delta_j_label": f"L{j + 2}-L{j + 1}",
                        "value": float(matrix[i, j]),
                    }
                )
    pd.DataFrame(mean_over_time_rows).to_csv(outdir / "residual_delta_cosine_mean_over_time.csv", index=False)


if __name__ == "__main__":
    main()
