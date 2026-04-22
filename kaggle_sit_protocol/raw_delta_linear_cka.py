from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle_sit_protocol.config import ProtocolConfig
    from kaggle_sit_protocol.model_specs import apply_sit_model_spec, sit_model_slug
    from kaggle_sit_protocol.modeling import forward_features, load_sit_model
    from kaggle_sit_protocol.progress import progress
else:
    from .config import ProtocolConfig
    from .model_specs import apply_sit_model_spec, sit_model_slug
    from .modeling import forward_features, load_sit_model
    from .progress import progress


def _device_from_config(config: ProtocolConfig) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean linear-CKA heatmaps for raw SiT block activations and "
            "layer deltas L_i - L_{i-1}."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override ProtocolConfig.model_name, e.g. SiT-B/2.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for the requested model.",
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
        "--forward-batch-size",
        type=int,
        default=None,
        help="Micro-batch size inside each timestep forward pass. Defaults to cfg.extraction_batch_size.",
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
        default=1e-12,
        help="Numerical epsilon used by the CKA normalization.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="float16",
        help="Autocast dtype on CUDA for lower activation memory.",
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


def _apply_model_overrides(config: ProtocolConfig, args: argparse.Namespace) -> None:
    if args.model_name is not None:
        config.model_name = str(args.model_name)
    if args.checkpoint_path is not None:
        config.checkpoint_path = str(args.checkpoint_path)
    apply_sit_model_spec(config)


def _resolve_forward_batch_size(args: argparse.Namespace, config: ProtocolConfig) -> int:
    if args.forward_batch_size is not None:
        return max(int(args.forward_batch_size), 1)
    return max(int(config.extraction_batch_size), 1)


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _pairwise_linear_cka_batched(tensors: torch.Tensor, eps: float) -> torch.Tensor:
    if tensors.ndim != 4:
        raise ValueError(f"Expected [B, L, N, D] tensor, got {tuple(tensors.shape)}")

    centered = tensors.float() - tensors.float().mean(dim=2, keepdim=True)
    grams = torch.matmul(centered, centered.transpose(-1, -2))
    flat = grams.reshape(grams.shape[0], grams.shape[1], -1)
    hsic = torch.einsum("bli,bji->blj", flat, flat)
    denom = (
        torch.diagonal(hsic, dim1=-2, dim2=-1).clamp_min(eps).sqrt().unsqueeze(-1)
        * torch.diagonal(hsic, dim1=-2, dim2=-1).clamp_min(eps).sqrt().unsqueeze(-2)
    )
    return (hsic / denom.clamp_min(eps)).clamp_(0.0, 1.0)


@torch.no_grad()
def _accumulate_batch_metrics(
    *,
    model: torch.nn.Module,
    config: ProtocolConfig,
    device: torch.device,
    x1: torch.Tensor,
    x0: torch.Tensor,
    labels: torch.Tensor,
    raw_sum: torch.Tensor,
    delta_sum: torch.Tensor,
    forward_batch_size: int,
    eps: float,
    amp_dtype: torch.dtype | None,
) -> None:
    x1 = x1.to(device=device, dtype=torch.float32)
    x0 = x0.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)

    batch_size = int(x1.shape[0])
    for time_pos, time_value in enumerate(config.time_values):
        time_value_f = float(time_value)
        for start in range(0, batch_size, forward_batch_size):
            stop = min(start + forward_batch_size, batch_size)
            x1_chunk = x1[start:stop]
            x0_chunk = x0[start:stop]
            labels_chunk = labels[start:stop]

            xt_chunk = time_value_f * x1_chunk + (1.0 - time_value_f) * x0_chunk
            t_chunk = torch.full((stop - start,), time_value_f, device=device, dtype=torch.float32)
            with _autocast_context(device, amp_dtype):
                _, blocks = forward_features(model, xt_chunk, t_chunk, labels_chunk)

            raw_batched = blocks.permute(1, 0, 2, 3).contiguous()
            raw_sum[time_pos] += _pairwise_linear_cka_batched(raw_batched, eps=eps).sum(dim=0).to(torch.float64).cpu()

            deltas = (blocks[1:] - blocks[:-1]).permute(1, 0, 2, 3).contiguous()
            delta_sum[time_pos] += _pairwise_linear_cka_batched(deltas, eps=eps).sum(dim=0).to(torch.float64).cpu()

            del xt_chunk, t_chunk, blocks, raw_batched, deltas
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del x1, x0, labels


def _longform_rows(
    *,
    kind: str,
    matrix_by_time: np.ndarray,
    time_indices: tuple[int, ...],
    time_values: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    num_layers = int(matrix_by_time.shape[1])
    for time_pos, (time_index, time_value) in enumerate(zip(time_indices, time_values)):
        matrix = matrix_by_time[time_pos]
        for i in range(num_layers):
            for j in range(num_layers):
                if kind == "raw":
                    layer_i = i + 1
                    layer_j = j + 1
                    label_i = f"L{layer_i}"
                    label_j = f"L{layer_j}"
                elif kind == "delta":
                    layer_i = i + 2
                    layer_j = j + 2
                    label_i = f"L{layer_i}-L{layer_i - 1}"
                    label_j = f"L{layer_j}-L{layer_j - 1}"
                else:
                    raise ValueError(f"Unsupported kind={kind!r}")
                rows.append(
                    {
                        "kind": kind,
                        "time_position": time_pos,
                        "time_index": int(time_index),
                        "time_value": float(time_value),
                        "layer_i": layer_i,
                        "layer_j": layer_j,
                        "layer_i_label": label_i,
                        "layer_j_label": label_j,
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
    _apply_model_overrides(config, args)
    if args.device is not None:
        config.device = str(args.device)
    config.ensure_directories()

    model_slug = sit_model_slug(config.model_name).lower()
    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.analysis_dir / f"raw_delta_linear_cka_{model_slug}_{args.subset_role}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(config.manifest_path)
    latents = torch.load(config.cache_dir / "x1_latents_fp16.pt", map_location="cpu")
    noise = torch.load(config.cache_dir / "x0_noise_seed0_fp16.pt", map_location="cpu")
    device = _device_from_config(config)
    model = load_sit_model(config, device)
    forward_batch_size = _resolve_forward_batch_size(args, config)
    amp_dtype = _resolve_amp_dtype(device, args.amp_dtype)

    subset_rows = manifest[manifest["subset_role"] == args.subset_role].reset_index(drop=True)
    if args.max_images is not None:
        subset_rows = subset_rows.head(int(args.max_images)).reset_index(drop=True)
    if subset_rows.empty:
        raise ValueError(f"No rows found for subset_role={args.subset_role!r}")

    latent_key, noise_key, position_lookup = _subset_arrays(args.subset_role, latents, noise)
    num_times = len(config.time_values)
    raw_sum = torch.zeros((num_times, config.num_layers, config.num_layers), dtype=torch.float64)
    delta_sum = torch.zeros((num_times, config.num_layers - 1, config.num_layers - 1), dtype=torch.float64)
    image_count = 0

    for start in progress(
        range(0, len(subset_rows), args.image_batch_size),
        desc="Raw/delta linear CKA",
        total=(len(subset_rows) + args.image_batch_size - 1) // args.image_batch_size,
    ):
        batch_rows = subset_rows.iloc[start : start + args.image_batch_size]
        positions = [position_lookup[str(image_id)] for image_id in batch_rows["image_id"].tolist()]
        x1 = latents[latent_key][positions]
        x0 = noise[noise_key][positions]
        labels = torch.as_tensor(batch_rows["imagenet_idx"].to_numpy(), dtype=torch.long)
        _accumulate_batch_metrics(
            model=model,
            config=config,
            device=device,
            x1=x1,
            x0=x0,
            labels=labels,
            raw_sum=raw_sum,
            delta_sum=delta_sum,
            forward_batch_size=forward_batch_size,
            eps=args.eps,
            amp_dtype=amp_dtype,
        )
        image_count += len(batch_rows)
        del x1, x0, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_mean = (raw_sum / float(max(image_count, 1))).numpy().astype(np.float32)
    delta_mean = (delta_sum / float(max(image_count, 1))).numpy().astype(np.float32)

    np.savez(
        outdir / "raw_delta_linear_cka_matrices.npz",
        model_name=np.array(config.model_name),
        subset_role=np.array(args.subset_role),
        image_count=np.array(image_count, dtype=np.int64),
        time_positions=np.arange(num_times, dtype=np.int64),
        time_indices=np.array(config.time_grid_indices, dtype=np.int64),
        time_values=np.array(config.time_values, dtype=np.float32),
        raw_layers=np.arange(1, config.num_layers + 1, dtype=np.int64),
        delta_layers=np.arange(2, config.num_layers + 1, dtype=np.int64),
        raw=raw_mean,
        delta=delta_mean,
        raw_mean_over_time=raw_mean.mean(axis=0).astype(np.float32),
        delta_mean_over_time=delta_mean.mean(axis=0).astype(np.float32),
    )

    rows: list[dict[str, object]] = []
    rows.extend(
        _longform_rows(
            kind="raw",
            matrix_by_time=raw_mean,
            time_indices=config.time_grid_indices,
            time_values=config.time_values,
        )
    )
    rows.extend(
        _longform_rows(
            kind="delta",
            matrix_by_time=delta_mean,
            time_indices=config.time_grid_indices,
            time_values=config.time_values,
        )
    )
    pd.DataFrame(rows).to_csv(outdir / "raw_delta_linear_cka_long.csv", index=False)

    mean_rows: list[dict[str, object]] = []
    for kind, matrix in (("raw", raw_mean.mean(axis=0)), ("delta", delta_mean.mean(axis=0))):
        for row in _longform_rows(
            kind=kind,
            matrix_by_time=matrix[None, ...],
            time_indices=(config.time_grid_indices[0],),
            time_values=(config.time_values[0],),
        ):
            row.pop("time_position", None)
            row.pop("time_index", None)
            row.pop("time_value", None)
            mean_rows.append(row)
    pd.DataFrame(mean_rows).to_csv(outdir / "raw_delta_linear_cka_mean_over_time.csv", index=False)

    print(f"Saved matrices to: {outdir / 'raw_delta_linear_cka_matrices.npz'}")
    print(f"Saved long-form CSV to: {outdir / 'raw_delta_linear_cka_long.csv'}")
    print(f"Saved mean-over-time CSV to: {outdir / 'raw_delta_linear_cka_mean_over_time.csv'}")


if __name__ == "__main__":
    main()
