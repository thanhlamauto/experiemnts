#!/usr/bin/env python3
"""
CLI: paper-style spatial hidden-state metrics on SiT-XL/2 checkpoints.

This runner compares hidden states on a canonical noise-level axis shared across
SiT and REPA, while mapping to each backend's native timestep convention.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sit_metrics.data import build_imagenet_indexed_loader, encode_latents
from sit_metrics.dense_probes import (
    f1_from_counts,
    fit_binary_token_probe,
    fit_multiclass_token_probe,
    mean_iou_from_confusion,
    objectness_iou_from_mask,
    update_binary_f1_counts,
    update_confusion_matrix,
)
from sit_metrics.extract import grid_size_from_N, run_forward_collect
from sit_metrics.geometry import spatial_metric_bundle, token_pairwise_manhattan_dist
from sit_metrics.frequency import high_frequency_ratio
from sit_metrics.model_loader import load_model
from sit_metrics.noising import (
    build_noise_bank,
    canonical_noise_level_to_model_t,
    canonical_noise_level_to_xt,
    parse_noise_levels,
)
from sit_metrics.pseudo_masks import PseudoMaskTargets, load_pseudo_mask_targets

DEFAULT_FULL_NOISE_LEVELS = "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.0"
DEFAULT_SPATIAL_METRICS = "lds,cds,rmsc,lgr,msdr,graph_gap,ubc,hf_ratio"
SUPPORTED_SPATIAL_METRICS = {
    "lds",
    "cds",
    "rmsc",
    "lgr",
    "msdr",
    "graph_gap",
    "ubc",
    "hf_ratio",
    "patch_miou",
    "boundary_f1",
    "objectness_iou",
}


def parse_layers(spec: str, model_depth: int) -> List[int]:
    text = spec.strip().lower()
    if text == "all":
        return list(range(model_depth))
    layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    for layer in layers:
        if layer < 0 or layer >= model_depth:
            raise ValueError(f"layer {layer} out of range [0, {model_depth})")
    return layers


def resolve_outdir(outdir: str | None, backend: str) -> Path:
    if outdir:
        return Path(outdir)
    return Path(f"/workspace/outputs/{backend}_imagenet_spatial_metrics")


def parse_float_csv(spec: str | None) -> List[float] | None:
    if spec is None:
        return None
    items = [x.strip() for x in spec.split(",") if x.strip()]
    if not items:
        return None
    return [float(x) for x in items]


def split_train_val_positions(num_items: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError(f"pseudo probe train_frac must be in (0,1), got {train_frac}")
    if num_items < 2:
        raise ValueError("Need at least 2 images to form pseudo probe train/val splits")

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator)
    n_train = int(round(float(train_frac) * num_items))
    n_train = max(1, min(n_train, num_items - 1))

    train_mask = torch.zeros(num_items, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask = ~train_mask
    return train_mask, val_mask


def _append_sampled_tokens(
    feature_store: Dict[int, List[torch.Tensor]],
    label_store: Dict[int, List[torch.Tensor]],
    counts: Dict[int, int],
    layer: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    max_tokens: int,
) -> None:
    remaining = int(max_tokens) - int(counts[layer])
    if remaining <= 0 or features.numel() == 0:
        return
    take = min(int(remaining), int(features.shape[0]))
    if take <= 0:
        return

    if take < features.shape[0]:
        perm = torch.randperm(features.shape[0], device=features.device)[:take]
        features = features[perm]
        labels = labels[perm]

    feature_store[layer].append(features.detach().cpu().to(torch.float16))
    label_store[layer].append(labels.detach().cpu().to(torch.int64))
    counts[layer] += take


def remap_dense_labels(labels: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, int]:
    valid = labels != int(ignore_index)
    if not bool(valid.any()):
        return labels.clone(), 0

    unique = torch.unique(labels[valid].cpu(), sorted=True)
    remapped = torch.full_like(labels, fill_value=int(ignore_index))
    mapped_vals = torch.searchsorted(unique, labels[valid].cpu())
    remapped[valid] = mapped_vals.to(dtype=torch.int64, device=labels.device)
    return remapped, int(unique.numel())


def write_rows_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "ckpt",
        "metric",
        "layer",
        "noise_level",
        "model_t",
        "value",
        "num_images",
        "path_type",
        "noise_protocol",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def accumulate_metrics_for_noise_level(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    loader,
    vae: torch.nn.Module,
    layers: Sequence[int],
    dist: torch.Tensor,
    noise_bank_cpu: torch.Tensor,
    noise_level: float,
    path_type: str,
    r_near: float,
    r_far: float,
    p: int,
    q: int,
    metrics: Sequence[str],
    lgr_tau: float,
    msdr_sigmas: Sequence[float],
    msdr_weights: Sequence[float] | None,
    graph_knn_k: int,
    graph_max_images: int | None,
    ubc_topk_ratio: float,
    hf_radius_frac: float,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, int], float, int]:
    model_t = canonical_noise_level_to_model_t(backend, noise_level, path_type=path_type)
    sums: Dict[str, Dict[int, float]] = {metric: {layer: 0.0 for layer in layers} for metric in metrics}
    counts: Dict[str, int] = {metric: 0 for metric in metrics}
    total_images = 0
    graph_images_used = 0
    requested = set(metrics)
    full_bundle_metrics = [m for m in metrics if m not in {"graph_gap", "hf_ratio"}]

    for images, labels, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_clean = encode_latents(vae, images)
        eps = noise_bank_cpu[sample_ids].to(device=device, dtype=x_clean.dtype, non_blocking=True)
        x_t = canonical_noise_level_to_xt(x_clean, eps, noise_level, path_type=path_type)
        t = torch.full((x_t.shape[0],), model_t, device=device, dtype=x_t.dtype)
        out = run_forward_collect(model, backend, x_t, t, labels, layers, compute_A=False)

        batch_size = x_t.shape[0]
        total_images += batch_size
        graph_batch_weight = 0
        if "graph_gap" in requested:
            if graph_max_images is None:
                graph_batch_weight = batch_size
            else:
                graph_batch_weight = min(batch_size, max(int(graph_max_images) - graph_images_used, 0))
        for layer in layers:
            H = out.H[layer]
            if full_bundle_metrics:
                bundle = spatial_metric_bundle(
                    H,
                    dist,
                    r_near=r_near,
                    r_far=r_far,
                    metrics=full_bundle_metrics,
                    p=p,
                    q=q,
                    lgr_tau=lgr_tau,
                    msdr_sigmas=msdr_sigmas,
                    msdr_weights=msdr_weights,
                    graph_knn_k=graph_knn_k,
                    ubc_topk_ratio=ubc_topk_ratio,
                )
                for metric in full_bundle_metrics:
                    sums[metric][layer] += float(bundle[metric].cpu()) * batch_size
            if "hf_ratio" in requested:
                hf = high_frequency_ratio(H, p, q, high_freq_radius_frac=hf_radius_frac)
                sums["hf_ratio"][layer] += float(hf.cpu()) * batch_size
            if graph_batch_weight > 0:
                graph_bundle = spatial_metric_bundle(
                    H[:graph_batch_weight],
                    dist,
                    r_near=r_near,
                    r_far=r_far,
                    metrics=("graph_gap",),
                    graph_knn_k=graph_knn_k,
                )
                sums["graph_gap"][layer] += float(graph_bundle["graph_gap"].cpu()) * graph_batch_weight
        for metric in full_bundle_metrics:
            counts[metric] += batch_size
        if "hf_ratio" in requested:
            counts["hf_ratio"] += batch_size
        if "graph_gap" in requested:
            graph_images_used += graph_batch_weight
            counts["graph_gap"] = graph_images_used

        del out, x_t, x_clean, eps, t

    if total_images == 0:
        raise RuntimeError("No validation images were processed")

    averages: Dict[str, Dict[int, float]] = {metric: {} for metric in metrics}
    for metric in metrics:
        denom = float(counts[metric])
        if denom <= 0:
            raise RuntimeError(f"Metric {metric} received zero images; adjust sampling/configuration.")
        inv = 1.0 / denom
        for layer in layers:
            averages[metric][layer] = sums[metric][layer] * inv
    return averages, counts, model_t, total_images


@torch.no_grad()
def accumulate_pseudo_metrics_for_noise_level(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    loader,
    vae: torch.nn.Module,
    layers: Sequence[int],
    noise_bank_cpu: torch.Tensor,
    noise_level: float,
    path_type: str,
    pseudo_targets: PseudoMaskTargets,
    requested_metrics: Sequence[str],
    pseudo_probe_train_frac: float,
    pseudo_probe_seed: int,
    pseudo_probe_max_train_tokens: int,
    pseudo_probe_epochs: int,
    pseudo_probe_batch_size: int,
    pseudo_probe_lr: float,
    pseudo_ignore_index: int,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, int]]:
    requested = set(requested_metrics)
    num_images = int(noise_bank_cpu.shape[0])
    train_mask, val_mask = split_train_val_positions(num_images, pseudo_probe_train_frac, pseudo_probe_seed)
    val_count = int(val_mask.sum().item())

    need_patch = "patch_miou" in requested
    need_boundary = "boundary_f1" in requested
    need_object = "objectness_iou" in requested
    model_t = canonical_noise_level_to_model_t(backend, noise_level, path_type=path_type)

    patch_targets = None
    patch_num_classes = 0
    if need_patch:
        patch_targets, patch_num_classes = remap_dense_labels(pseudo_targets.patch_labels, pseudo_ignore_index)
        if patch_num_classes <= 0:
            raise RuntimeError("Pseudo patch labels contain no valid tokens.")

    patch_features: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    patch_labels: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    patch_counts: Dict[int, int] = {layer: 0 for layer in layers}

    boundary_features: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    boundary_labels: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    boundary_counts: Dict[int, int] = {layer: 0 for layer in layers}

    objectness_sums: Dict[int, float] = {layer: 0.0 for layer in layers}
    objectness_counts: Dict[int, int] = {layer: 0 for layer in layers}

    for images, labels, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_ids_cpu = sample_ids.to(torch.long).cpu()

        x_clean = encode_latents(vae, images)
        eps = noise_bank_cpu[sample_ids_cpu].to(device=device, dtype=x_clean.dtype, non_blocking=True)
        x_t = canonical_noise_level_to_xt(x_clean, eps, noise_level, path_type=path_type)
        t = torch.full((x_t.shape[0],), model_t, device=device, dtype=x_t.dtype)
        out = run_forward_collect(model, backend, x_t, t, labels, layers, compute_A=False)

        train_sel = train_mask[sample_ids_cpu].to(device=device)
        val_sel = val_mask[sample_ids_cpu].to(device=device)
        patch_batch = None
        boundary_batch = None
        object_batch = None
        if need_patch:
            patch_batch = patch_targets[sample_ids_cpu].to(device=device)
        if need_boundary:
            boundary_batch = pseudo_targets.boundary_labels[sample_ids_cpu].to(device=device)
        if need_object:
            object_batch = pseudo_targets.object_masks[sample_ids_cpu].to(device=device)

        for layer in layers:
            H = out.H[layer].float()
            if need_patch and bool(train_sel.any()) and patch_counts[layer] < pseudo_probe_max_train_tokens:
                feats = H[train_sel].reshape(-1, H.shape[-1])
                labs = patch_batch[train_sel].reshape(-1)
                valid = labs != int(pseudo_ignore_index)
                _append_sampled_tokens(
                    patch_features,
                    patch_labels,
                    patch_counts,
                    layer,
                    feats[valid],
                    labs[valid],
                    pseudo_probe_max_train_tokens,
                )
            if need_boundary and bool(train_sel.any()) and boundary_counts[layer] < pseudo_probe_max_train_tokens:
                feats = H[train_sel].reshape(-1, H.shape[-1])
                labs = boundary_batch[train_sel].reshape(-1)
                valid = labs != int(pseudo_ignore_index)
                _append_sampled_tokens(
                    boundary_features,
                    boundary_labels,
                    boundary_counts,
                    layer,
                    feats[valid],
                    labs[valid],
                    pseudo_probe_max_train_tokens,
                )
            if need_object and bool(val_sel.any()):
                score, count = objectness_iou_from_mask(
                    H[val_sel],
                    object_batch[val_sel],
                    ignore_index=pseudo_ignore_index,
                )
                if count > 0 and torch.isfinite(score):
                    objectness_sums[layer] += float(score.cpu()) * count
                    objectness_counts[layer] += count

        del out, x_t, x_clean, eps, t

    patch_models: Dict[int, torch.nn.Module] = {}
    boundary_models: Dict[int, torch.nn.Module] = {}
    if need_patch:
        for layer in layers:
            if patch_counts[layer] <= 0:
                raise RuntimeError(f"Layer {layer} collected zero pseudo training tokens for patch_miou.")
            Z = torch.cat(patch_features[layer], dim=0).float()
            y = torch.cat(patch_labels[layer], dim=0).long()
            patch_models[layer] = fit_multiclass_token_probe(
                Z,
                y,
                num_classes=patch_num_classes,
                device=device,
                epochs=pseudo_probe_epochs,
                batch_size=pseudo_probe_batch_size,
                lr=pseudo_probe_lr,
                seed=pseudo_probe_seed,
            )

    if need_boundary:
        for layer in layers:
            if boundary_counts[layer] <= 0:
                raise RuntimeError(f"Layer {layer} collected zero pseudo training tokens for boundary_f1.")
            Z = torch.cat(boundary_features[layer], dim=0).float()
            y = torch.cat(boundary_labels[layer], dim=0).long()
            boundary_models[layer] = fit_binary_token_probe(
                Z,
                y,
                device=device,
                epochs=pseudo_probe_epochs,
                batch_size=pseudo_probe_batch_size,
                lr=pseudo_probe_lr,
                seed=pseudo_probe_seed,
            )

    patch_confusion: Dict[int, torch.Tensor] = {}
    boundary_stats: Dict[int, Dict[str, int]] = {}
    if need_patch:
        patch_confusion = {layer: torch.zeros(patch_num_classes, patch_num_classes, dtype=torch.int64) for layer in layers}
    if need_boundary:
        boundary_stats = {layer: {"tp": 0, "fp": 0, "fn": 0} for layer in layers}

    if need_patch or need_boundary:
        for images, labels, sample_ids in loader:
            sample_ids_cpu = sample_ids.to(torch.long).cpu()
            val_sel = val_mask[sample_ids_cpu]
            if not bool(val_sel.any()):
                continue
            val_sel_device = val_sel.to(device=device)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_clean = encode_latents(vae, images)
            eps = noise_bank_cpu[sample_ids_cpu].to(device=device, dtype=x_clean.dtype, non_blocking=True)
            x_t = canonical_noise_level_to_xt(x_clean, eps, noise_level, path_type=path_type)
            t = torch.full((x_t.shape[0],), model_t, device=device, dtype=x_t.dtype)
            out = run_forward_collect(model, backend, x_t, t, labels, layers, compute_A=False)

            patch_batch = None
            boundary_batch = None
            if need_patch:
                patch_batch = patch_targets[sample_ids_cpu[val_sel]].to(device=device)
            if need_boundary:
                boundary_batch = pseudo_targets.boundary_labels[sample_ids_cpu[val_sel]].to(device=device)

            for layer in layers:
                H = out.H[layer][val_sel_device].reshape(-1, out.H[layer].shape[-1]).float()
                if need_patch:
                    target = patch_batch.reshape(-1)
                    valid = target != int(pseudo_ignore_index)
                    if bool(valid.any()):
                        logits = patch_models[layer](H[valid])
                        pred = logits.argmax(dim=-1)
                        update_confusion_matrix(patch_confusion[layer], pred, target[valid])
                if need_boundary:
                    target = boundary_batch.reshape(-1)
                    valid = target != int(pseudo_ignore_index)
                    if bool(valid.any()):
                        logits = boundary_models[layer](H[valid]).squeeze(-1)
                        pred = logits > 0.0
                        update_binary_f1_counts(boundary_stats[layer], pred, target[valid] > 0)

            del out, x_t, x_clean, eps, t

    metrics_out: Dict[str, Dict[int, float]] = {}
    counts_out: Dict[str, int] = {}

    if need_patch:
        metrics_out["patch_miou"] = {layer: mean_iou_from_confusion(patch_confusion[layer]) for layer in layers}
        counts_out["patch_miou"] = val_count
    if need_boundary:
        metrics_out["boundary_f1"] = {layer: f1_from_counts(boundary_stats[layer]) for layer in layers}
        counts_out["boundary_f1"] = val_count
    if need_object:
        metrics_out["objectness_iou"] = {}
        for layer in layers:
            if objectness_counts[layer] <= 0:
                metrics_out["objectness_iou"][layer] = float("nan")
            else:
                metrics_out["objectness_iou"][layer] = objectness_sums[layer] / float(objectness_counts[layer])
        counts_out["objectness_iou"] = max(objectness_counts.values()) if objectness_counts else 0

    return metrics_out, counts_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sit", "repa"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--imagenet-root", type=str, required=True)
    parser.add_argument("--sit-root", type=str, default="/workspace/SiT")
    parser.add_argument("--repa-root", type=str, default="/workspace/REPA")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae", choices=["ema", "mse"], default="mse")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--noise-levels", type=str, default=DEFAULT_FULL_NOISE_LEVELS)
    parser.add_argument("--metrics", type=str, default=DEFAULT_SPATIAL_METRICS)
    parser.add_argument("--path-type", type=str, default="linear")
    parser.add_argument("--lds-near-radius", type=float, default=None)
    parser.add_argument("--lds-far-radius", type=float, default=None)
    parser.add_argument("--lgr-tau", type=float, default=10.0)
    parser.add_argument("--msdr-sigmas", type=str, default="1.0,2.0,4.0")
    parser.add_argument("--msdr-weights", type=str, default=None)
    parser.add_argument("--graph-knn-k", type=int, default=10)
    parser.add_argument(
        "--graph-max-images",
        type=int,
        default=128,
        help="If set, averages graph_gap over at most this many images per noise level to keep runtime practical.",
    )
    parser.add_argument("--ubc-topk-ratio", type=float, default=0.1)
    parser.add_argument("--hf-radius-frac", type=float, default=0.5)
    parser.add_argument(
        "--pseudo-mask-npz",
        type=str,
        default=None,
        help=(
            "NPZ containing one or more of: patch_labels, boundary_labels, object_masks. "
            "Arrays may be [N,T], [N,p,q], or [N,H,W]. Optional dataset_indices/subset_indices "
            "align rows to the current validation subset."
        ),
    )
    parser.add_argument("--pseudo-ignore-index", type=int, default=-1)
    parser.add_argument(
        "--pseudo-background-label",
        type=int,
        default=None,
        help="Needed when deriving binary object masks from patch_labels for objectness_iou.",
    )
    parser.add_argument("--pseudo-probe-train-frac", type=float, default=0.8)
    parser.add_argument("--pseudo-probe-seed", type=int, default=0)
    parser.add_argument("--pseudo-probe-max-train-tokens", type=int, default=10000)
    parser.add_argument("--pseudo-probe-epochs", type=int, default=20)
    parser.add_argument("--pseudo-probe-batch-size", type=int, default=4096)
    parser.add_argument("--pseudo-probe-lr", type=float, default=1e-3)
    parser.add_argument("--model-num-classes", type=int, default=1000)
    parser.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repa-args-json", type=str, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--projector-embed-dims", type=str, default=None)
    parser.add_argument("--use-cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unsupported = sorted(set(metrics) - SUPPORTED_SPATIAL_METRICS)
    if unsupported:
        raise SystemExit(f"Unsupported metrics requested: {', '.join(unsupported)}")
    pseudo_metrics = [m for m in metrics if m in {"patch_miou", "boundary_f1", "objectness_iou"}]
    base_metrics = [m for m in metrics if m not in {"patch_miou", "boundary_f1", "objectness_iou"}]
    if pseudo_metrics and not args.pseudo_mask_npz:
        raise SystemExit(
            "Pseudo metrics requested but `--pseudo-mask-npz` was not provided. "
            "Add the NPZ or drop patch_miou/boundary_f1/objectness_iou from --metrics."
        )
    msdr_sigmas = parse_float_csv(args.msdr_sigmas)
    if not msdr_sigmas:
        raise SystemExit("--msdr-sigmas must contain at least one positive sigma")
    msdr_weights = parse_float_csv(args.msdr_weights)
    if msdr_weights is not None and len(msdr_weights) != len(msdr_sigmas):
        raise SystemExit("--msdr-weights must match --msdr-sigmas length")
    graph_max_images = None if args.graph_max_images is not None and args.graph_max_images <= 0 else args.graph_max_images

    model = load_model(
        backend=args.backend,
        ckpt=args.ckpt,
        device=device,
        sit_root=args.sit_root,
        repa_root=args.repa_root,
        resolution=args.resolution,
        num_classes=args.model_num_classes,
        learn_sigma=args.learn_sigma,
        repa_args_json=args.repa_args_json,
        encoder_depth=args.encoder_depth,
        projector_embed_dims=args.projector_embed_dims,
        use_cfg=args.use_cfg,
        fused_attn=args.fused_attn,
        qk_norm=args.qk_norm,
        legacy=args.legacy,
    )
    model.eval()

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    loader, subset_indices = build_imagenet_indexed_loader(
        root=args.imagenet_root,
        split="val",
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
        subset_seed=args.subset_seed,
    )

    latent_size = args.resolution // 8
    noise_bank_cpu = build_noise_bank(
        num_samples=len(subset_indices),
        sample_shape=(model.in_channels, latent_size, latent_size),
        seed=args.noise_seed,
        dtype=torch.float32,
    )

    N = model.x_embedder.num_patches
    p, q = grid_size_from_N(N)
    dist = token_pairwise_manhattan_dist(p, q, device=device, dtype=torch.float32)
    r_near = float(args.lds_near_radius if args.lds_near_radius is not None else p / 2.0)
    r_far = float(args.lds_far_radius if args.lds_far_radius is not None else p / 2.0)
    layers = parse_layers(args.layers, len(model.blocks))
    noise_levels = parse_noise_levels(args.noise_levels)

    print(
        f"[info] backend={args.backend} ckpt={args.ckpt} num_images={len(subset_indices)} "
        f"layers={len(layers)} noise_levels={len(noise_levels)} path_type={args.path_type}"
    )

    outdir = resolve_outdir(args.outdir, args.backend)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "subset_indices.npy", np.asarray(subset_indices, dtype=np.int64))

    pseudo_targets = None
    if pseudo_metrics:
        pseudo_targets = load_pseudo_mask_targets(
            args.pseudo_mask_npz,
            subset_indices=subset_indices,
            p=p,
            q=q,
            requested_metrics=pseudo_metrics,
            ignore_index=args.pseudo_ignore_index,
            background_label=args.pseudo_background_label,
        )

    rows: List[Dict[str, object]] = []
    tsv_path = outdir / "metrics.tsv"
    noise_protocol = "fixed_per_image"

    for noise_level in noise_levels:
        averages: Dict[str, Dict[int, float]] = {}
        counts: Dict[str, int] = {}
        model_t = canonical_noise_level_to_model_t(args.backend, noise_level, path_type=args.path_type)

        if base_metrics:
            base_averages, base_counts, model_t, _total_images = accumulate_metrics_for_noise_level(
                model=model,
                backend=args.backend,
                device=device,
                loader=loader,
                vae=vae,
                layers=layers,
                dist=dist,
                noise_bank_cpu=noise_bank_cpu,
                noise_level=noise_level,
                path_type=args.path_type,
                r_near=r_near,
                r_far=r_far,
                p=p,
                q=q,
                metrics=base_metrics,
                lgr_tau=args.lgr_tau,
                msdr_sigmas=msdr_sigmas,
                msdr_weights=msdr_weights,
                graph_knn_k=args.graph_knn_k,
                graph_max_images=graph_max_images,
                ubc_topk_ratio=args.ubc_topk_ratio,
                hf_radius_frac=args.hf_radius_frac,
            )
            averages.update(base_averages)
            counts.update(base_counts)

        if pseudo_metrics:
            pseudo_averages, pseudo_counts = accumulate_pseudo_metrics_for_noise_level(
                model=model,
                backend=args.backend,
                device=device,
                loader=loader,
                vae=vae,
                layers=layers,
                noise_bank_cpu=noise_bank_cpu,
                noise_level=noise_level,
                path_type=args.path_type,
                pseudo_targets=pseudo_targets,
                requested_metrics=pseudo_metrics,
                pseudo_probe_train_frac=args.pseudo_probe_train_frac,
                pseudo_probe_seed=args.pseudo_probe_seed,
                pseudo_probe_max_train_tokens=args.pseudo_probe_max_train_tokens,
                pseudo_probe_epochs=args.pseudo_probe_epochs,
                pseudo_probe_batch_size=args.pseudo_probe_batch_size,
                pseudo_probe_lr=args.pseudo_probe_lr,
                pseudo_ignore_index=args.pseudo_ignore_index,
            )
            averages.update(pseudo_averages)
            counts.update(pseudo_counts)

        for metric in metrics:
            for layer in layers:
                value = averages[metric][layer]
                rows.append(
                    {
                        "backend": args.backend,
                        "ckpt": args.ckpt,
                        "metric": metric,
                        "layer": layer,
                        "noise_level": f"{noise_level:.8f}",
                        "model_t": f"{model_t:.8f}",
                        "value": f"{value:.8f}",
                        "num_images": counts[metric],
                        "path_type": args.path_type,
                        "noise_protocol": noise_protocol,
                    }
                )
                print(
                    f"{metric} layer={layer:02d} noise={noise_level:.3f} "
                    f"model_t={model_t:.3f} value={value:.6g}"
                )

    write_rows_tsv(tsv_path, rows)
    print(f"[saved] {tsv_path}")


if __name__ == "__main__":
    main()
