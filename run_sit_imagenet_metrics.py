#!/usr/bin/env python3
"""
CLI: SiT-XL/2 (vanilla SiT or REPA) hidden-state and self-attention metrics on ImageNet.
Metrics: sanity | linear | knn | cka | cknna | nc1 | ncm_acc | etf_dev | participation_ratio
| effective_rank | mad | entropy | decay | hf

Example (random latents, sanity only):
  python run_sit_imagenet_metrics.py --backend sit --ckpt path.pt --metrics sanity

Example (ImageNet root required for probes):
  python run_sit_imagenet_metrics.py --backend sit --ckpt path.pt --imagenet-root /data/imagenet \\
    --metrics linear,knn,nc1,ncm_acc,etf_dev,mad,entropy --layers all --probe-layers all \\
    --dino-model dinov2_vitb14 --linear-probe-epochs 90 --max-train-samples 5000 --max-val-samples 1000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Repo root (this file lives at /workspace)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sit_metrics.cka import centered_kernel_nearest_neighbor_alignment, linear_cka
from sit_metrics.class_geometry import ClassGeometryAccumulator, ClassGeometrySummary, ncm_predictions
from sit_metrics.data import build_imagenet_loaders, build_imagenet_per_class_loaders, encode_latents
from sit_metrics.extract import grid_size_from_N, run_forward_collect
from sit_metrics.frequency import high_frequency_ratio
from sit_metrics.geometry import (
    attention_entropy_mean,
    mean_attention_distance,
    similarity_distance_decay_slope,
    token_pairwise_manhattan_dist,
)
from sit_metrics.model_loader import load_model
from sit_metrics.probes import fit_linear_probe, knn_probe, pool_features
from sit_metrics.reference_encoders import dinov2_global_features, load_dinov2_model


CLASS_GEOMETRY_METRICS = ("nc1", "ncm_acc", "etf_dev", "participation_ratio", "effective_rank")


def parse_layers(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_layers_arg(layer_arg: str, model: torch.nn.Module) -> List[int]:
    """Comma-separated indices, or ``all`` for every block ``0 .. len(blocks)-1``."""
    s = layer_arg.strip().lower()
    if s == "all":
        return list(range(len(model.blocks)))
    return parse_layers(layer_arg)


def resolve_probe_layers(
    probe_layers_arg: Optional[str],
    probe_layer: Optional[int],
    model: torch.nn.Module,
) -> List[int]:
    if probe_layer is not None:
        return [int(probe_layer)]
    if not probe_layers_arg:
        return list(range(len(model.blocks)))
    return resolve_layers_arg(probe_layers_arg, model)


def parse_timesteps(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def append_tsv(path: Path, row: Tuple):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if not exists:
            w.writerow(["metric", "layer", "timestep", "value", "extra"])
        w.writerow(row)


@torch.no_grad()
def run_sanity(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    latent_size: int,
    batch_size: int,
) -> None:
    x = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
    t = torch.full((batch_size,), 0.5, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    L = len(model.blocks)
    layers = [0, L // 2, L - 1]
    out = run_forward_collect(model, backend, x, t, y, layers, compute_A=True)
    N = next(iter(out.H.values())).shape[1]
    p, q = grid_size_from_N(N)
    print(f"[sanity] N={N}, p=q={p}, D={model.pos_embed.shape[-1]}, heads={model.blocks[0].attn.num_heads}")
    for li in layers:
        H = out.H[li]
        A = out.A[li]
        print(f"  layer {li}: H={tuple(H.shape)}, A={tuple(A.shape)}")
    dist = token_pairwise_manhattan_dist(p, q, device=device, dtype=torch.float32)
    mad = mean_attention_distance(out.A[layers[1]], dist)
    ent = attention_entropy_mean(out.A[layers[1]])
    print(f"[sanity] MAD~{float(mad):.4f} entropy~{float(ent):.4f}")


@torch.no_grad()
def accumulate_spatial_metrics(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vae: torch.nn.Module,
    layers: Sequence[int],
    t_val: float,
    max_batches: int,
    dist: torch.Tensor,
    p: int,
    q: int,
) -> Dict[str, Dict[int, float]]:
    """Mean MAD, entropy, decay slope, HF per layer over batches."""
    sums_mad = {li: 0.0 for li in layers}
    sums_ent = {li: 0.0 for li in layers}
    sums_decay = {li: 0.0 for li in layers}
    sums_hf = {li: 0.0 for li in layers}
    n_batch = 0
    for bi, (images, labels) in enumerate(batches):
        if bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        x = encode_latents(vae, images)
        t = torch.full((B,), t_val, device=device, dtype=x.dtype)
        y = labels.to(device)
        out = run_forward_collect(model, backend, x, t, y, list(layers), compute_A=True)
        n_batch += 1
        for li in layers:
            A = out.A[li]
            H = out.H[li]
            sums_mad[li] += float(mean_attention_distance(A, dist).cpu())
            sums_ent[li] += float(attention_entropy_mean(A).cpu())
            _, slope = similarity_distance_decay_slope(H, dist, num_bins=16)
            sums_decay[li] += float(slope.cpu())
            sums_hf[li] += float(high_frequency_ratio(H, p, q).cpu())
    inv = 1.0 / max(n_batch, 1)
    out_d: Dict[str, Dict[int, float]] = {"mad": {}, "entropy": {}, "decay_slope": {}, "hf_ratio": {}}
    for li in layers:
        out_d["mad"][li] = sums_mad[li] * inv
        out_d["entropy"][li] = sums_ent[li] * inv
        out_d["decay_slope"][li] = sums_decay[li] * inv
        out_d["hf_ratio"][li] = sums_hf[li] * inv
    return out_d


def collect_pooled_features_multi(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vae: torch.nn.Module,
    layers: Sequence[int],
    t_val: float,
    max_batches: int,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    zs: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
    ys: List[torch.Tensor] = []
    for bi, (images, labels) in enumerate(batches):
        if bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        x = encode_latents(vae, images)
        t = torch.full((x.shape[0],), t_val, device=device, dtype=x.dtype)
        y = labels.to(device)
        out = run_forward_collect(model, backend, x, t, y, list(layers), compute_A=False)
        for li in layers:
            zs[li].append(pool_features(out.H[li]).float().cpu())
        ys.append(labels.cpu())
    return {li: torch.cat(parts, dim=0) for li, parts in zs.items()}, torch.cat(ys, dim=0)


def collect_reference_features(
    reference_model: torch.nn.Module,
    device: torch.device,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    image_resolution: int,
    feature_kind: str,
    max_batches: int,
) -> torch.Tensor:
    refs: List[torch.Tensor] = []
    for bi, (images, labels) in enumerate(batches):
        if bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        refs.append(dinov2_global_features(reference_model, images, image_resolution, feature_kind).float().cpu())
    return torch.cat(refs, dim=0)


def class_geometry_extra(
    train_per_class: int,
    val_per_class: int,
    ncm_metric: str,
    subset_protocol: str,
) -> str:
    return (
        "pool=mean_tokens;"
        f"subset={subset_protocol};"
        f"train_k={int(train_per_class)};"
        f"val_k={int(val_per_class)};"
        f"ncm={ncm_metric}"
    )


@torch.no_grad()
def evaluate_class_geometry_metrics(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    train_loader,
    val_loader,
    vae: torch.nn.Module,
    layers: Sequence[int],
    t_val: float,
    num_classes: int,
    ncm_metric: str,
) -> tuple[Dict[str, Dict[int, float]], Dict[int, ClassGeometrySummary]]:
    feat_dim = int(model.pos_embed.shape[-1])
    accumulators = {li: ClassGeometryAccumulator(num_classes=num_classes, feature_dim=feat_dim) for li in layers}

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        x = encode_latents(vae, images)
        t = torch.full((x.shape[0],), t_val, device=device, dtype=x.dtype)
        y = labels.to(device)
        out = run_forward_collect(model, backend, x, t, y, list(layers), compute_A=False)
        for li in layers:
            accumulators[li].update(pool_features(out.H[li]).float(), y)

    summaries = {li: accumulators[li].summary() for li in layers}
    correct = {li: 0 for li in layers}
    total = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        x = encode_latents(vae, images)
        t = torch.full((x.shape[0],), t_val, device=device, dtype=x.dtype)
        y = labels.to(device)
        out = run_forward_collect(model, backend, x, t, y, list(layers), compute_A=False)
        y_cpu = labels.cpu()
        total += int(y_cpu.shape[0])
        for li in layers:
            Z = pool_features(out.H[li]).float().cpu()
            pred = ncm_predictions(Z, summaries[li].means, metric=ncm_metric)
            correct[li] += int((pred == y_cpu).sum().item())

    metrics_out: Dict[str, Dict[int, float]] = {
        "nc1": {},
        "ncm_acc": {},
        "etf_dev": {},
        "participation_ratio": {},
        "effective_rank": {},
    }
    denom = max(total, 1)
    for li, summary in summaries.items():
        metrics_out["nc1"][li] = summary.nc1
        metrics_out["ncm_acc"][li] = float(correct[li]) / float(denom)
        metrics_out["etf_dev"][li] = summary.etf_dev
        metrics_out["participation_ratio"][li] = summary.participation_ratio
        metrics_out["effective_rank"][li] = summary.effective_rank
    return metrics_out, summaries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["sit", "repa"], default="sit")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--sit-root", type=str, default="/workspace/SiT")
    p.add_argument("--repa-root", type=str, default="/workspace/REPA")
    p.add_argument("--outdir", type=str, default="./outputs/sit_metrics")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--model-num-classes",
        type=int,
        default=1000,
        help="SiT LabelEmbedder size — MUST match checkpoint (ImageNet pretrained = 1000). "
        "Do not set to 64 for Mini-ImageNet when loading standard SiT weights.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Optional, passed to linear-probe helper; sklearn infers classes from labels. "
        "Mini-ImageNet: use 64 or leave default (probe still works).",
    )
    p.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repa-args-json", type=str, default=None)
    p.add_argument("--encoder-depth", type=int, default=None)
    p.add_argument("--projector-embed-dims", type=str, default=None)
    p.add_argument("--use-cfg", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--imagenet-root", type=str, default=None, help="train/ and val/ subfolders (ImageFolder)")
    p.add_argument("--vae", choices=["ema", "mse"], default="mse")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=5000)
    p.add_argument("--max-val-samples", type=int, default=1000)
    p.add_argument("--max-batches-spatial", type=int, default=50)
    p.add_argument("--max-batches-probe", type=int, default=200)
    p.add_argument("--linear-probe-epochs", type=int, default=90, help="Paper-style linear probe epochs.")
    p.add_argument(
        "--linear-probe-batch-size",
        type=int,
        default=16384,
        help="Paper-style linear probe batch size before min(num_train, batch_size).",
    )
    p.add_argument("--linear-probe-lr", type=float, default=1e-3, help="Paper-style linear probe initial LR.")
    p.add_argument("--class-geometry-train-per-class", type=int, default=50)
    p.add_argument("--class-geometry-val-per-class", type=int, default=20)
    p.add_argument("--class-geometry-subset-seed", type=int, default=0)
    p.add_argument("--ncm-metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument(
        "--metrics",
        type=str,
        default="sanity",
        help=(
            "comma: sanity,linear,knn,cka,cknna,nc1,ncm_acc,etf_dev,"
            "participation_ratio,effective_rank,mad,entropy,decay,hf"
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Spatial metrics (mad, entropy, decay, hf): 'all' = every block 0..L-1, or comma list e.g. 4,8,16,24,27",
    )
    p.add_argument(
        "--probe-layers",
        type=str,
        default="all",
        help="Representation metrics (linear, knn, cka, cknna): 'all' or a comma list.",
    )
    p.add_argument("--probe-layer", type=int, default=None, help="Legacy single-layer override for representation metrics.")
    p.add_argument("--cka-layer-a", type=int, default=8, help="Deprecated legacy arg; ignored by DINO-based CKA.")
    p.add_argument("--cka-layer-b", type=int, default=27, help="Deprecated legacy arg; ignored by DINO-based CKA.")
    p.add_argument("--dino-model", type=str, default="dinov2_vitb14", help="Reference encoder for CKA/CKNNA.")
    p.add_argument(
        "--dino-feature",
        choices=["mean_patch", "cls"],
        default="mean_patch",
        help="Global DINO feature used for alignment metrics.",
    )
    p.add_argument("--cknna-k", type=int, default=10, help="Neighborhood size for CKNNA.")
    p.add_argument("--timesteps", type=str, default="0.5", help="comma-separated t in (0,1)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    latent_size = args.resolution // 8

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tsv_path = outdir / "metrics.tsv"

    if "sanity" in metrics:
        run_sanity(model, args.backend, device, latent_size, min(4, args.batch_size))

    need_imagenet = any(
        m in metrics
        for m in ("linear", "knn", "cka", "cknna", "mad", "entropy", "decay", "hf", *CLASS_GEOMETRY_METRICS)
    )
    if need_imagenet and not args.imagenet_root:
        print("[error] --imagenet-root required for metrics that use real images.")
        sys.exit(1)

    if not need_imagenet:
        print("[done] (only sanity or no imagenet metrics)")
        return

    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    train_loader, val_loader = build_imagenet_loaders(
        args.imagenet_root,
        args.resolution,
        args.batch_size,
        args.num_workers,
        args.max_train_samples,
        args.max_val_samples,
    )
    train_batches = list(train_loader)
    val_batches = list(val_loader)

    geometry_train_loader = None
    geometry_val_loader = None
    geometry_num_classes = None
    geometry_subset_protocol = "per_class_fixed_split_pair"
    if any(m in metrics for m in CLASS_GEOMETRY_METRICS):
        (
            geometry_train_loader,
            geometry_val_loader,
            geometry_train_indices,
            geometry_val_indices,
            geometry_num_classes,
            geometry_subset_protocol,
        ) = build_imagenet_per_class_loaders(
            root=args.imagenet_root,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_samples_per_class=args.class_geometry_train_per_class,
            val_samples_per_class=args.class_geometry_val_per_class,
            subset_seed=args.class_geometry_subset_seed,
        )
        np.save(outdir / "train_subset_indices.npy", np.asarray(geometry_train_indices, dtype=np.int64))
        np.save(outdir / "val_subset_indices.npy", np.asarray(geometry_val_indices, dtype=np.int64))
        print(
            "[geometry] "
            f"train={len(geometry_train_indices)} val={len(geometry_val_indices)} "
            f"classes={geometry_num_classes} protocol={geometry_subset_protocol} "
            f"k_train={args.class_geometry_train_per_class} k_val={args.class_geometry_val_per_class}"
        )

    # grid / distance matrix from model
    N = model.x_embedder.num_patches
    p, q = grid_size_from_N(N)
    dist = token_pairwise_manhattan_dist(p, q, device=device, dtype=torch.float32)

    timesteps = parse_timesteps(args.timesteps)
    layers = resolve_layers_arg(args.layers, model)
    probe_layers = resolve_probe_layers(args.probe_layers, args.probe_layer, model)
    if any(m in metrics for m in ("mad", "entropy", "decay", "hf")) and not layers:
        print("[error] --layers is empty; use 'all' or a comma list of indices.")
        sys.exit(1)
    if layers:
        print(f"[spatial] MAD/entropy/decay/HF on layers {layers[0]}..{layers[-1]} ({len(layers)} total)")
    if any(m in metrics for m in ("linear", "knn", "cka", "cknna")):
        if not probe_layers:
            print("[error] --probe-layers resolved to empty; use 'all' or a comma list of indices.")
            sys.exit(1)
        print(
            f"[repr] linear/kNN/CKA/CKNNA on layers {probe_layers[0]}..{probe_layers[-1]} "
            f"({len(probe_layers)} total)"
        )
    if any(m in metrics for m in CLASS_GEOMETRY_METRICS):
        if not probe_layers:
            print("[error] --probe-layers resolved to empty; use 'all' or a comma list of indices.")
            sys.exit(1)
        print(
            f"[geometry] NC1/NCM/ETF/PR/EffRank on layers {probe_layers[0]}..{probe_layers[-1]} "
            f"({len(probe_layers)} total), ncm={args.ncm_metric}"
        )

    reference_model = None
    if any(m in metrics for m in ("cka", "cknna")):
        print(f"[reference] loading {args.dino_model} with feature={args.dino_feature}...")
        reference_model = load_dinov2_model(args.dino_model, device, args.resolution)

    for t_val in timesteps:
        if any(m in metrics for m in ("mad", "entropy", "decay", "hf")):
            sm = accumulate_spatial_metrics(
                model,
                args.backend,
                device,
                val_batches,
                vae,
                layers,
                t_val,
                args.max_batches_spatial,
                dist,
                p,
                q,
            )
            key_to_flag = {
                "mad": "mad",
                "entropy": "entropy",
                "decay_slope": "decay",
                "hf_ratio": "hf",
            }
            for name, per_l in sm.items():
                flag = key_to_flag.get(name, name)
                if flag not in metrics:
                    continue
                for li, val in per_l.items():
                    append_tsv(tsv_path, (name, li, t_val, val, ""))
                    print(f"{name} layer={li} t={t_val}: {val:.6g}")

        if any(m in metrics for m in CLASS_GEOMETRY_METRICS):
            geometry_metrics, _summaries = evaluate_class_geometry_metrics(
                model,
                args.backend,
                device,
                geometry_train_loader,
                geometry_val_loader,
                vae,
                probe_layers,
                t_val,
                geometry_num_classes,
                args.ncm_metric,
            )
            geometry_extra = class_geometry_extra(
                train_per_class=args.class_geometry_train_per_class,
                val_per_class=args.class_geometry_val_per_class,
                ncm_metric=args.ncm_metric,
                subset_protocol=geometry_subset_protocol,
            )
            for metric_name, per_layer in geometry_metrics.items():
                if metric_name not in metrics:
                    continue
                for li, val in per_layer.items():
                    append_tsv(tsv_path, (metric_name, li, t_val, val, geometry_extra))
                    print(f"{metric_name} layer={li} t={t_val}: {val:.6g}")

        Ztr_layers: Optional[Dict[int, torch.Tensor]] = None
        Zva_layers: Optional[Dict[int, torch.Tensor]] = None
        ytr: Optional[torch.Tensor] = None
        yva: Optional[torch.Tensor] = None
        if "linear" in metrics or "knn" in metrics:
            print("[probe] collecting train features across requested layers...")
            Ztr_layers, ytr = collect_pooled_features_multi(
                model,
                args.backend,
                device,
                train_batches,
                vae,
                probe_layers,
                t_val,
                args.max_batches_probe,
            )
            print("[probe] collecting val features across requested layers...")
            Zva_layers, yva = collect_pooled_features_multi(
                model,
                args.backend,
                device,
                val_batches,
                vae,
                probe_layers,
                t_val,
                args.max_batches_probe,
            )
            ytr_np = ytr.numpy()
            yva_np = yva.numpy()
            for li in probe_layers:
                Ztr_np = Ztr_layers[li].numpy()
                Zva_np = Zva_layers[li].numpy()
                if "linear" in metrics:
                    top1, top5 = fit_linear_probe(
                        Ztr_layers[li],
                        ytr,
                        Zva_layers[li],
                        yva,
                        args.num_classes,
                        device=device,
                        epochs=args.linear_probe_epochs,
                        batch_size=args.linear_probe_batch_size,
                        lr=args.linear_probe_lr,
                        seed=args.seed,
                    )
                    append_tsv(tsv_path, ("linear_top1", li, t_val, top1, ""))
                    append_tsv(tsv_path, ("linear_top5", li, t_val, top5, ""))
                    print(f"linear layer={li} top1={top1:.4f} top5={top5:.4f}")
                if "knn" in metrics:
                    k1, rk = knn_probe(Ztr_np, ytr_np, Zva_np, yva_np, k=20)
                    append_tsv(tsv_path, ("knn_top1", li, t_val, k1, ""))
                    append_tsv(tsv_path, ("knn_recall_at_k", li, t_val, rk, ""))
                    print(f"k-NN layer={li} top1={k1:.4f} recall@k={rk:.4f}")

        if "cka" in metrics or "cknna" in metrics:
            if Zva_layers is None:
                print("[alignment] collecting val features across requested layers...")
                Zva_layers, yva = collect_pooled_features_multi(
                    model,
                    args.backend,
                    device,
                    val_batches,
                    vae,
                    probe_layers,
                    t_val,
                    args.max_batches_probe,
                )
            print("[alignment] collecting DINO reference features on val...")
            Zref = collect_reference_features(
                reference_model,
                device,
                val_batches,
                args.resolution,
                args.dino_feature,
                args.max_batches_probe,
            )
            ref_extra = f"ref={args.dino_model};feature={args.dino_feature}"
            for li in probe_layers:
                if "cka" in metrics:
                    cka_score = float(linear_cka(Zva_layers[li], Zref).item())
                    append_tsv(tsv_path, ("cka", li, t_val, cka_score, ref_extra))
                    print(f"CKA(layer {li} vs {args.dino_model}/{args.dino_feature}) = {cka_score:.6f}")
                if "cknna" in metrics:
                    cknna_score = float(
                        centered_kernel_nearest_neighbor_alignment(Zva_layers[li], Zref, k=args.cknna_k).item()
                    )
                    append_tsv(tsv_path, ("cknna", li, t_val, cknna_score, f"{ref_extra};k={args.cknna_k}"))
                    print(
                        f"CKNNA(layer {li} vs {args.dino_model}/{args.dino_feature}, k={args.cknna_k}) "
                        f"= {cknna_score:.6f}"
                    )

    print(f"[saved] {tsv_path}")


if __name__ == "__main__":
    main()
