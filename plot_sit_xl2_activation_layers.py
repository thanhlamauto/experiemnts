#!/usr/bin/env python3
# Copyright / usage: standalone script for analyzing SiT hidden-state activation magnitudes.
"""
Activations along layers (heatmap) for SiT-XL/2 — massive-activations style figure.

What we measure
---------------
- Hook point: output of each ``SiTBlock`` (after both residual branches: attention + MLP).
  Tensor shape ``[B, T, D]``: batch, patch tokens, hidden size. We do NOT include the initial
  patch embed + pos_embed as a "layer" in this plot; layer index 1..L corresponds to block 1..L.
- Scalar per element: ``|h|`` (absolute value of each hidden unit activation).
- Per layer, per log-spaced bin of ``|h|``: we accumulate histogram counts over **all** tokens from
  **all** micro-batches, then ``proportion = count / sum(counts)`` independently for each layer
  (column normalization). This matches pooling all activations across samples before normalizing.

Differences vs. typical LLM "Activations along layers" plots
------------------------------------------------------------
- Here ``T`` is spatial patch tokens over a 2D latent grid, not text tokens; ``D`` is the ViT
  channel dimension (e.g. 1152 for XL). The measurement recipe (|h| histograms per block) matches
  the paper spirit.

Histogram bins
--------------
- Edges ``np.logspace(log10(a_min), log10(a_max), num_bins + 1)``. If ``--a-max`` is omitted, the
  upper edge is **(1 + margin) × max |h| observed** over the run (two forward passes: one to
  measure max, one to histogram). Pass ``--a-max`` explicitly to pin the range (single pass).

Timestep sampling
-----------------
- By default, ``t`` is drawn i.i.d. per sample from ``Uniform(t_min, t_max)`` with ``t_min=0``, ``t_max=1``,
  i.e. the full continuous range from maximum noise (T) down to clean (0) in the usual [0, 1] convention.
  Override with ``--t-min`` / ``--t-max`` if you need an interior range (e.g. numerical epsilon).

Per-noise heatmaps (optional)
----------------------------
- With ``--per-noise-heatmaps``, also writes one figure per fixed ``t`` (default: 12 values
  ``linspace(0,1)``). Bins match the mixed-t run so plots are comparable. Output:
  ``<outdir>/per_noise/activations_sit_xl2_t_<tag>.{png,pdf}``. Use ``--noise-ts`` to set levels;
  ``--samples-per-noise`` to change sample count per level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Matplotlib (Agg chỉ bật trong main() khi lưu figure — để notebook có thể dùng inline backend)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator


def _add_repo_paths(sit_root: str, repa_root: str, backend: str) -> None:
    if backend == "sit":
        sys.path.insert(0, sit_root)
    else:
        sys.path.insert(0, repa_root)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "", 1) if k.startswith("module.") else k
        out[nk] = v
    return out


def _load_checkpoint_raw(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict):
        if "ema" in obj and isinstance(obj["ema"], dict):
            return obj["ema"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        # Often a raw state_dict
        if any(k.endswith("blocks.0.attn.qkv.weight") for k in obj.keys()):
            return obj
        # Fallback: first tensor-valued dict
        for k, v in obj.items():
            if isinstance(v, dict) and any(str(x).endswith(".weight") for x in v.keys()):
                return v
    raise ValueError(f"Could not parse checkpoint at {path}")


def load_repa_args_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_transformer_candidates(model: torch.nn.Module, max_lines: int = 80) -> None:
    """Debug: list modules that look like transformer blocks."""
    lines = []
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__
        if "Block" in cn or "block" in name.lower():
            lines.append(f"  {name}: {cn}")
    print("[model tree] candidate blocks (truncated):")
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} more)")


def register_block_output_hooks(
    blocks: torch.nn.ModuleList,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[Optional[torch.Tensor]]]:
    """Forward hooks on each block; output is post-residual hidden state [B,T,D]."""
    storage: List[Optional[torch.Tensor]] = [None] * len(blocks)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(i: int):
        def _hook(_module, _inp, out):
            # out: [B, T, D]
            storage[i] = out.detach()

        return _hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(_make_hook(i)))
    return handles, storage


def build_log_bins(a_min: float, a_max: float, num_bins: int) -> np.ndarray:
    lo = np.log10(a_min)
    hi = np.log10(a_max)
    return np.logspace(lo, hi, num_bins + 1)


def histogram_abs_counts(abs_flat: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(abs_flat, bins=bin_edges)
    return counts.astype(np.float64)


def parse_noise_ts(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def default_noise_ts() -> List[float]:
    """Discrete t values in [0, 1] (12 steps, endpoints inclusive)."""
    return [float(x) for x in np.linspace(0.0, 1.0, 12)]


def counts_to_heatmap(count_sum: np.ndarray) -> np.ndarray:
    col_sums = count_sum.sum(axis=1, keepdims=True)
    return np.divide(
        count_sum,
        np.maximum(col_sums, 1e-12),
        out=np.zeros_like(count_sum),
        where=col_sums > 0,
    )


def _proportion_cbar_ticks(vmin: float, vmax: float) -> list[float]:
    """Powers of 10 from vmin to vmax (e.g. 1e-4 … 1e-1)."""
    lo = int(np.floor(np.log10(vmin)))
    hi = int(np.ceil(np.log10(vmax)))
    return [10**k for k in range(lo, hi + 1)]


def save_activation_heatmap_figure(
    count_sum: np.ndarray,
    bin_edges: np.ndarray,
    num_layers: int,
    a_min: float,
    a_max_plot: float,
    png_path: Path,
    pdf_path: Path,
    title: str,
    dpi: int,
    *,
    prop_vmin: float = 1e-4,
    prop_vmax: float = 1e-1,
) -> None:
    """Log-y histogram heatmap: rows = |h| bins, cols = layer.

    Proportion color scale: log mapping (default 10⁻⁴–10⁻¹), plasma, zeros → black.
    """
    heatmap = counts_to_heatmap(count_sum)
    pv0 = min(prop_vmin, prop_vmax)
    pv1 = max(prop_vmin, prop_vmax)
    hm = np.ma.masked_where(heatmap <= 0, heatmap)

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="#000000")

    clip_low = bool(np.any(np.logical_and(heatmap > 0, heatmap < pv0)))
    clip_high = bool(np.any(heatmap > pv1))
    if clip_low and clip_high:
        extend = "both"
    elif clip_low:
        extend = "min"
    elif clip_high:
        extend = "max"
    else:
        extend = "neither"

    fig, ax = plt.subplots(figsize=(11.0, 7.5), dpi=dpi)
    ax.set_facecolor("#000000")
    pcm = ax.pcolormesh(
        np.arange(num_layers + 1),
        bin_edges,
        hm.T,
        shading="flat",
        cmap=cmap,
        norm=LogNorm(vmin=pv0, vmax=pv1),
    )
    ax.set_yscale("log")
    ax.set_ylim(a_min, a_max_plot)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("layer", fontsize=12, fontweight="600")
    ax.set_ylabel("Activation value", fontsize=12, fontweight="600")
    ax.set_title(title, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10, fontweight="600")
    plt.setp(ax.get_yticklabels(), fontsize=10, fontweight="600")

    cbar = fig.colorbar(pcm, ax=ax, extend=extend)
    cbar.set_label("proportion", fontsize=11, fontweight="600")
    ticks = _proportion_cbar_ticks(pv0, pv1)
    cbar.set_ticks(ticks)
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10))
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.ax.tick_params(labelsize=10)
    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_edgecolor("face")

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _t_tag(tv: float) -> str:
    """Filename fragment for timestep, e.g. 0.5 -> 0p5000."""
    return f"{tv:.4f}".replace(".", "p")


def run_forward_sit(model, x, t, y):
    return model(x, t, y)


def run_forward_repa(model, x, t, y):
    out = model(x, t, y)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def _collect_from_forwards(
    model: torch.nn.Module,
    backend: str,
    device: torch.device,
    storage: List[Optional[torch.Tensor]],
    num_layers: int,
    hidden_size: int,
    latent_size: int,
    num_samples: int,
    batch_size: int,
    use_amp: bool,
    t_min: float,
    t_max: float,
    t_fixed: Optional[float],
    num_classes: int,
    bin_edges: Optional[np.ndarray],
    num_bins: int,
    count_sum: Optional[np.ndarray],
    dim_sum: Optional[np.ndarray],
    dim_count: Optional[np.ndarray],
    max_only: bool,
    print_shapes_once: bool = False,
) -> Tuple[float, int]:
    """
    Run forwards over random latents. If max_only, only return max |h| seen.
    Otherwise fill count_sum (histogram), dim_sum, dim_count.
    """
    max_abs = 0.0
    remaining = num_samples
    n_forwards = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, model.in_channels, latent_size, latent_size, device=device)
        if t_fixed is not None:
            t = torch.full((bs,), t_fixed, device=device, dtype=x.dtype)
        else:
            t = torch.rand(bs, device=device) * (t_max - t_min) + t_min
        y = torch.randint(0, num_classes, (bs,), device=device)

        storage[:] = [None] * num_layers
        if use_amp:
            with torch.cuda.amp.autocast():
                if backend == "sit":
                    _ = run_forward_sit(model, x, t, y)
                else:
                    _ = run_forward_repa(model, x, t, y)
        else:
            if backend == "sit":
                _ = run_forward_sit(model, x, t, y)
            else:
                _ = run_forward_repa(model, x, t, y)

        for li in range(num_layers):
            h = storage[li]
            if h is None:
                raise RuntimeError(f"Hook did not capture layer {li}")
            if print_shapes_once and n_forwards == 0:
                print(f"[shape] layer {li + 1}/{num_layers} hidden state: {tuple(h.shape)}")
                if li == num_layers - 1:
                    print(f"[info] num transformer layers measured: {num_layers}")
                    print(f"[info] hidden size D = {hidden_size}")
            hf = h.float()
            abs_h = hf.abs()
            max_abs = max(max_abs, float(abs_h.max().cpu()))
            if not max_only:
                assert bin_edges is not None and count_sum is not None
                flat = abs_h.detach().cpu().numpy().ravel()
                count_sum[li] += histogram_abs_counts(flat, bin_edges)
                if dim_sum is not None and dim_count is not None:
                    dim_sum[li] += abs_h.sum(dim=(0, 1)).cpu().numpy()
                    dim_count[li] += float(abs_h.shape[0] * abs_h.shape[1])

        remaining -= bs
        n_forwards += 1

    return max_abs, n_forwards


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot per-layer |activation| distribution heatmap for SiT-XL/2 (SiT or REPA checkpoint)."
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for PNG/PDF.")
    p.add_argument(
        "--backend",
        type=str,
        choices=["sit", "repa"],
        default="sit",
        help="sit: /workspace/SiT style SiT; repa: REPA/models/sit.py SiT.",
    )
    p.add_argument("--sit-root", type=str, default="/workspace/SiT")
    p.add_argument("--repa-root", type=str, default="/workspace/REPA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="Fewer samples and bins for quick debug.")
    p.add_argument("--num-samples", type=int, default=None, help="Total samples (overrides fast default).")
    p.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per forward.")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--resolution", type=int, default=256, help="Image resolution (latent = //8).")
    p.add_argument("--model", type=str, default="SiT-XL/2")
    # SiT-only
    p.add_argument(
        "--learn-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SiT backbone learn_sigma (256x256 checkpoints usually True).",
    )
    # REPA-only
    p.add_argument("--repa-args-json", type=str, default=None, help="Path to REPA train args.json.")
    p.add_argument("--encoder-depth", type=int, default=None)
    p.add_argument("--projector-embed-dims", type=str, default=None, help="Comma-separated z_dims, e.g. 768 or 768,1024")
    p.add_argument("--use-cfg", action=argparse.BooleanOptionalAction, default=None, help="REPA LabelEmbedder CFG table.")
    p.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False, help="REPA legacy decoder_blocks keys.")
    # Time sampling: full [0, T] with T=1 by default (massive-activations / DiT-style “T down to 0” range)
    p.add_argument(
        "--t-min",
        type=float,
        default=0.0,
        help="Lower bound for continuous timestep t (default 0 = clean end of range).",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=1.0,
        help="Upper bound for continuous timestep t (default 1 = T, noisy end). Sampled uniformly in [t_min, t_max].",
    )
    # Histogram
    p.add_argument("--a-min", type=float, default=1e-4, help="Min activation magnitude for histogram / y-axis.")
    p.add_argument(
        "--a-max",
        type=float,
        default=None,
        help="Max |h| for histogram / y-axis. Omit for auto: max observed × (1 + --a-max-margin).",
    )
    p.add_argument(
        "--a-max-margin",
        type=float,
        default=0.05,
        help="When --a-max is omitted: upper range = (1 + this) × max |h| (default 0.05 = 5%% headroom).",
    )
    p.add_argument("--num-bins", type=int, default=None)
    p.add_argument(
        "--prop-vmin",
        type=float,
        default=1e-4,
        help="Lower bound of proportion color scale (LogNorm), default 10⁻⁴.",
    )
    p.add_argument(
        "--prop-vmax",
        type=float,
        default=1e-1,
        help="Upper bound of proportion color scale (LogNorm), default 10⁻¹.",
    )
    p.add_argument(
        "--per-noise-heatmaps",
        action="store_true",
        help="Also save one heatmap per fixed timestep t (see --noise-ts). Uses same |h| bins as the mixed-t run.",
    )
    p.add_argument(
        "--noise-ts",
        type=str,
        default=None,
        help="Comma-separated t in [0,1] for per-noise plots. Default: 12 points linspace(0,1).",
    )
    p.add_argument(
        "--samples-per-noise",
        type=int,
        default=None,
        help="Microbatch quota per fixed-t run (default: same as --num-samples).",
    )
    # Plot
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--top-k-dims", type=int, default=10)
    p.add_argument("--print-tree", action="store_true", help="Print module names resembling blocks.")
    args = p.parse_args()
    if args.t_min >= args.t_max:
        raise SystemExit(f"--t-min ({args.t_min}) must be < --t-max ({args.t_max})")

    if args.fast:
        num_samples = args.num_samples if args.num_samples is not None else 24
        num_bins = args.num_bins if args.num_bins is not None else 64
    else:
        num_samples = args.num_samples if args.num_samples is not None else 512
        num_bins = args.num_bins if args.num_bins is not None else 192

    _add_repo_paths(args.sit_root, args.repa_root, args.backend)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = args.resolution // 8

    print(
        f"[info] timestep sampling: t ~ Uniform({args.t_min}, {args.t_max}) "
        f"(i.i.d. per micro-batch; full T→0 range when t_min=0, t_max=1)"
    )

    # --- Build model ---
    if args.backend == "sit":
        from download import find_model
        from models import SiT_models

        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=args.learn_sigma,
        ).to(device)
        state_dict = find_model(args.ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict strict=False missing={missing!r} unexpected={unexpected!r}")
    else:
        from models.sit import SiT_models
        from utils import load_legacy_checkpoints

        j = load_repa_args_json(args.repa_args_json)
        enc_d = args.encoder_depth if args.encoder_depth is not None else int(j.get("encoder_depth", 8))
        fused = args.fused_attn if args.fused_attn is not None else bool(j.get("fused_attn", True))
        qk = args.qk_norm if args.qk_norm is not None else bool(j.get("qk_norm", False))
        res = int(j.get("resolution", args.resolution))
        if args.resolution != res and args.repa_args_json:
            print(f"[info] using resolution={res} from args json (CLI was {args.resolution})")
        latent_size = res // 8

        cfg_prob = float(j.get("cfg_prob", 0.1))
        use_cfg = args.use_cfg if args.use_cfg is not None else (cfg_prob > 0)

        if args.projector_embed_dims:
            z_dims = [int(x) for x in args.projector_embed_dims.split(",") if x.strip()]
        else:
            # Reasonable default matching many REPA runs (single projector)
            z_dims = [768]
            print("[warn] --projector-embed-dims not set; defaulting z_dims=[768]. Set explicitly if load fails.")

        block_kwargs = {"fused_attn": fused, "qk_norm": qk}
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=int(j.get("num_classes", args.num_classes)),
            use_cfg=use_cfg,
            z_dims=z_dims,
            encoder_depth=enc_d,
            **block_kwargs,
        ).to(device)

        sd = _load_checkpoint_raw(args.ckpt, map_location="cpu")
        sd = _strip_module_prefix(sd)
        if args.legacy:
            sd = load_legacy_checkpoints(sd, enc_d)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    if args.print_tree:
        print_transformer_candidates(model)

    blocks = model.blocks
    num_layers = len(blocks)
    handles, storage = register_block_output_hooks(blocks)

    hidden_size = int(model.pos_embed.shape[-1])
    use_amp = device.type == "cuda"

    try:
        if args.a_max is None:
            max_abs, n_pass1 = _collect_from_forwards(
                model,
                args.backend,
                device,
                storage,
                num_layers,
                hidden_size,
                latent_size,
                num_samples,
                args.batch_size,
                use_amp,
                args.t_min,
                args.t_max,
                None,
                args.num_classes,
                None,
                num_bins,
                None,
                None,
                None,
                max_only=True,
                print_shapes_once=False,
            )
            a_max_plot = max(max_abs * (1.0 + args.a_max_margin), args.a_min * 10.0)
            print(
                f"[info] max |h| observed: {max_abs:.6g} -> a_max={a_max_plot:.6g} "
                f"(+{100.0 * args.a_max_margin:.1f}% margin over max)"
            )
        else:
            a_max_plot = float(args.a_max)
            n_pass1 = 0
            print(f"[info] fixed a_max={a_max_plot:.6g} (single forward pass over samples)")

        bin_edges = build_log_bins(args.a_min, a_max_plot, num_bins)
        count_sum = np.zeros((num_layers, num_bins), dtype=np.float64)
        dim_sum = np.zeros((num_layers, hidden_size), dtype=np.float64)
        dim_count = np.zeros(num_layers, dtype=np.float64)

        _, n_pass2 = _collect_from_forwards(
            model,
            args.backend,
            device,
            storage,
            num_layers,
            hidden_size,
            latent_size,
            num_samples,
            args.batch_size,
            use_amp,
            args.t_min,
            args.t_max,
            None,
            args.num_classes,
            bin_edges,
            num_bins,
            count_sum,
            dim_sum,
            dim_count,
            max_only=False,
            print_shapes_once=True,
        )
    finally:
        for h in handles:
            h.remove()

    n_forwards = n_pass1 + n_pass2
    print(f"[info] total forward passes: {n_forwards}  (target samples ≈ {num_samples})")

    dim_mean = dim_sum / np.maximum(dim_count[:, None], 1.0)
    score_mean = dim_mean.mean(axis=0)
    score_max = dim_mean.max(axis=0)
    topk_mean = np.argsort(-score_mean)[: args.top_k_dims]
    topk_max = np.argsort(-score_max)[: args.top_k_dims]

    print("\n--- Top hidden dimensions by mean |h| averaged over layers ---")
    for rank, d in enumerate(topk_mean, start=1):
        print(f"  rank {rank}: dim={int(d):4d}  score_mean_layers={score_mean[d]:.6g}")
    print("\n--- Top hidden dimensions by max over layers of mean |h| ---")
    for rank, d in enumerate(topk_max, start=1):
        print(f"  rank {rank}: dim={int(d):4d}  max_layer_mean={score_max[d]:.6g}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = "activations_sit_xl2"
    png_path = outdir / f"{prefix}.png"
    pdf_path = outdir / f"{prefix}.pdf"

    import matplotlib

    matplotlib.pyplot.switch_backend("Agg")
    plt.style.use("dark_background")

    save_activation_heatmap_figure(
        count_sum,
        bin_edges,
        num_layers,
        args.a_min,
        a_max_plot,
        png_path,
        pdf_path,
        "(a) Activations along layers — SiT-XL/2 (t ~ Uniform[t_min, t_max])",
        args.dpi,
        prop_vmin=args.prop_vmin,
        prop_vmax=args.prop_vmax,
    )

    print(f"\n[saved] {png_path}")
    print(f"[saved] {pdf_path}")

    if args.per_noise_heatmaps:
        noise_ts = parse_noise_ts(args.noise_ts) if args.noise_ts else default_noise_ts()
        sn = args.samples_per_noise if args.samples_per_noise is not None else num_samples
        print(
            f"[info] per-noise heatmaps: {len(noise_ts)} levels, t ∈ {noise_ts[0]:.4g} … {noise_ts[-1]:.4g}, "
            f"~{sn} samples each (same |h| bins as mixed plot)"
        )
        sub = outdir / "per_noise"
        sub.mkdir(parents=True, exist_ok=True)
        handles3, storage3 = register_block_output_hooks(blocks)
        n_extra = 0
        try:
            for tv in noise_ts:
                cs = np.zeros((num_layers, num_bins), dtype=np.float64)
                _, nf = _collect_from_forwards(
                    model,
                    args.backend,
                    device,
                    storage3,
                    num_layers,
                    hidden_size,
                    latent_size,
                    sn,
                    args.batch_size,
                    use_amp,
                    args.t_min,
                    args.t_max,
                    float(tv),
                    args.num_classes,
                    bin_edges,
                    num_bins,
                    cs,
                    None,
                    None,
                    max_only=False,
                    print_shapes_once=False,
                )
                n_extra += nf
                tag = _t_tag(float(tv))
                p_png = sub / f"{prefix}_t_{tag}.png"
                p_pdf = sub / f"{prefix}_t_{tag}.pdf"
                save_activation_heatmap_figure(
                    cs,
                    bin_edges,
                    num_layers,
                    args.a_min,
                    a_max_plot,
                    p_png,
                    p_pdf,
                    f"Activations along layers — fixed t = {tv:.4f}",
                    args.dpi,
                    prop_vmin=args.prop_vmin,
                    prop_vmax=args.prop_vmax,
                )
                print(f"[saved] {p_png}")
        finally:
            for h in handles3:
                h.remove()
        print(f"[info] extra forward passes (per-noise): {n_extra}")


if __name__ == "__main__":
    main()
