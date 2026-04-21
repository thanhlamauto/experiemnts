"""Radial band compatibility experiment for layer-0 / input frequency probing.

What this experiment measures
-----------------------------
For each probed model layer ``i``, this module asks:

    "Which radial frequency bands of layer 0 / the input image are most
    compatible with layer ``i`` on the original image?"

The primary comparison is:

    h_i(x)  <->  h_0^(k)(x)

where:
- ``x`` is the original input image
- ``h_i(x)`` is the representation from layer ``i`` on the original input
- ``h_0^(k)(x)`` is the band-pass filtered layer-0 representation

By default, layer 0 is the input image itself. The experiment does *not*
primarily test whether a layer stays similar to itself under filtered input.

Expected inputs
---------------
- A PyTorch model
- An iterable / dataloader that yields batches
- ``layers_to_probe`` identifying the deeper layers to record
- ``input_extractor(batch, device)`` that returns a ``[B, C, H, W]`` tensor
- ``model_forward_fn(model, images, batch)`` that runs the model on the
  original images so hooks can capture layer outputs

What gets written
-----------------
Inside ``output_dir`` the experiment writes:

- ``raw_metric_scores.csv``
- ``layer_band_summary.csv``
- ``inferred_intervals.csv``
- ``heatmap_overall.png``
- ``heatmap_agreement.png``
- ``interval_summary.png``
- ``layer_<layer_name>_profile.png`` for every probed layer

Minimal Kaggle-style usage
--------------------------
```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from kaggle_sit_protocol.radial_band_compat import (
    RadialBandExperimentConfig,
    run_radial_band_compatibility_experiment,
)


images = torch.randn(8, 3, 256, 256)
labels = torch.zeros(8, dtype=torch.long)
loader = DataLoader(TensorDataset(images, labels), batch_size=2)

model = ...  # your own PyTorch model
layers_to_probe = ["blocks.0", "blocks.5", "blocks.11"]

config = RadialBandExperimentConfig()

results = run_radial_band_compatibility_experiment(
    model=model,
    dataloader=loader,
    layers_to_probe=layers_to_probe,
    output_dir=Path("/kaggle/working/radial_band_experiment"),
    config=config,
    input_extractor=lambda batch, device: batch[0].to(device),
    model_forward_fn=lambda model, images, batch: model(images),
)

print(results["raw_metric_scores_csv"])
print(results["layer_band_summary_csv"])
print(results["inferred_intervals_csv"])
```
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class RadialBandExperimentConfig:
    """Configuration for the radial band compatibility experiment."""

    num_bands: int = 16
    mask_tau: float = 0.02
    mask_eps: float = 1e-8
    hist_bins: int = 64
    hist_quantile_low: float = 0.01
    hist_quantile_high: float = 0.99
    hist_standardized_clip: float = 4.0
    metric_support_threshold: float = 0.7
    agreement_threshold: float = 0.5
    overall_quantile: float = 0.7
    fixed_overall_threshold: float | None = None
    cosine_weight: float = 0.4
    var_weight: float = 0.3
    hist_weight: float = 0.3
    device: str | None = None
    max_batches: int | None = None
    max_images: int | None = None
    eval_mode: bool = True
    progress_every: int = 10
    plot_dpi: int = 180


@dataclass(frozen=True)
class RadialBand:
    """Equal-area radial band definition."""

    band_id: int
    start_r: float
    end_r: float
    center_r: float


@dataclass
class TensorDescriptor:
    """Compact shared descriptor for arbitrary tensors."""

    mean: float
    logvar: float
    histogram: np.ndarray
    vector: np.ndarray


def build_equal_area_bands(num_bands: int) -> list[RadialBand]:
    """Create equal-area radial bands with edges r_k = sqrt(k / K)."""

    bands: list[RadialBand] = []
    for band_idx in range(num_bands):
        start_r = float(np.sqrt(band_idx / num_bands))
        end_r = float(np.sqrt((band_idx + 1) / num_bands))
        center_r = float(np.sqrt((band_idx + 0.5) / num_bands))
        bands.append(
            RadialBand(
                band_id=band_idx,
                start_r=start_r,
                end_r=end_r,
                center_r=center_r,
            )
        )
    return bands


def build_radial_coordinate_map(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a normalized radial coordinate map in [0, 1]."""

    yy = torch.arange(height, device=device, dtype=dtype) - (height / 2.0)
    xx = torch.arange(width, device=device, dtype=dtype) - (width / 2.0)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(grid_x.square() + grid_y.square())
    radius = radius / torch.clamp(radius.max(), min=torch.finfo(dtype).eps)
    return radius


def build_soft_band_mask(
    radius_map: Tensor,
    band: RadialBand,
    *,
    tau: float,
    eps: float,
) -> Tensor:
    """Create a soft radial band-pass mask with sigmoid edges."""

    tau_value = max(float(tau), eps)
    start = torch.sigmoid((radius_map - band.start_r) / tau_value)
    end = torch.sigmoid((radius_map - band.end_r) / tau_value)
    mask = (start - end).clamp_min(0.0)
    # L2-normalize the mask so band energy is more comparable across bands.
    norm = torch.sqrt(mask.square().mean()).clamp_min(eps)
    return mask / norm


def build_soft_band_masks(
    height: int,
    width: int,
    bands: Sequence[RadialBand],
    *,
    tau: float,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, Tensor]:
    """Build one soft mask per radial band."""

    radius_map = build_radial_coordinate_map(height, width, device=device, dtype=dtype)
    return {
        band.band_id: build_soft_band_mask(radius_map, band, tau=tau, eps=eps)
        for band in bands
    }


def apply_bandpass_fft(images: Tensor, mask: Tensor) -> Tensor:
    """Apply a shared radial band-pass mask to all image channels."""

    if images.ndim != 4:
        raise ValueError(f"Expected images shaped [B, C, H, W], got {tuple(images.shape)}")

    spectrum = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    masked = spectrum * mask.unsqueeze(0).unsqueeze(0).to(spectrum.dtype)
    reconstructed = torch.fft.ifft2(torch.fft.ifftshift(masked, dim=(-2, -1)), dim=(-2, -1)).real
    return reconstructed


def _coerce_to_tensor(output: Any) -> Tensor:
    """Best-effort extraction of a tensor from a module output."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            try:
                return _coerce_to_tensor(item)
            except TypeError:
                continue
    if isinstance(output, dict):
        for item in output.values():
            try:
                return _coerce_to_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Could not extract a tensor from output of type {type(output)!r}")


class LayerHookRecorder:
    """Record selected layer outputs for one forward pass."""

    def __init__(self, layer_map: OrderedDict[str, torch.nn.Module]) -> None:
        self.layer_map = layer_map
        self.activations: dict[str, Tensor] = {}
        self.handles: list[Any] = []

    def clear(self) -> None:
        self.activations.clear()

    def register(self) -> None:
        self.close()
        for layer_name, module in self.layer_map.items():
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self.handles.append(handle)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _coerce_to_tensor(output).detach()
            self.activations[layer_name] = tensor

        return hook


def resolve_layers_to_probe(
    model: torch.nn.Module,
    layers_to_probe: Sequence[str] | Mapping[str, torch.nn.Module],
) -> OrderedDict[str, torch.nn.Module]:
    """Resolve user-provided layer identifiers to modules."""

    if isinstance(layers_to_probe, Mapping):
        return OrderedDict((str(name), module) for name, module in layers_to_probe.items())

    named_modules = dict(model.named_modules())
    resolved: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for layer_name in layers_to_probe:
        if layer_name not in named_modules:
            raise KeyError(f"Layer {layer_name!r} was not found in model.named_modules().")
        resolved[str(layer_name)] = named_modules[layer_name]
    return resolved


def register_layer_hooks(
    model: torch.nn.Module,
    layers_to_probe: Sequence[str] | Mapping[str, torch.nn.Module],
) -> LayerHookRecorder:
    """Create and register hooks for the requested layers."""

    layer_map = resolve_layers_to_probe(model, layers_to_probe)
    recorder = LayerHookRecorder(layer_map)
    recorder.register()
    return recorder


def tensor_to_descriptor(
    tensor: Tensor,
    *,
    hist_bins: int,
    q_low: float,
    q_high: float,
    hist_standardized_clip: float,
    eps: float,
) -> TensorDescriptor:
    """Convert any tensor into the shared descriptor g(T)."""

    flat = tensor.detach().reshape(-1).float()
    if flat.numel() == 0:
        raise ValueError("Descriptor source tensor is empty.")

    mean = float(flat.mean().item())
    var = float(flat.var(unbiased=False).item())
    logvar = float(np.log(var + eps))

    if flat.numel() == 1:
        histogram = np.zeros(hist_bins, dtype=np.float32)
        histogram[0] = 1.0
    else:
        std = float(np.sqrt(var + eps))
        standardized = (flat - mean) / max(std, eps)
        low = max(-float(hist_standardized_clip), float(torch.quantile(standardized, q_low).item()))
        high = min(float(hist_standardized_clip), float(torch.quantile(standardized, q_high).item()))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + eps:
            histogram = np.zeros(hist_bins, dtype=np.float32)
            histogram[0] = 1.0
        else:
            clipped = standardized.clamp(min=low, max=high)
            hist = torch.histc(clipped, bins=hist_bins, min=low, max=high)
            hist = hist / hist.sum().clamp_min(eps)
            histogram = hist.cpu().numpy().astype(np.float32)

    vector = np.concatenate(
        [
            np.asarray([mean, logvar], dtype=np.float32),
            histogram.astype(np.float32),
        ]
    )
    return TensorDescriptor(mean=mean, logvar=logvar, histogram=histogram, vector=vector)


def cosine_similarity(desc_a: TensorDescriptor, desc_b: TensorDescriptor, eps: float) -> float:
    """Cosine similarity on the shared descriptor vectors."""

    vec_a = torch.from_numpy(desc_a.vector)
    vec_b = torch.from_numpy(desc_b.vector)
    sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1, eps=eps)
    return float(sim.item())


def log_variance_similarity(desc_a: TensorDescriptor, desc_b: TensorDescriptor) -> float:
    """Similarity based on closeness in log-variance."""

    return float(np.exp(-abs(desc_a.logvar - desc_b.logvar)))


def _js_distance(hist_a: np.ndarray, hist_b: np.ndarray, eps: float) -> float:
    """Jensen-Shannon distance for normalized histograms."""

    p = hist_a.astype(np.float64) + eps
    q = hist_b.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js_div = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(np.sqrt(max(js_div, 0.0)))


def histogram_similarity(desc_a: TensorDescriptor, desc_b: TensorDescriptor, eps: float) -> float:
    """Histogram similarity defined as 1 - JS distance."""

    return float(1.0 - _js_distance(desc_a.histogram, desc_b.histogram, eps=eps))


def _minmax_or_neutral(values: pd.Series, neutral_value: float = 0.5) -> pd.Series:
    """Min-max scale to [0, 1], or return a neutral score if constant."""

    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-12:
        return pd.Series(np.full(len(values), neutral_value, dtype=np.float32), index=values.index)
    return (values - min_value) / (max_value - min_value)


def normalize_metrics_and_score(
    raw_df: pd.DataFrame,
    *,
    metric_support_threshold: float,
    cosine_weight: float,
    var_weight: float,
    hist_weight: float,
) -> pd.DataFrame:
    """Normalize metrics per (image_id, layer) across bands and build scores."""

    df = raw_df.copy()
    group_cols = ["image_id", "layer"]
    df["cosine_norm"] = df.groupby(group_cols)["cosine"].transform(_minmax_or_neutral)
    df["var_sim_norm"] = df.groupby(group_cols)["var_sim"].transform(_minmax_or_neutral)
    df["hist_sim_norm"] = df.groupby(group_cols)["hist_sim"].transform(_minmax_or_neutral)

    df["overall_score"] = (
        cosine_weight * df["cosine_norm"]
        + var_weight * df["var_sim_norm"]
        + hist_weight * df["hist_sim_norm"]
    )

    support_matrix = np.column_stack(
        [
            (df["cosine_norm"] >= metric_support_threshold).to_numpy(dtype=np.int32),
            (df["var_sim_norm"] >= metric_support_threshold).to_numpy(dtype=np.int32),
            (df["hist_sim_norm"] >= metric_support_threshold).to_numpy(dtype=np.int32),
        ]
    )
    df["metric_support_count"] = support_matrix.sum(axis=1)
    return df


def aggregate_layer_band_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image scores into per-layer, per-band summaries."""

    summary = (
        scored_df.groupby(["layer", "band_id", "start_r", "end_r", "center_r"], as_index=False)
        .agg(
            mean_cosine=("cosine", "mean"),
            mean_var_sim=("var_sim", "mean"),
            mean_hist_sim=("hist_sim", "mean"),
            std_overall=("overall_score", "std"),
            agreement_rate=("metric_support_count", lambda s: float(np.mean(np.asarray(s) >= 2))),
        )
        .sort_values(["layer", "band_id"])
        .reset_index(drop=True)
    )
    summary["std_overall"] = summary["std_overall"].fillna(0.0)
    return summary


def attach_summary_overall_scores(
    summary_df: pd.DataFrame,
    *,
    cosine_weight: float,
    var_weight: float,
    hist_weight: float,
) -> pd.DataFrame:
    """Normalize aggregated mean metrics across bands within each layer."""

    df = summary_df.copy()
    group_cols = ["layer"]
    df["mean_cosine_norm"] = df.groupby(group_cols)["mean_cosine"].transform(_minmax_or_neutral)
    df["mean_var_sim_norm"] = df.groupby(group_cols)["mean_var_sim"].transform(_minmax_or_neutral)
    df["mean_hist_sim_norm"] = df.groupby(group_cols)["mean_hist_sim"].transform(_minmax_or_neutral)
    df["mean_overall"] = (
        cosine_weight * df["mean_cosine_norm"]
        + var_weight * df["mean_var_sim_norm"]
        + hist_weight * df["mean_hist_sim_norm"]
    )
    return df


def infer_intervals(
    summary_df: pd.DataFrame,
    *,
    overall_quantile: float,
    fixed_overall_threshold: float | None,
    agreement_threshold: float,
) -> pd.DataFrame:
    """Infer active radial intervals for each layer from aggregated band profiles."""

    rows: list[dict[str, Any]] = []
    for layer_name, layer_df in summary_df.groupby("layer", sort=False):
        layer_df = layer_df.sort_values("band_id").reset_index(drop=True)
        if fixed_overall_threshold is None:
            overall_threshold = float(np.quantile(layer_df["mean_overall"], overall_quantile))
        else:
            overall_threshold = float(fixed_overall_threshold)

        active = (
            (layer_df["mean_overall"] >= overall_threshold)
            & (layer_df["agreement_rate"] >= agreement_threshold)
        ).to_numpy(dtype=bool)

        interval_start: int | None = None
        interval_id = 0
        for idx, is_active in enumerate(active):
            if is_active and interval_start is None:
                interval_start = idx
            at_end = idx == len(active) - 1
            should_close = interval_start is not None and ((not is_active) or at_end)
            if not should_close:
                continue

            interval_stop = idx if is_active and at_end else idx - 1
            interval_df = layer_df.iloc[interval_start : interval_stop + 1]
            peak_local_idx = int(interval_df["mean_overall"].to_numpy().argmax())
            peak_row = interval_df.iloc[peak_local_idx]

            rows.append(
                {
                    "layer": layer_name,
                    "interval_id": interval_id,
                    "start_r": float(interval_df.iloc[0]["start_r"]),
                    "end_r": float(interval_df.iloc[-1]["end_r"]),
                    "peak_r": float(peak_row["center_r"]),
                    "peak_score": float(peak_row["mean_overall"]),
                    "mean_agreement": float(interval_df["agreement_rate"].mean()),
                    "num_bands": int(len(interval_df)),
                }
            )
            interval_id += 1
            interval_start = None

    return pd.DataFrame(rows)


def _make_heatmap_matrix(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    layer_order: Sequence[str],
    band_order: Sequence[int],
) -> np.ndarray:
    """Build a layer x band matrix for plotting."""

    pivot = summary_df.pivot(index="layer", columns="band_id", values=value_column)
    pivot = pivot.reindex(index=list(layer_order), columns=list(band_order))
    return pivot.to_numpy(dtype=np.float32)


def _format_band_labels(summary_df: pd.DataFrame, band_order: Sequence[int]) -> list[str]:
    """Create readable x-axis labels from band centers."""

    centers = (
        summary_df[["band_id", "center_r"]]
        .drop_duplicates()
        .sort_values("band_id")
        .set_index("band_id")
        .loc[list(band_order), "center_r"]
    )
    return [f"{center:.2f}" for center in centers]


def plot_heatmap(
    matrix: np.ndarray,
    *,
    layer_order: Sequence[str],
    x_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a simple matplotlib heatmap."""

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(x_labels)), max(4, 0.4 * len(layer_order))))
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Radial band center")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks(range(len(layer_order)))
    ax.set_yticklabels(layer_order)
    fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def sanitize_layer_name(layer_name: str) -> str:
    """Turn a layer label into a safe filename fragment."""

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(layer_name))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "layer"


def plot_layer_profiles(
    summary_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot one profile per layer with agreement and shaded intervals."""

    for layer_name, layer_df in summary_df.groupby("layer", sort=False):
        layer_df = layer_df.sort_values("band_id").reset_index(drop=True)
        interval_rows = intervals_df[intervals_df["layer"] == layer_name] if not intervals_df.empty else pd.DataFrame()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(layer_df["center_r"], layer_df["mean_overall"], marker="o", label="mean overall", color="tab:blue")
        ax.plot(
            layer_df["center_r"],
            layer_df["agreement_rate"],
            marker="s",
            linestyle="--",
            label="agreement rate",
            color="tab:orange",
        )

        for _, row in interval_rows.iterrows():
            ax.axvspan(float(row["start_r"]), float(row["end_r"]), color="tab:green", alpha=0.15)

        ax.set_title(f"Layer {layer_name} compatibility profile")
        ax.set_xlabel("Radial band center")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"layer_{sanitize_layer_name(str(layer_name))}_profile.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_interval_summary(
    layer_order: Sequence[str],
    intervals_df: pd.DataFrame,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    """Summarize inferred intervals as horizontal segments per layer."""

    fig_height = max(4, 0.45 * len(layer_order))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = {layer_name: idx for idx, layer_name in enumerate(layer_order)}
    if not intervals_df.empty:
        for _, row in intervals_df.iterrows():
            y = y_positions[row["layer"]]
            ax.hlines(y, float(row["start_r"]), float(row["end_r"]), linewidth=5, color="tab:blue", alpha=0.8)
            ax.plot(float(row["peak_r"]), y, marker="o", color="tab:red")

    ax.set_title("Inferred radial compatibility intervals")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Layer")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(range(len(layer_order)))
    ax.set_yticklabels(layer_order)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _default_image_ids_from_batch(batch: Any, batch_size: int, offset: int) -> list[str]:
    """Best-effort extraction of image ids from a batch."""

    if isinstance(batch, Mapping):
        for key in ("image_id", "image_ids", "id", "ids", "path", "paths"):
            if key in batch:
                values = batch[key]
                if isinstance(values, (list, tuple)):
                    return [str(value) for value in values]
                if torch.is_tensor(values):
                    return [str(value.item()) for value in values.reshape(-1)]
    return [f"image_{offset + idx:06d}" for idx in range(batch_size)]


def _prepare_device(model: torch.nn.Module, device_hint: str | None) -> torch.device:
    """Resolve the device used for the experiment."""

    if device_hint is not None:
        return torch.device(device_hint)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_radial_band_compatibility_experiment(
    *,
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    layers_to_probe: Sequence[str] | Mapping[str, torch.nn.Module],
    output_dir: str | Path,
    config: RadialBandExperimentConfig | None = None,
    input_extractor: Callable[[Any, torch.device], Tensor],
    model_forward_fn: Callable[[torch.nn.Module, Tensor, Any], Any],
    image_id_extractor: Callable[[Any, int, int], list[str]] | None = None,
    layer0_source_fn: Callable[[Tensor, dict[str, Tensor], Any], Tensor] | None = None,
) -> dict[str, Any]:
    """Run the full radial band compatibility experiment."""

    cfg = config or RadialBandExperimentConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = _prepare_device(model, cfg.device)
    model = model.to(device)

    layer_map = resolve_layers_to_probe(model, layers_to_probe)
    recorder = LayerHookRecorder(layer_map)
    recorder.register()

    previous_training_state = model.training
    if cfg.eval_mode:
        model.eval()

    bands = build_equal_area_bands(cfg.num_bands)
    raw_rows: list[dict[str, Any]] = []
    processed_images = 0
    batch_count = 0
    masks: dict[int, Tensor] | None = None

    try:
        for batch_index, batch in enumerate(dataloader):
            if cfg.max_batches is not None and batch_index >= cfg.max_batches:
                break

            images = input_extractor(batch, device)
            if images.ndim != 4:
                raise ValueError(
                    "input_extractor(...) must return a [B, C, H, W] tensor, "
                    f"got {tuple(images.shape)}"
                )
            batch_size = int(images.shape[0])
            if batch_size == 0:
                continue

            current_offset = processed_images
            if image_id_extractor is None:
                image_ids = _default_image_ids_from_batch(batch, batch_size, current_offset)
            else:
                image_ids = image_id_extractor(batch, current_offset, batch_size)
            if len(image_ids) != batch_size:
                raise ValueError("image_id_extractor(...) must return one id per image in the batch.")

            recorder.clear()
            with torch.inference_mode():
                _ = model_forward_fn(model, images, batch)

            layer_outputs = {name: tensor.detach() for name, tensor in recorder.activations.items()}
            missing_layers = [name for name in layer_map if name not in layer_outputs]
            if missing_layers:
                raise RuntimeError(f"Hooks did not capture outputs for layers: {missing_layers}")

            layer0_source = images if layer0_source_fn is None else layer0_source_fn(images, layer_outputs, batch)
            if layer0_source.ndim != 4:
                raise ValueError(
                    "layer0_source_fn(...) must return a [B, C, H, W] tensor, "
                    f"got {tuple(layer0_source.shape)}"
                )
            if masks is None:
                masks = build_soft_band_masks(
                    int(layer0_source.shape[-2]),
                    int(layer0_source.shape[-1]),
                    bands,
                    tau=cfg.mask_tau,
                    eps=cfg.mask_eps,
                    device=device,
                    dtype=layer0_source.dtype,
                )

            band_descriptors_by_band: dict[int, list[TensorDescriptor]] = {}
            for band in bands:
                bandpass = apply_bandpass_fft(layer0_source, masks[band.band_id])
                descriptors = [
                    tensor_to_descriptor(
                        bandpass[sample_idx],
                        hist_bins=cfg.hist_bins,
                        q_low=cfg.hist_quantile_low,
                        q_high=cfg.hist_quantile_high,
                        hist_standardized_clip=cfg.hist_standardized_clip,
                        eps=cfg.mask_eps,
                    )
                    for sample_idx in range(batch_size)
                ]
                band_descriptors_by_band[band.band_id] = descriptors

            layer_descriptors_by_name: dict[str, list[TensorDescriptor]] = {}
            for layer_name, activation in layer_outputs.items():
                if activation.shape[0] != batch_size:
                    raise ValueError(
                        f"Layer {layer_name!r} produced batch dimension {activation.shape[0]}, "
                        f"expected {batch_size}."
                    )
                layer_descriptors_by_name[layer_name] = [
                    tensor_to_descriptor(
                        activation[sample_idx],
                        hist_bins=cfg.hist_bins,
                        q_low=cfg.hist_quantile_low,
                        q_high=cfg.hist_quantile_high,
                        hist_standardized_clip=cfg.hist_standardized_clip,
                        eps=cfg.mask_eps,
                    )
                    for sample_idx in range(batch_size)
                ]

            for sample_idx, image_id in enumerate(image_ids):
                for layer_name in layer_map:
                    layer_desc = layer_descriptors_by_name[layer_name][sample_idx]
                    for band in bands:
                        band_desc = band_descriptors_by_band[band.band_id][sample_idx]
                        raw_rows.append(
                            {
                                "image_id": str(image_id),
                                "layer": str(layer_name),
                                "band_id": int(band.band_id),
                                "start_r": float(band.start_r),
                                "end_r": float(band.end_r),
                                "center_r": float(band.center_r),
                                "cosine": cosine_similarity(layer_desc, band_desc, eps=cfg.mask_eps),
                                "var_sim": log_variance_similarity(layer_desc, band_desc),
                                "hist_sim": histogram_similarity(layer_desc, band_desc, eps=cfg.mask_eps),
                            }
                        )

            processed_images += batch_size
            batch_count += 1

            if cfg.progress_every > 0 and batch_count % cfg.progress_every == 0:
                print(f"[radial-band-compat] processed {processed_images} images across {batch_count} batches")

            if cfg.max_images is not None and processed_images >= cfg.max_images:
                break

    finally:
        recorder.close()
        if cfg.eval_mode and previous_training_state:
            model.train()

    if not raw_rows:
        raise RuntimeError("No rows were produced. Check the dataloader and extractor callbacks.")

    raw_df = pd.DataFrame(raw_rows)
    scored_df = normalize_metrics_and_score(
        raw_df,
        metric_support_threshold=cfg.metric_support_threshold,
        cosine_weight=cfg.cosine_weight,
        var_weight=cfg.var_weight,
        hist_weight=cfg.hist_weight,
    )
    summary_df = aggregate_layer_band_summary(scored_df)
    summary_df = attach_summary_overall_scores(
        summary_df,
        cosine_weight=cfg.cosine_weight,
        var_weight=cfg.var_weight,
        hist_weight=cfg.hist_weight,
    )
    intervals_df = infer_intervals(
        summary_df,
        overall_quantile=cfg.overall_quantile,
        fixed_overall_threshold=cfg.fixed_overall_threshold,
        agreement_threshold=cfg.agreement_threshold,
    )

    raw_metric_scores_path = output_path / "raw_metric_scores.csv"
    layer_band_summary_path = output_path / "layer_band_summary.csv"
    inferred_intervals_path = output_path / "inferred_intervals.csv"

    raw_export = scored_df[
        [
            "image_id",
            "layer",
            "band_id",
            "start_r",
            "end_r",
            "center_r",
            "cosine",
            "var_sim",
            "hist_sim",
            "overall_score",
            "metric_support_count",
        ]
    ].copy()
    raw_export.to_csv(raw_metric_scores_path, index=False)
    summary_df.to_csv(layer_band_summary_path, index=False)
    intervals_df.to_csv(inferred_intervals_path, index=False)

    layer_order = list(layer_map.keys())
    band_order = list(range(cfg.num_bands))
    band_labels = _format_band_labels(summary_df, band_order)

    overall_matrix = _make_heatmap_matrix(
        summary_df,
        value_column="mean_overall",
        layer_order=layer_order,
        band_order=band_order,
    )
    agreement_matrix = _make_heatmap_matrix(
        summary_df,
        value_column="agreement_rate",
        layer_order=layer_order,
        band_order=band_order,
    )

    heatmap_overall_path = output_path / "heatmap_overall.png"
    heatmap_agreement_path = output_path / "heatmap_agreement.png"
    interval_summary_path = output_path / "interval_summary.png"

    plot_heatmap(
        overall_matrix,
        layer_order=layer_order,
        x_labels=band_labels,
        title="Mean overall compatibility score",
        colorbar_label="mean_overall",
        output_path=heatmap_overall_path,
        dpi=cfg.plot_dpi,
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        agreement_matrix,
        layer_order=layer_order,
        x_labels=band_labels,
        title="Agreement rate",
        colorbar_label="agreement_rate",
        output_path=heatmap_agreement_path,
        dpi=cfg.plot_dpi,
        vmin=0.0,
        vmax=1.0,
    )
    plot_layer_profiles(summary_df, intervals_df, output_dir=output_path, dpi=cfg.plot_dpi)
    plot_interval_summary(layer_order, intervals_df, output_path=interval_summary_path, dpi=cfg.plot_dpi)

    return {
        "raw_metric_scores": raw_export,
        "layer_band_summary": summary_df,
        "inferred_intervals": intervals_df,
        "raw_metric_scores_csv": str(raw_metric_scores_path),
        "layer_band_summary_csv": str(layer_band_summary_path),
        "inferred_intervals_csv": str(inferred_intervals_path),
        "heatmap_overall_png": str(heatmap_overall_path),
        "heatmap_agreement_png": str(heatmap_agreement_path),
        "interval_summary_png": str(interval_summary_path),
        "output_dir": str(output_path),
    }


__all__ = [
    "RadialBand",
    "RadialBandExperimentConfig",
    "TensorDescriptor",
    "apply_bandpass_fft",
    "attach_summary_overall_scores",
    "build_equal_area_bands",
    "build_radial_coordinate_map",
    "build_soft_band_mask",
    "build_soft_band_masks",
    "infer_intervals",
    "plot_interval_summary",
    "plot_layer_profiles",
    "plot_heatmap",
    "register_layer_hooks",
    "resolve_layers_to_probe",
    "run_radial_band_compatibility_experiment",
    "tensor_to_descriptor",
]
