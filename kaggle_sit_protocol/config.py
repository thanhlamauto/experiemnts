from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_TIME_GRID_INDICES = (0, 27, 55, 83, 110, 138, 166, 193, 221, 249)
DEFAULT_TSVD_RANKS = (16, 32, 64)
DEFAULT_PREVIEW_LAYERS = (1, 14, 28)
DEFAULT_PREVIEW_TIMESTEP_POSITIONS = (0, 4, 9)
DEFAULT_PCA_PANEL_LAYERS = (1, 3, 8, 13, 18, 23, 28)
DEFAULT_PCA_PANEL_TIMESTEP_POSITIONS = (6, 5, 3, 1)


@dataclass
class ProtocolConfig:
    seed: int = 0
    dataset_search_roots: tuple[str, ...] = ("/kaggle/input",)
    dataset_name_hint: str = "miniimagenet"
    explicit_dataset_root: str | None = None

    output_root: str = "outputs/kaggle_protocol"
    manifest_path: str = "outputs/kaggle_protocol/cache/manifest.parquet"
    image_size: int = 256
    latent_size: int = 32
    patch_grid_size: int = 16
    hidden_dim: int = 1152
    num_layers: int = 28
    model_name: str = "SiT-XL/2"
    num_classes: int = 1000
    checkpoint_path: str | None = None
    vae_model: str = "stabilityai/sd-vae-ft-ema"
    sit_root: str = "SiT"

    path_type: str = "Linear"
    prediction: str = "velocity"
    learn_sigma: bool = True
    time_grid_size: int = 250
    time_grid_indices: tuple[int, ...] = field(default_factory=lambda: DEFAULT_TIME_GRID_INDICES)

    main_images_per_class: int = 5
    control_images_per_class: int = 1
    preview_images: int = 16
    fallback_main_images: int = 256
    fallback_control_images: int = 64

    main_images_target: int = 500
    control_images_target: int = 100

    latent_batch_size: int = 32
    extraction_batch_size: int = 2
    probe_batch_size: int = 4096
    probe_epochs: int = 10
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4

    stats_eps: float = 1e-6
    spatial_norm_gamma: float = 1.0
    spatial_norm_eps: float = 1e-6
    preview_layers_1indexed: tuple[int, ...] = field(default_factory=lambda: DEFAULT_PREVIEW_LAYERS)
    preview_timestep_positions: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_PREVIEW_TIMESTEP_POSITIONS
    )
    pca_panel_layers_1indexed: tuple[int, ...] = field(default_factory=lambda: DEFAULT_PCA_PANEL_LAYERS)
    pca_panel_timestep_positions: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_PCA_PANEL_TIMESTEP_POSITIONS
    )
    tsvd_ranks: tuple[int, ...] = field(default_factory=lambda: DEFAULT_TSVD_RANKS)

    smoke_images: int = 8
    smoke_timestep_positions: tuple[int, ...] = (0, 9)

    overwrite: bool = False
    device: str = "cuda"

    @classmethod
    def from_kaggle_defaults(cls) -> "ProtocolConfig":
        kaggle_working = Path("/kaggle/working")
        output_root = (
            str(kaggle_working / "outputs" / "kaggle_protocol")
            if kaggle_working.exists()
            else "outputs/kaggle_protocol"
        )
        return cls(
            output_root=output_root,
            manifest_path=f"{output_root}/cache/manifest.parquet",
        )

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root)

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def scratch_dir(self) -> Path:
        return self.output_dir / "scratch"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / "analysis"

    @property
    def time_values(self) -> tuple[float, ...]:
        denom = max(int(self.time_grid_size) - 1, 1)
        return tuple(float(idx) / float(denom) for idx in self.time_grid_indices)

    @property
    def preview_layers_zeroindexed(self) -> tuple[int, ...]:
        return tuple(layer - 1 for layer in self.preview_layers_1indexed)

    @property
    def pca_panel_layers_zeroindexed(self) -> tuple[int, ...]:
        return tuple(layer - 1 for layer in self.pca_panel_layers_1indexed)

    @property
    def variant_names(self) -> tuple[str, ...]:
        base = ["mean-common", "mean-residual"]
        for rank in self.tsvd_ranks:
            base.append(f"tsvd-common-k{rank}")
            base.append(f"tsvd-residual-k{rank}")
        return tuple(base)

    def ensure_directories(self) -> None:
        manifest_path = Path(self.manifest_path)
        if not manifest_path.is_absolute() or not str(manifest_path).startswith(str(self.output_dir)):
            self.manifest_path = str(self.output_dir / "cache" / "manifest.parquet")
        for path in (self.output_dir, self.cache_dir, self.scratch_dir, self.reports_dir, self.analysis_dir):
            path.mkdir(parents=True, exist_ok=True)

    def iter_dataset_search_roots(self) -> Iterable[Path]:
        if self.explicit_dataset_root:
            yield Path(self.explicit_dataset_root)
        for root in self.dataset_search_roots:
            yield Path(root)
