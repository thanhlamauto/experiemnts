from __future__ import annotations

from pathlib import Path

from .config import ProtocolConfig

SIT_MODEL_SPECS: dict[str, dict[str, int]] = {
    "SiT-XL/2": {"hidden_dim": 1152, "num_layers": 28},
    "SiT-XL/4": {"hidden_dim": 1152, "num_layers": 28},
    "SiT-XL/8": {"hidden_dim": 1152, "num_layers": 28},
    "SiT-L/2": {"hidden_dim": 1024, "num_layers": 24},
    "SiT-L/4": {"hidden_dim": 1024, "num_layers": 24},
    "SiT-L/8": {"hidden_dim": 1024, "num_layers": 24},
    "SiT-B/2": {"hidden_dim": 768, "num_layers": 12},
    "SiT-B/4": {"hidden_dim": 768, "num_layers": 12},
    "SiT-B/8": {"hidden_dim": 768, "num_layers": 12},
    "SiT-S/2": {"hidden_dim": 384, "num_layers": 12},
    "SiT-S/4": {"hidden_dim": 384, "num_layers": 12},
    "SiT-S/8": {"hidden_dim": 384, "num_layers": 12},
}


def get_sit_model_spec(model_name: str) -> dict[str, int]:
    try:
        return SIT_MODEL_SPECS[str(model_name)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Known models: {sorted(SIT_MODEL_SPECS)}"
        ) from exc


def apply_sit_model_spec(config: ProtocolConfig) -> None:
    spec = get_sit_model_spec(config.model_name)
    config.hidden_dim = int(spec["hidden_dim"])
    config.num_layers = int(spec["num_layers"])


def sit_model_slug(model_name: str) -> str:
    return str(model_name).replace("/", "-")


def candidate_sit_checkpoint_names(model_name: str, image_size: int) -> tuple[str, ...]:
    slug = sit_model_slug(model_name)
    return (
        f"{slug}-{int(image_size)}x{int(image_size)}.pt",
        f"{slug}-{int(image_size)}.pt",
    )


def iter_sit_checkpoint_candidates(model_name: str, image_size: int) -> tuple[Path, ...]:
    names = candidate_sit_checkpoint_names(model_name, image_size)
    candidates: list[Path] = []
    for name in names:
        candidates.append(Path(name))
        candidates.append(Path("pretrained_models") / name)
    return tuple(candidates)
