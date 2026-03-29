from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from sit_metrics.extract import run_forward_collect
from sit_metrics.model_loader import load_model


@pytest.mark.parametrize(
    ("backend", "ckpt"),
    [
        ("sit", ROOT / "SiT/pretrained_models/SiT-XL-2-256x256.pt"),
        ("repa", ROOT / "REPA/pretrained_models/last.pt"),
    ],
)
def test_checkpoint_smoke_hidden_shapes(backend: str, ckpt: Path) -> None:
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not found: {ckpt}")

    device = torch.device("cpu")
    model = load_model(
        backend=backend,
        ckpt=str(ckpt),
        device=device,
        sit_root=str(ROOT / "SiT"),
        repa_root=str(ROOT / "REPA"),
        resolution=256,
        num_classes=1000,
        learn_sigma=True,
    )
    model.eval()

    x = torch.randn(1, model.in_channels, 32, 32, device=device)
    t = torch.full((1,), 0.5, device=device, dtype=x.dtype)
    y = torch.zeros(1, device=device, dtype=torch.long)
    layers = [0, len(model.blocks) - 1]
    out = run_forward_collect(model, backend, x, t, y, layers, compute_A=False)

    assert set(out.H.keys()) == set(layers)
    for H in out.H.values():
        assert tuple(H.shape[:2]) == (1, 256)
