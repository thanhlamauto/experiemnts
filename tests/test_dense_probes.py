from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from sit_metrics.dense_probes import (
    fit_binary_token_probe,
    fit_multiclass_token_probe,
    mean_iou_from_confusion,
    objectness_iou_from_mask,
    update_binary_f1_counts,
    update_confusion_matrix,
)


def test_multiclass_token_probe_smoke() -> None:
    Z = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.1],
            [0.0, 1.0],
            [0.1, 0.8],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])
    model = fit_multiclass_token_probe(Z, y, num_classes=2, device="cpu", epochs=20, batch_size=4, lr=1e-2, seed=0)

    with torch.no_grad():
        pred = model(Z).argmax(dim=-1)
    conf = torch.zeros(2, 2, dtype=torch.int64)
    update_confusion_matrix(conf, pred, y)
    assert mean_iou_from_confusion(conf) >= 0.5


def test_binary_token_probe_smoke() -> None:
    Z = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])
    model = fit_binary_token_probe(Z, y, device="cpu", epochs=10, batch_size=4, lr=1e-2, seed=0)

    with torch.no_grad():
        pred = model(Z).squeeze(-1) > 0.0
    stats = {"tp": 0, "fp": 0, "fn": 0}
    update_binary_f1_counts(stats, pred, y > 0)
    assert stats["tp"] >= 1


def test_token_probes_train_inside_no_grad_context() -> None:
    Z = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    y_multi = torch.tensor([0, 0, 1, 1])
    y_binary = torch.tensor([0, 0, 1, 1])

    with torch.no_grad():
        multiclass = fit_multiclass_token_probe(
            Z,
            y_multi,
            num_classes=2,
            device="cpu",
            epochs=5,
            batch_size=4,
            lr=1e-2,
            seed=0,
        )
        binary = fit_binary_token_probe(
            Z,
            y_binary,
            device="cpu",
            epochs=5,
            batch_size=4,
            lr=1e-2,
            seed=0,
        )

    with torch.no_grad():
        multiclass_logits = multiclass(Z)
        binary_logits = binary(Z)

    assert multiclass_logits.shape == (4, 2)
    assert binary_logits.shape == (4, 1)


def test_objectness_iou_from_mask_is_high_for_separable_tokens() -> None:
    H = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        ]
    )
    mask = torch.tensor([[1, 1, 0, 0]])
    score, count = objectness_iou_from_mask(H, mask, ignore_index=-1)
    assert count == 1
    assert float(score) > 0.9
