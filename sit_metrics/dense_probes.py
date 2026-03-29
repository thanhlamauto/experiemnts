"""Token-level linear probes and dense pseudo-mask metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _TokenProbe(nn.Module):
    def __init__(self, feat_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(feat_dim, affine=False),
            nn.Linear(feat_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.enable_grad()
def fit_multiclass_token_probe(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device | str,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
) -> nn.Module:
    device = torch.device(device)
    Ztr = torch.as_tensor(Z_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.long)

    model = _TokenProbe(int(Ztr.shape[1]), int(num_classes)).to(device)
    dataset = TensorDataset(Ztr, ytr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(len(dataset), 1)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    criterion = nn.CrossEntropyLoss()

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


@torch.enable_grad()
def fit_binary_token_probe(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    device: torch.device | str,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
) -> nn.Module:
    device = torch.device(device)
    Ztr = torch.as_tensor(Z_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.float32)

    model = _TokenProbe(int(Ztr.shape[1]), 1).to(device)
    dataset = TensorDataset(Ztr, ytr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(len(dataset), 1)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    pos = float(ytr.sum().item())
    neg = float(ytr.numel() - pos)
    pos_weight = None
    if pos > 0.0 and neg > 0.0:
        pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def update_confusion_matrix(
    confusion: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    num_classes = int(confusion.shape[0])
    flat = (target.to(torch.int64) * num_classes + pred.to(torch.int64)).cpu()
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes)


def mean_iou_from_confusion(confusion: torch.Tensor) -> float:
    confusion = confusion.to(torch.float64)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    if not bool(valid.any()):
        return float("nan")
    iou = tp[valid] / denom[valid]
    return float(iou.mean().item())


def update_binary_f1_counts(
    stats: Dict[str, int],
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    pred_b = pred.to(torch.bool)
    target_b = target.to(torch.bool)
    stats["tp"] += int((pred_b & target_b).sum().item())
    stats["fp"] += int((pred_b & ~target_b).sum().item())
    stats["fn"] += int((~pred_b & target_b).sum().item())


def f1_from_counts(stats: Dict[str, int]) -> float:
    tp = float(stats.get("tp", 0))
    fp = float(stats.get("fp", 0))
    fn = float(stats.get("fn", 0))
    denom = 2.0 * tp + fp + fn
    if denom <= 0.0:
        return float("nan")
    return (2.0 * tp) / denom


@torch.no_grad()
def objectness_iou_from_mask(
    H: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    ignore_index: int = -1,
) -> Tuple[torch.Tensor, int]:
    Hn = torch.nn.functional.normalize(H.float(), dim=-1)
    mask = object_mask.to(device=H.device, dtype=torch.int64)
    scores = []
    count = 0

    for b in range(Hn.shape[0]):
        valid = mask[b] != int(ignore_index)
        fg = (mask[b] == 1) & valid
        if not bool(valid.any()) or not bool(fg.any()):
            continue

        anchor = Hn[b, fg].mean(dim=0, keepdim=True)
        sim = torch.matmul(Hn[b], anchor.t()).squeeze(-1)
        sim_valid = sim[valid]
        k = int(fg.sum().item())
        k = max(1, min(k, int(valid.sum().item())))

        pred_valid = torch.zeros_like(sim_valid, dtype=torch.bool)
        top_idx = torch.topk(sim_valid, k=k, dim=0).indices
        pred_valid[top_idx] = True

        pred = torch.zeros_like(valid, dtype=torch.bool)
        pred[valid] = pred_valid

        inter = (pred & fg).sum().to(torch.float32)
        union = (pred | fg).sum().to(torch.float32)
        if float(union.item()) <= 0.0:
            continue
        scores.append(inter / union)
        count += 1

    if not scores:
        return torch.tensor(float("nan"), device=H.device), 0
    return torch.stack(scores).mean(), count
