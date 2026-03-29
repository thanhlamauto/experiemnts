"""Linear and k-NN probes on pooled hidden vectors."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def pool_features(H: torch.Tensor) -> torch.Tensor:
    """H: [B, N, D] -> [B, D]"""
    return H.mean(dim=1)


def fit_linear_probe(
    Z_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    Z_val: np.ndarray | torch.Tensor,
    y_val: np.ndarray | torch.Tensor,
    num_classes: int,
    device: torch.device | str = "cpu",
    epochs: int = 90,
    batch_size: int = 16384,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Paper-style linear probe:
    parameter-free BatchNorm1d + linear layer, trained with Adam and cosine LR.
    """
    device = torch.device(device)

    Ztr = torch.as_tensor(Z_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.long)
    Zva = torch.as_tensor(Z_val, dtype=torch.float32)
    yva = torch.as_tensor(y_val, dtype=torch.long)

    inferred_classes = int(max(ytr.max().item(), yva.max().item()) + 1)
    out_dim = max(int(num_classes), inferred_classes)
    feat_dim = int(Ztr.shape[1])

    model = nn.Sequential(
        nn.BatchNorm1d(feat_dim, affine=False),
        nn.Linear(feat_dim, out_dim, bias=True),
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(Ztr, ytr),
        batch_size=min(int(batch_size), len(Ztr)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(Zva.to(device, non_blocking=True))
        top1 = (logits.argmax(dim=1) == yva.to(device)).float().mean().item()
        k = min(5, logits.shape[1])
        idx = logits.topk(k=k, dim=1).indices
        top5 = (idx == yva.to(device).unsqueeze(1)).any(dim=1).float().mean().item()
    return float(top1), float(top5)


def knn_probe(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    k: int = 20,
) -> Tuple[float, float]:
    """k-NN with cosine similarity on L2-normalized vectors."""
    from sklearn.neighbors import KNeighborsClassifier

    # normalize
    Zt = Z_train / (np.linalg.norm(Z_train, axis=1, keepdims=True) + 1e-12)
    Zv = Z_val / (np.linalg.norm(Z_val, axis=1, keepdims=True) + 1e-12)
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(Zt, y_train)
    pred = knn.predict(Zv)
    top1 = (pred == y_val).mean()
    # recall@k as fraction of correct in top-k from kneighbors
    dist, idx = knn.kneighbors(Zv, n_neighbors=k, return_distance=True)
    rec = np.mean([y_val[i] in y_train[idx[i]] for i in range(len(y_val))])
    return float(top1), float(rec)
