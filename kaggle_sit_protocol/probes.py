from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeResult:
    variant: str
    target: str
    accuracy: float
    num_train: int
    num_test: int


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_linear_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
) -> tuple[float, np.ndarray]:
    train_x, test_x = _standardize(train_x, test_x)
    generator = torch.Generator()
    generator.manual_seed(seed)

    x_train = torch.from_numpy(train_x).float()
    y_train = torch.from_numpy(train_y).long()
    x_test = torch.from_numpy(test_x).float()
    y_test = torch.from_numpy(test_y).long()

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    model = nn.Linear(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
    accuracy = float((predictions == test_y).mean())
    return accuracy, predictions
