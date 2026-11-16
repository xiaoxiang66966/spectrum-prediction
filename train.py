"""Training / evaluation loops and regression metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    use_estimator: bool,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, use_estimator=use_estimator)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    """Return {loss, rmse, mae, r2} on the original PSD scale."""
    model.eval()
    preds, trues = [], []
    total, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, use_estimator=False)
        total += criterion(pred, y).item() * x.shape[0]
        n += x.shape[0]
        preds.append(pred.cpu())
        trues.append(y.cpu())

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)

    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    pred_o = pred * std_t + mean_t
    true_o = true * std_t + mean_t

    rmse = float((pred_o - true_o).pow(2).mean().sqrt())
    mae = float((pred_o - true_o).abs().mean())
    ss_res = float((true_o - pred_o).pow(2).sum())
    ss_tot = float((true_o - true_o.mean()).pow(2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {"loss": total / n, "rmse": rmse, "mae": mae, "r2": r2}
