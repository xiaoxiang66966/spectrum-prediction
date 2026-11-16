"""Spectrum CSV loading, normalisation and sliding-window dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_csv(path) -> np.ndarray:
    """Load a spectrum CSV as a (T, N) float32 array.

    The first row of these CSVs is an empty header, so it is skipped.
    """
    df = pd.read_csv(path, header=None, skiprows=1)
    return df.to_numpy(dtype=np.float32)


def fit_zscore(data: np.ndarray):
    """Per-band z-score statistics. `data` is (T, N); returns 1-D (N,) vectors."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_zscore(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std


def inverse_zscore(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data * std + mean


class SpectrumDataset(Dataset):
    """Sliding-window multivariate time-series dataset."""

    def __init__(self, data: np.ndarray, window: int, horizon: int):
        data = np.asarray(data, dtype=np.float32)
        assert data.ndim == 2, "expected a (T, N) array"
        self.x = torch.from_numpy(data)
        self.window = window
        self.horizon = horizon
        self.n_samples = len(data) - window - horizon + 1
        assert self.n_samples > 0, "series is shorter than window + horizon"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.x[idx : idx + self.window]                            # (W, N)
        y = self.x[idx + self.window : idx + self.window + self.horizon]  # (H, N)
        return x, y


def make_dataloader(
    data: np.ndarray,
    window: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = SpectrumDataset(data, window, horizon)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
