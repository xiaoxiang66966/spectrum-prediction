"""Adjacency construction (paper Section III-B, eqs. 1-2) and loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def correlation_matrix(data: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Band-to-band similarity S in [0, 1], with the diagonal zeroed.

    `data` is (T, N). The paper treats S as bounded in [0, 1], so the absolute
    value of the (Pearson/Spearman) correlation is used.
    """
    data = np.asarray(data, dtype=np.float64)
    if method == "pearson":
        s = np.corrcoef(data, rowvar=False)          # (N, N)
    elif method == "spearman":
        s, _ = spearmanr(data, axis=0)               # (N, N)
    else:
        raise ValueError(f"unknown correlation method: {method}")

    s = np.nan_to_num(s)
    s = np.abs(s)
    np.fill_diagonal(s, 0.0)
    return s.astype(np.float32)


def topk_binarize(s: np.ndarray, k: int) -> np.ndarray:
    """Binary sparse adjacency from eq. (2): keep the top-k neighbours per node.

    Self-loops are excluded here; the GCN normalisation re-adds them.
    """
    s = np.asarray(s, dtype=np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    n = s.shape[0]
    k = min(max(int(k), 1), n - 1)
    a = np.zeros_like(s)
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    for i in range(n):
        a[i, idx[i]] = 1.0
    return a.astype(np.float32)


def load_adjacency_csv(path) -> np.ndarray:
    """Load a provided N x N adjacency CSV (e.g. v2 0.5-spearman).

    The provided files contain self-loops on the diagonal; they are removed
    because `normalize_adjacency` re-adds them for the GCN.
    """
    a = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    np.fill_diagonal(a, 0.0)
    return a


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)
