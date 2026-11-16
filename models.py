"""GSL-GCRNN model: GCN + graph structure estimator + GRU + MLP.

Mapping to the paper (Spectrum Prediction via Graph Structure Learning):
  eq. (3)-(4)  GCN            -> GCN / GCNLayer
  eq. (5)-(6)  graph estimator -> GraphEstimator
  eq. (7)      graph fusion    -> GSLGCRNN.forward (mu * A + (1-mu) * A_Q)
  eq. (8)      GRU            -> torch.nn.GRU
  eq. (9)      MSE loss       -> train.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(a: torch.Tensor) -> torch.Tensor:
    """Symmetric GCN normalisation D^{-1/2} (A + I) D^{-1/2}."""
    a = a + torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    deg = a.sum(dim=1).clamp(min=1e-8)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    return deg_inv_sqrt[:, None] * a * deg_inv_sqrt[None, :]


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # x: (..., N, in), a_norm: (N, N)
        return torch.matmul(a_norm, self.linear(x))


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(n_layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList(
            GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, a_norm)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class GraphEstimator(nn.Module):
    """Learn A_Q from node embeddings (eqs. 5-6).

    w_vu = [Z_v || Z_u] W + b, followed by a top-k softmax over each node's
    neighbours. The result is symmetrised before being fused with the static A.
    """

    def __init__(self, d: int, top_k: int):
        super().__init__()
        self.ffn = nn.Linear(2 * d, 1)
        self.top_k = top_k

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        left = z.unsqueeze(1).expand(n, n, -1)   # (N, N, d)
        right = z.unsqueeze(0).expand(n, n, -1)  # (N, N, d)
        w = self.ffn(torch.cat([left, right], dim=-1)).squeeze(-1)  # (N, N)

        # Exclude self-loops from the top-k candidate set.
        w = w.masked_fill(torch.eye(n, device=w.device, dtype=torch.bool), float("-inf"))

        k = min(self.top_k, n - 1)
        idx = w.topk(k, dim=1).indices
        mask = torch.full_like(w, float("-inf"))
        mask.scatter_(1, idx, 0.0)
        a_q = F.softmax(w + mask, dim=1)         # each row sums to 1
        a_q = (a_q + a_q.t()) / 2                # symmetrise for the undirected graph
        return a_q


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GSLGCRNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        horizon: int,
        static_a: torch.Tensor,
        gcn_hidden: int,
        gcn_layers: int,
        embed_dim: int,
        gru_hidden: int,
        top_k: int,
        mu: float,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.mu = mu
        self.register_buffer("static_a", static_a)
        self.register_buffer("a_cur", static_a.clone())

        self.gcn = GCN(in_dim=1, hidden_dim=gcn_hidden, out_dim=embed_dim, n_layers=gcn_layers)
        self.estimator = GraphEstimator(embed_dim, top_k)
        self.gru = nn.GRU(embed_dim, gru_hidden, num_layers=1, batch_first=True)
        self.mlp = MLP(gru_hidden, gru_hidden, horizon)

    def _gcn_embed(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_norm = normalize_adjacency(a)
        return self.gcn(x.unsqueeze(-1), a_norm)  # (B, T, N, d)

    def forward(self, x: torch.Tensor, use_estimator: bool = False) -> torch.Tensor:
        """x: (B, T, N); returns (B, horizon, N)."""
        b, t, n = x.shape

        a = self.a_cur
        z = self._gcn_embed(x, a)  # (B, T, N, d)

        if use_estimator and self.training:
            # One global embedding per node (averaged over batch and time),
            # used to re-estimate the graph structure on the fly.
            z_g = z.mean(dim=(0, 1))                 # (N, d)
            a_q = self.estimator(z_g)                # (N, N)
            a_new = self.mu * self.static_a + (1.0 - self.mu) * a_q
            self.a_cur.data.copy_(a_new.detach())    # graph for the next iteration
            z = self._gcn_embed(x, a_new)            # refine embeddings with A~

        # GRU over time; all nodes share the same recurrent weights.
        z = z.permute(0, 2, 1, 3).reshape(b * n, t, -1)  # (B*N, T, d)
        _, h = self.gru(z)                                # (1, B*N, H)
        h = h[-1].view(b, n, -1)                          # (B, N, H)
        out = self.mlp(h)                                 # (B, N, horizon)
        return out.transpose(1, 2)                        # (B, horizon, N)
