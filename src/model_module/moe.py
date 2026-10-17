"""Mixture-of-experts layer: gating (Eq. 11), top-k routing (Eq. 12), balance loss (Eqs. 15-16)."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ALL_COMBOS", "Expert", "GatingNetwork", "MoELayer", "balance_loss", "combo_code"]

# Non-empty modality subsets in (E, T, I) bit order: E, T, I, ET, EI, TI, ETI (Fig. 2 expert pool).
ALL_COMBOS: list[tuple[int, int, int]] = sorted(
    ((e, t, i) for e, t, i in itertools.product((0, 1), repeat=3) if e or t or i),
    key=lambda c: (sum(c), [-b for b in c]),
)


def combo_code(combo: tuple[int, int, int]) -> int:
    """Bit code used in the data: E=1, T=2, I=4."""
    return combo[0] * 1 + combo[1] * 2 + combo[2] * 4


class Expert(nn.Module):
    """Lightweight MLP expert."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    """g = Softmax(W_g ReLU(W_R R + b_R) + b_g)  (Eq. 11)."""

    def __init__(self, in_dim: int, hidden: int, n_experts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_experts))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(r), dim=-1)


class MoELayer(nn.Module):
    """Experts + gate. ``gating='uniform'`` averages all experts (w/o dynamic gating ablation)."""

    def __init__(
        self, in_dim: int, out_dim: int, n_experts: int, expert_hidden: int, gate_hidden: int,
        top_k: int, gating: str, dropout: float,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(Expert(in_dim, expert_hidden, out_dim, dropout) for _ in range(n_experts))
        self.gate = GatingNetwork(in_dim, gate_hidden, n_experts)
        self.top_k = min(top_k, n_experts)
        self.gating = gating

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outs = torch.stack([e(r) for e in self.experts], dim=1)  # [B, K, out]
        n, k_all = outs.shape[0], outs.shape[1]
        if self.gating == "uniform":
            gates = torch.full((n, k_all), 1.0 / k_all, device=r.device)
            topk_idx = torch.arange(k_all, device=r.device).expand(n, -1)
            topk_w = gates
        else:
            gates = self.gate(r)
            topk_w, topk_idx = torch.topk(gates, self.top_k, dim=-1)
            denom = topk_w.sum(dim=-1, keepdim=True)
            # Renormalise (Eq. 12). For top-1 this is exactly 1 and would cut the task-loss gradient to
            # the gate; a detached denominator keeps the forward value and the gradient.
            topk_w = topk_w / (denom.detach() if self.top_k == 1 else denom)
        sel = torch.gather(outs, 1, topk_idx.unsqueeze(-1).expand(-1, -1, outs.shape[-1]))
        y = (sel * topk_w.unsqueeze(-1)).sum(dim=1)
        return y, {"gates": gates, "topk_idx": topk_idx, "topk_w": topk_w}


def balance_loss(gates: torch.Tensor, topk_idx: torch.Tensor, kind: str = "cv") -> torch.Tensor:
    """Load-balancing loss.

    cv:     CV(f * p) with f_k = #times expert k is in the top-k, p = mean gate (Eqs. 15-16)
    switch: K * sum_k (f_k / #assignments) * p_k (Switch Transformer form)
    none:   0
    Gradients flow through p only (f is a count).
    """
    if kind == "none":
        return gates.new_zeros(())
    n_exp = gates.shape[1]
    f = F.one_hot(topk_idx, n_exp).sum(dim=(0, 1)).to(gates.dtype)
    p = gates.mean(dim=0)
    if kind == "cv":
        x = f * p
        return x.std(unbiased=False) / (x.mean() + 1e-9)
    if kind == "switch":
        return n_exp * (f / f.sum() * p).sum()
    raise ValueError(f"unknown balance loss '{kind}'")
