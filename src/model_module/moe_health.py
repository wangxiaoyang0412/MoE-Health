"""MoE-Health (Wang & Yang, BCB 2025) as specified by Section 3.2 and Fig. 2.

Forward pass:
    tokens X [B, 3, d_h]  <- modality encoders; absent modality -> e_absent,m (Eq. 9)
    R = flatten(MHA(X + type_emb))          (Fig. 2 "Multi-Head Attention", 4 heads)
    repr experts:  H = LayerNorm(flatten(X) + sum_topk w_j E_j(R));  logit = Head(H)   (Fig. 2)
    logit experts: logit = sum_topk w_j E_j(R)                                       (Eq. 12 literal)

``mode`` selects the fusion path: "moe" (default), "dense" (stage-A warm-up with a single
generic expert) or "expert" (stage-B pretraining of one expert, index ``expert``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model_module.encoders import ModalityEncoders
from src.model_module.moe import Expert, MoELayer
from src.model_module.registry import register_model
from src.utils.config import ModelConfig

__all__ = ["MoEHealth"]

N_MODALITIES = 3


@register_model("moe_health")
class MoEHealth(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        ts_dim: int,
        static_cardinalities: list[int],
        note_dim: int,
        cxr_dim: int,
        combos: list[tuple[int, int, int]],
        stats_dim: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.combos = list(combos)
        d_h = cfg.d_h
        self.encoders = ModalityEncoders(ts_dim, static_cardinalities, note_dim, cxr_dim, d_h,
                                         cfg.lstm_hidden, cfg.lstm_layers, cfg.static_emb_dim, cfg.dropout,
                                         stats_dim)
        if cfg.missing == "learned":
            self.absent = nn.Parameter(torch.randn(N_MODALITIES, d_h) * 0.02)
        else:  # zero padding (w/o missing indicator ablation)
            self.register_buffer("absent", torch.zeros(N_MODALITIES, d_h))
        self.type_emb = nn.Parameter(torch.randn(N_MODALITIES, d_h) * 0.02)
        self.mha = nn.MultiheadAttention(d_h, cfg.n_heads, dropout=cfg.attn_dropout, batch_first=True)
        r_dim = N_MODALITIES * d_h
        out_dim = r_dim if cfg.expert_output == "repr" else 1
        self.moe = MoELayer(r_dim, out_dim, len(self.combos), cfg.expert_hidden, cfg.gate_hidden,
                            cfg.top_k, cfg.gating, cfg.dropout)
        self.dense = Expert(r_dim, cfg.expert_hidden, out_dim, cfg.dropout)  # stage-A generic expert
        self.norm = nn.LayerNorm(r_dim)
        self.head = nn.Sequential(nn.Linear(r_dim, cfg.head_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.head_hidden, 1))

    def tokens(self, batch: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        x = self.encoders(batch)
        m = mask.unsqueeze(-1)
        return m * x + (1.0 - m) * self.absent.unsqueeze(0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
        mode: str = "moe",
        expert: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return ``logits`` [B] and, for mode='moe', routing tensors."""
        x_flat, r = self.encode(batch, mask)
        aux: dict[str, torch.Tensor] = {}
        if mode == "moe":
            m, aux = self.moe(r)
        elif mode == "dense":
            m = self.dense(r)
        elif mode == "expert":
            m = self.moe.experts[expert](r)
        else:
            raise ValueError(f"unknown mode '{mode}'")
        return {"logits": self.readout(x_flat, m), **aux}

    def encode(
        self, batch: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened tokens X (residual path) and R = flatten(MHA(X)) (gate / expert input)."""
        mask = batch["mask"] if mask is None else mask
        x = self.tokens(batch, mask)
        q = self.attention_input(x, mask)
        attn, _ = self.mha(q, q, q, need_weights=False)
        return x.flatten(1), attn.flatten(1)

    def attention_input(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tokens + modality-type embedding on present tokens only (zero-padded absences stay zero)."""
        return x + mask.unsqueeze(-1) * self.type_emb.unsqueeze(0)

    def readout(self, x_flat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Add & Norm + prediction head (repr experts) or the expert logit itself (Eq. 12)."""
        if self.cfg.expert_output == "repr":
            return self.head(self.norm(x_flat + m)).squeeze(-1)
        return m.squeeze(-1)

    def shared_parameters(self) -> list[nn.Parameter]:
        """Everything except experts and gate (trained in stage A, frozen in stage B)."""
        moe_ids = {id(p) for p in self.moe.parameters()}
        return [p for p in self.parameters() if id(p) not in moe_ids]
