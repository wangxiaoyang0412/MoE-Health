"""Unimodal model: one modality encoder connected directly to a prediction head (paper Section 4.2).

Used for stage-0 encoder pretraining of MoE-Health and for the single-modality reference runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model_module.encoders import EHREncoder, FeatureEncoder
from src.model_module.registry import register_model
from src.utils.config import ModelConfig

__all__ = ["MODALITY_INDEX", "UnimodalModel"]

MODALITY_INDEX = {"ehr": 0, "note": 1, "cxr": 2}


def _head(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


@register_model("unimodal")
class UnimodalModel(nn.Module):
    """Single-modality encoder connected directly to a prediction head."""

    def __init__(
        self, cfg: ModelConfig, ts_dim: int, static_cardinalities: list[int], note_dim: int, cxr_dim: int,
        modality: str, stats_dim: int = 0,
    ) -> None:
        super().__init__()
        self.modality = modality
        if modality == "ehr":
            self.encoder: nn.Module = EHREncoder(ts_dim, static_cardinalities, cfg.d_h, cfg.lstm_hidden,
                                                 cfg.lstm_layers, cfg.static_emb_dim, cfg.dropout, stats_dim)
        else:
            self.encoder = FeatureEncoder(note_dim if modality == "note" else cxr_dim, cfg.d_h, cfg.dropout)
        self.head = _head(cfg.d_h, cfg.head_hidden, cfg.dropout)

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.modality == "ehr":
            return self.encoder(batch["ehr_ts"], batch["static_num"], batch["static_cat"], batch.get("ehr_stats"))
        return self.encoder(batch[self.modality])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embed(batch)).squeeze(-1)}
