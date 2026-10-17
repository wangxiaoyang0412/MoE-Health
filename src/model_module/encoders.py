"""Modality-specific encoders (paper Section 3.2.1) producing d_h-dimensional tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EHREncoder", "FeatureEncoder", "ModalityEncoders"]


class EHREncoder(nn.Module):
    """BiLSTM over the 48 hourly bins + static embeddings -> Linear -> d_h (Eq. 6).

    Uses the final hidden state of each direction (h_n), so both directions have seen
    the full 48 h sequence. With ``stats_dim > 0`` an MLP over 48 h summary statistics is
    concatenated as a third branch.
    """

    def __init__(
        self,
        ts_dim: int,
        static_cardinalities: list[int],
        d_h: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        static_emb_dim: int = 16,
        dropout: float = 0.1,
        stats_dim: int = 0,
        stats_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            ts_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.cat_embs = nn.ModuleList(nn.Embedding(c, static_emb_dim) for c in static_cardinalities)
        self.age = nn.Linear(1, static_emb_dim)
        static_dim = static_emb_dim * (len(static_cardinalities) + 1)
        self.stats = (nn.Sequential(nn.Linear(stats_dim, stats_hidden), nn.GELU(), nn.Dropout(dropout))
                      if stats_dim else None)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * lstm_hidden + static_dim + (stats_hidden if stats_dim else 0), d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self, ts: torch.Tensor, static_num: torch.Tensor, static_cat: torch.Tensor,
        stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, (h_n, _) = self.lstm(ts)
        parts = [torch.cat([h_n[-2], h_n[-1]], dim=-1)]  # last layer: forward, backward
        parts += [emb(static_cat[:, i]) for i, emb in enumerate(self.cat_embs)] + [self.age(static_num)]
        if self.stats is not None:
            parts.append(self.stats(stats))
        return self.norm(self.out(self.drop(torch.cat(parts, dim=-1))))


class FeatureEncoder(nn.Module):
    """Projection of a frozen backbone feature ([CLS] or DenseNet GAP) to d_h (Eqs. 7-8)."""

    def __init__(self, in_dim: int, d_h: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, d_h), nn.LayerNorm(d_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoders(nn.Module):
    """EHR / notes / CXR encoders; returns tokens [B, 3, d_h] in order (E, T, I)."""

    def __init__(
        self,
        ts_dim: int,
        static_cardinalities: list[int],
        note_dim: int,
        cxr_dim: int,
        d_h: int,
        lstm_hidden: int,
        lstm_layers: int,
        static_emb_dim: int,
        dropout: float,
        stats_dim: int = 0,
    ) -> None:
        super().__init__()
        self.ehr = EHREncoder(ts_dim, static_cardinalities, d_h, lstm_hidden, lstm_layers, static_emb_dim, dropout,
                              stats_dim)
        self.note = FeatureEncoder(note_dim, d_h, dropout)
        self.cxr = FeatureEncoder(cxr_dim, d_h, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.ehr(batch["ehr_ts"], batch["static_num"], batch["static_cat"], batch.get("ehr_stats"))
        return torch.stack([e, self.note(batch["note"]), self.cxr(batch["cxr"])], dim=1)

    def encode_one(self, batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
        if modality == "ehr":
            return self.ehr(batch["ehr_ts"], batch["static_num"], batch["static_cat"], batch.get("ehr_stats"))
        return self.note(batch["note"]) if modality == "note" else self.cxr(batch["cxr"])
