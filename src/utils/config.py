"""Frozen configuration dataclasses built from Hydra/OmegaConf nodes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

__all__ = ["ModelConfig", "StageConfig", "TrainConfig", "from_node"]

T = TypeVar("T")


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters shared by MoE-Health and the unimodal model."""

    name: str = "moe_health"
    d_h: int = 256
    n_heads: int = 4
    attn_dropout: float = 0.1
    lstm_hidden: int = 128
    lstm_layers: int = 1
    static_emb_dim: int = 16
    expert_hidden: int = 512
    gate_hidden: int = 256
    head_hidden: int = 128
    dropout: float = 0.1
    experts: str = "all7"  # all7 | observed4
    pretrain: str = "superset"  # superset | exact | generic | none
    encoder_pretrain: str = "none"  # none | unimodal (stage 0: each encoder trained on its own modality)
    top_k: int = 2
    gating: str = "learned"  # learned | uniform
    missing: str = "learned"  # learned | zeros
    expert_output: str = "repr"  # repr (Fig. 2) | logit (Eq. 12 literal)
    balance: str = "cv"  # cv (Eq. 16) | switch | none
    alpha: float = 0.01
    modality: str = "ehr"  # unimodal model only


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings (paper: AdamW, lr 1e-4, batch 32, <= 50 epochs, early stop on val AUROC)."""

    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    eval_batch_size: int = 1024
    gate_lr_mult: float = 1.0  # MoE-Health stage C: gate lr = lr * gate_lr_mult
    encoder_lr_mult: float = 1.0  # MoE-Health stage C: lr multiplier for stage-A-pretrained encoders/MHA
    evals_per_epoch: int = 1  # validation checks per epoch (early stopping granularity)
    modality_dropout: float = 0.0  # training-time drop prob for present notes / CXR
    ema_decay: float = 0.0  # > 0: validate / keep an exponential moving average of the weights


@dataclass(frozen=True)
class StageConfig:
    """Epoch budget of an auxiliary training stage (encoder warm-up, expert pretraining)."""

    max_epochs: int = 30
    patience: int = 5


def from_node(cls: type[T], node: DictConfig | dict[str, Any] | None) -> T:
    """Instantiate a frozen dataclass from a config node, ignoring unknown keys."""
    if node is None:
        return cls()
    if isinstance(node, DictConfig):
        raw = cast("dict[str, Any]", OmegaConf.to_container(node, resolve=True))
    else:
        raw = dict(node)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in raw.items() if k in names})
