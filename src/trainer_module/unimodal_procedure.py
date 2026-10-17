"""Unimodal training: stage-0 encoder pretraining for MoE-Health and the single-modality reference runs."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.data_module.dataset import SplitData, TaskData
from src.model_module.unimodal import MODALITY_INDEX, UnimodalModel
from src.trainer_module.registry import MethodOutput, register_method
from src.trainer_module.trainer import fit, predict, val_auroc
from src.utils.config import ModelConfig, TrainConfig, from_node

__all__ = ["run_unimodal", "train_unimodal"]


def _has(split: SplitData, modality: str) -> torch.Tensor:
    return split.tensors["mask"][:, MODALITY_INDEX[modality]] > 0.5


def train_unimodal(
    cfg: DictConfig, data: TaskData, modality: str, device: torch.device, gen: torch.Generator
) -> tuple[UnimodalModel, float]:
    """Train on the samples that have ``modality``; returns the model and its best val AUROC."""
    mcfg, tcfg = from_node(ModelConfig, cfg.model), from_node(TrainConfig, cfg.train)
    model = UnimodalModel(mcfg, data.ts_dim, data.static_cardinalities, data.note_dim, data.cxr_dim,
                          modality, data.stats_dim).to(device)
    train, val = data.train.subset(_has(data.train, modality)), data.val.subset(_has(data.val, modality))
    bce = nn.BCEWithLogitsLoss()
    res = fit(model, list(model.parameters()), lambda b: bce(model(b)["logits"], b["label"]), train,
              lambda: val_auroc(model, model, val), tcfg, tcfg.max_epochs, tcfg.patience, gen,
              tag=f"unimodal {modality}")
    return model, res.best_score


@register_method("unimodal")
def run_unimodal(cfg: DictConfig, data: TaskData, device: torch.device, gen: torch.Generator) -> MethodOutput:
    """Evaluated on the samples that have the modality (n is reported with the metrics)."""
    modality = str(cfg.model.modality)
    model, best = train_unimodal(cfg, data, modality, device, gen)
    val = predict(model, model, data.val.subset(_has(data.val, modality)))
    test = predict(model, model, data.test.subset(_has(data.test, modality)))
    return MethodOutput(val, test, {"modality": modality, "val_auroc_best": best})
