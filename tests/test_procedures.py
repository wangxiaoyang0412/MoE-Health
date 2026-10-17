"""End-to-end smoke run of every registered method on a tiny synthetic task (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data_module.dataset import SplitData, TaskData
from src.trainer_module import METHOD_FACTORY, MethodFactory
from tests.test_model import CARDS, CXR, NOTE, TS_DIM, toy_batch


def toy_task(stats_dim: int = 0) -> TaskData:
    def split(n: int, seed: int) -> SplitData:
        b = toy_batch(n, seed)
        b["label"] = (b["ehr_ts"][:, -1, 0] > 0).float()  # learnable signal
        if stats_dim:
            b["ehr_stats"] = torch.randn(n, stats_dim)
        return SplitData(b)

    return TaskData(split(96, 0), split(48, 1), split(48, 2), ts_dim=TS_DIM, static_cardinalities=CARDS,
                    note_dim=NOTE, cxr_dim=CXR, stats_dim=stats_dim, info={"n_continuous": 4, "n_interval": 2})


def toy_cfg(method: str, **model_kw: object) -> OmegaConf:
    model = {"name": method, "method": method, "d_h": 16, "n_heads": 4, "lstm_hidden": 8, "static_emb_dim": 4,
             "expert_hidden": 32, "gate_hidden": 16, "head_hidden": 8, "modality": "ehr", **model_kw}
    return OmegaConf.create({
        "seed": 0, "model": model,
        "train": {"lr": 1e-3, "batch_size": 32, "max_epochs": 2, "patience": 2, "eval_batch_size": 64},
        "stage_a": {"max_epochs": 1, "patience": 1}, "stage_b": {"max_epochs": 1, "patience": 1},
    })


@pytest.mark.parametrize("method", ["moe_health", "unimodal"])
def test_method_runs(method: str) -> None:
    assert method in METHOD_FACTORY
    out = MethodFactory(method)(toy_cfg(method), toy_task(), torch.device("cpu"), torch.Generator().manual_seed(0))
    for pred in (out.val, out.test):
        assert set(pred) >= {"prob", "label", "row", "combo"}
        assert np.all((pred["prob"] >= 0) & (pred["prob"] <= 1))


@pytest.mark.parametrize("variant", [{"pretrain": "exact", "experts": "observed4"}, {"pretrain": "generic"},
                                     {"pretrain": "none", "top_k": 1}, {"gating": "uniform"},
                                     {"expert_output": "logit", "balance": "switch"}])
def test_moe_variants_run(variant: dict) -> None:
    out = MethodFactory("moe_health")(toy_cfg("moe_health", **variant), toy_task(), torch.device("cpu"),
                                      torch.Generator().manual_seed(0))
    assert "routing_test" in out.extra and "stages" in out.extra


def test_gate_warmup_and_gate_lr() -> None:
    cfg = toy_cfg("moe_health")
    cfg.gate_warmup = {"max_epochs": 1, "patience": 1}
    cfg.train.gate_lr_mult = 10.0
    out = MethodFactory("moe_health")(cfg, toy_task(), torch.device("cpu"), torch.Generator().manual_seed(0))
    assert "gate_warmup" in out.extra["stages"]


def test_training_recipe_options() -> None:
    cfg = toy_cfg("moe_health")
    cfg.train.update({"evals_per_epoch": 3, "modality_dropout": 0.3, "encoder_lr_mult": 0.1, "gate_lr_mult": 10.0})
    out = MethodFactory("moe_health")(cfg, toy_task(), torch.device("cpu"), torch.Generator().manual_seed(0))
    assert len(out.extra["stages"]["C"]["history"]) >= 3  # several validation checks per epoch


def test_modality_dropout_keeps_ehr() -> None:
    from src.trainer_module.trainer import modality_dropout

    mask = torch.ones(64, 3)
    assert torch.equal(modality_dropout(mask, 0.0), mask)
    dropped = modality_dropout(mask, 1.0)
    assert dropped[:, 0].eq(1).all() and dropped[:, 1:].eq(0).all()


def test_unimodal_encoder_pretraining_stage() -> None:
    cfg = toy_cfg("moe_health", encoder_pretrain="unimodal")
    cfg.train.encoder_lr_mult = 0.1
    out = MethodFactory("moe_health")(cfg, toy_task(), torch.device("cpu"), torch.Generator().manual_seed(0))
    assert set(out.extra["stages"]["0"]["val_auroc"]) == {"ehr", "note", "cxr"}


@pytest.mark.parametrize("method", ["moe_health", "unimodal"])
def test_methods_with_ehr_stats_branch(method: str) -> None:
    cfg = toy_cfg(method, encoder_pretrain="unimodal") if method == "moe_health" else toy_cfg(method)
    out = MethodFactory(method)(cfg, toy_task(stats_dim=9), torch.device("cpu"), torch.Generator().manual_seed(0))
    assert np.all(np.isfinite(out.test["prob"]))


def test_ema_weights_option() -> None:
    cfg = toy_cfg("moe_health")
    cfg.train.update({"ema_decay": 0.9, "evals_per_epoch": 2})
    out = MethodFactory("moe_health")(cfg, toy_task(), torch.device("cpu"), torch.Generator().manual_seed(0))
    assert np.all(np.isfinite(out.test["prob"]))
