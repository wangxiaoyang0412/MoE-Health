"""Three-stage MoE-Health training.

A  encoder warm-up: encoders, missing/type embeddings, MHA, a single generic expert,
   Add & Norm and head, trained on all samples.
B  combination-specific expert pretraining (paper Section 3.2.2): shared parts frozen;
   expert j is trained on samples whose modalities include combination c_j ("superset",
   extra modalities masked) or equal it ("exact"). "generic" copies the stage-A expert into
   every slot; "none" keeps random init (w/o expert specialization ablations).
C  joint training of everything with BCE + alpha * L_balance, early stopping on val AUROC.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.data_module.dataset import SplitData, TaskData
from src.model_module.moe import ALL_COMBOS, balance_loss, combo_code
from src.model_module.moe_health import MoEHealth
from src.trainer_module.registry import MethodOutput, register_method
from src.trainer_module.trainer import Forward, fit, modality_dropout, predict, val_auroc, val_bce
from src.trainer_module.unimodal_procedure import train_unimodal
from src.utils.config import ModelConfig, StageConfig, TrainConfig, from_node

logger = logging.getLogger(__name__)

__all__ = ["expert_combos", "pretrain_subset", "routing_summary", "run_moe_health"]


def expert_combos(preset: str, train: SplitData) -> list[tuple[int, int, int]]:
    """'all7': every non-empty modality subset (Fig. 2); 'observed4': combos seen in training."""
    if preset == "all7":
        return list(ALL_COMBOS)
    seen = set(train.tensors["combo"].unique().tolist())
    return [c for c in ALL_COMBOS if combo_code(c) in seen]


def pretrain_subset(combo_codes: torch.Tensor, combo: tuple[int, int, int], policy: str) -> torch.Tensor:
    code = combo_code(combo)
    if policy == "exact":
        return combo_codes == code
    return (combo_codes & code) == code


def routing_summary(pred: dict[str, np.ndarray], combos: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Per data-combination: share of samples per top-1 expert and mean gate weights."""
    names = ["".join(L for L, b in zip("ETI", c, strict=True) if b) for c in combos]
    out: dict[str, Any] = {"experts": names}
    if "gates" not in pred:
        return out
    top1 = pred["gates"].argmax(axis=1)
    for code in np.unique(pred["combo"]):
        sel = pred["combo"] == code
        name = "".join(L for L, bit in zip("ETI", (1, 2, 4), strict=True) if code & bit)
        out[name] = {
            "n": int(sel.sum()),
            "top1_share": np.bincount(top1[sel], minlength=len(combos)).astype(float).tolist(),
            "mean_gate": pred["gates"][sel].mean(axis=0).round(4).tolist(),
        }
        out[name]["top1_share"] = [round(x / sel.sum(), 4) for x in out[name]["top1_share"]]
    return out


def _is_gate(name: str) -> bool:
    return name.startswith("moe.gate.")


def _is_encoder(name: str) -> bool:
    """Parts pretrained in stage A that feed the experts: encoders, missing/type embeddings, MHA."""
    return name.startswith(("encoders.", "mha.")) or name in ("absent", "type_emb")


@torch.no_grad()
def _frozen_cache(model: MoEHealth, split: SplitData, cmask: torch.Tensor, batch_size: int) -> SplitData:
    """Stage-B inputs computed once with the frozen shared parts (modalities outside the combo masked)."""
    model.eval()
    xs, rs = [], []
    for b in split.batches(batch_size, shuffle=False):
        x, r = model.encode(b, mask=b["mask"] * cmask)
        xs.append(x)
        rs.append(r)
    return SplitData({"x": torch.cat(xs), "r": torch.cat(rs), "label": split.tensors["label"]})


def _cached_expert_forward(model: MoEHealth, cache: dict[str, torch.Tensor], j: int) -> dict[str, torch.Tensor]:
    return {"logits": model.readout(cache["x"], model.moe.experts[j](cache["r"]))}


def _expert_bce(bce: nn.Module, forward: Forward, b: dict[str, torch.Tensor]) -> torch.Tensor:
    return bce(forward(b)["logits"], b["label"])


def _neg_val_bce(model: nn.Module, forward: Forward, split: SplitData) -> float:
    return -val_bce(model, forward, split)


@register_method("moe_health")
def run_moe_health(cfg: DictConfig, data: TaskData, device: torch.device, gen: torch.Generator) -> MethodOutput:
    mcfg = from_node(ModelConfig, cfg.model)
    tcfg = from_node(TrainConfig, cfg.train)
    combos = expert_combos(mcfg.experts, data.train)
    model = MoEHealth(mcfg, data.ts_dim, data.static_cardinalities, data.note_dim, data.cxr_dim, combos,
                      data.stats_dim).to(device)
    bce = nn.BCEWithLogitsLoss()
    extra: dict[str, Any] = {"n_params": sum(p.numel() for p in model.parameters()), "stages": {}}

    # Stage 0 (optional): each modality encoder pretrained on its own modality.
    enc_mult_a = 1.0
    if mcfg.encoder_pretrain == "unimodal":
        stage0 = {}
        for j, modality in enumerate(("ehr", "note", "cxr")):
            if not bool((data.train.tensors["mask"][:, j] > 0.5).any()):
                continue  # modality masked out for every sample (Table 2 subsets)
            uni, best = train_unimodal(cfg, data, modality, device, gen)
            getattr(model.encoders, modality).load_state_dict(uni.encoder.state_dict())
            stage0[modality] = best
        extra["stages"]["0"] = {"val_auroc": stage0}
        enc_mult_a = tcfg.encoder_lr_mult

    # Stage A: fusion warm-up with one generic expert.
    sa = from_node(StageConfig, cfg.stage_a)
    fwd_dense = partial(model, mode="dense")

    def loss_a(b: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = modality_dropout(b["mask"], tcfg.modality_dropout)
        return bce(model(b, mask=mask, mode="dense")["logits"], b["label"])

    shared = [(n, p) for n, p in model.named_parameters() if not n.startswith("moe.")]
    params_a: list[dict[str, object]] = [
        {"params": [p for n, p in shared if n.startswith("encoders.")], "lr": tcfg.lr * enc_mult_a},
        {"params": [p for n, p in shared if not n.startswith("encoders.")]},
    ]
    res = fit(model, params_a, loss_a, data.train,
              lambda: val_auroc(model, fwd_dense, data.val), tcfg, sa.max_epochs, sa.patience, gen, tag="stage A")
    extra["stages"]["A"] = {"val_auroc": res.best_score, "epochs": res.epochs_run, "history": res.history}

    # Stage B: combination-specific expert pretraining.
    if mcfg.pretrain in ("superset", "exact"):
        sb = from_node(StageConfig, cfg.stage_b)
        frozen = model.shared_parameters()
        for p in frozen:
            p.requires_grad_(False)
        for j, combo in enumerate(combos):
            tr = data.train.subset(pretrain_subset(data.train.tensors["combo"], combo, mcfg.pretrain))
            va = data.val.subset(pretrain_subset(data.val.tensors["combo"], combo, mcfg.pretrain))
            name = "".join(L for L, b in zip("ETI", combo, strict=True) if b)
            if len(tr) == 0:
                logger.warning("no pretraining samples for expert %s; keeping random init", name)
                continue
            cmask = torch.tensor(combo, dtype=torch.float32, device=device)
            tr_c = _frozen_cache(model, tr, cmask, tcfg.eval_batch_size)
            va_c = _frozen_cache(model, va, cmask, tcfg.eval_batch_size)
            fwd_j = partial(_cached_expert_forward, model, j=j)
            res = fit(model, list(model.moe.experts[j].parameters()), partial(_expert_bce, bce, fwd_j), tr_c,
                      partial(_neg_val_bce, model, fwd_j, va_c), tcfg, sb.max_epochs, sb.patience, gen,
                      train_modules=[model.moe.experts[j]], tag=f"stage B {name}")
            extra["stages"][f"B_{name}"] = {"n_train": len(tr), "n_val": len(va), "val_bce": -res.best_score,
                                            "epochs": res.epochs_run, "history": res.history}
        for p in frozen:
            p.requires_grad_(True)
    elif mcfg.pretrain == "generic":
        for expert in model.moe.experts:
            expert.load_state_dict(model.dense.state_dict())

    # Stage C: joint MoE training.
    def loss_c(b: dict[str, torch.Tensor]) -> torch.Tensor:
        out = model(b, mask=modality_dropout(b["mask"], tcfg.modality_dropout))
        loss = bce(out["logits"], b["label"])
        if mcfg.gating == "learned" and mcfg.balance != "none":
            loss = loss + mcfg.alpha * balance_loss(out["gates"], out["topk_idx"], mcfg.balance)
        return loss

    gw_node = cfg.get("gate_warmup")
    gw = from_node(StageConfig, gw_node) if gw_node is not None else StageConfig(max_epochs=0)
    if mcfg.gating == "learned" and gw.max_epochs > 0:
        # Optional: learn routing over the pretrained experts before joint training.
        gate_params = list(model.moe.gate.parameters())
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("moe.gate."))
        res = fit(model, gate_params, loss_c, data.train, lambda: val_auroc(model, model, data.val), tcfg,
                  gw.max_epochs, gw.patience, gen, train_modules=[model.moe.gate], tag="gate warm-up",
                  eval_init=True)
        for p in model.parameters():
            p.requires_grad_(True)
        extra["stages"]["gate_warmup"] = {"val_auroc": res.best_score, "epochs": res.epochs_run}

    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("dense.")]
    params_c: list[dict[str, object]] = [
        {"params": [p for n, p in named if _is_gate(n)], "lr": tcfg.lr * tcfg.gate_lr_mult},
        {"params": [p for n, p in named if _is_encoder(n)], "lr": tcfg.lr * tcfg.encoder_lr_mult},
        {"params": [p for n, p in named if not (_is_gate(n) or _is_encoder(n))]},
    ]
    res = fit(model, params_c, loss_c, data.train, lambda: val_auroc(model, model, data.val), tcfg,
              tcfg.max_epochs, tcfg.patience, gen, tag="stage C", eval_init=True)
    extra["stages"]["C"] = {"val_auroc": res.best_score, "epochs": res.epochs_run, "best_epoch": res.best_epoch,
                            "history": res.history}

    val = predict(model, model, data.val, tcfg.eval_batch_size)
    test = predict(model, model, data.test, tcfg.eval_batch_size)
    extra["routing_test"] = routing_summary(test, combos)
    return MethodOutput(val=val, test=test, extra=extra)
