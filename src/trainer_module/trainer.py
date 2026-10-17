"""Generic training loop with early stopping, and batched prediction."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_module.dataset import SplitData
from src.metrics_module.metrics import safe_auroc
from src.utils.config import TrainConfig

logger = logging.getLogger(__name__)

__all__ = ["FitResult", "fit", "modality_dropout", "predict", "val_auroc", "val_bce"]

Batch = dict[str, torch.Tensor]
Forward = Callable[[Batch], dict[str, torch.Tensor]]


@dataclass
class FitResult:
    best_score: float
    best_epoch: float
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)


def _set_mode(model: nn.Module, train_modules: list[nn.Module] | None) -> None:
    if train_modules is None:
        model.train()
        return
    model.eval()
    for m in train_modules:
        m.train()


def fit(
    model: nn.Module,
    params: list[nn.Parameter] | list[dict[str, object]],
    loss_fn: Callable[[Batch], torch.Tensor],
    train: SplitData,
    val_score: Callable[[], float],
    tcfg: TrainConfig,
    max_epochs: int,
    patience: int,
    generator: torch.Generator,
    train_modules: list[nn.Module] | None = None,
    tag: str = "",
    eval_init: bool = False,
) -> FitResult:
    """AdamW training; keeps the parameters with the best validation score.

    Validation runs ``tcfg.evals_per_epoch`` times per epoch; ``patience`` is in epochs.
    ``eval_init``: score the starting weights first and keep them if training never beats them
    (useful when the stage starts from already-trained parts).
    ``params``: parameters or AdamW parameter groups (e.g. a larger lr for the gate).
    ``train_modules``: if given, the rest of the model stays in eval mode (frozen stages).
    """
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    flat = [p for g in opt.param_groups for p in g["params"]]
    ema = {id(p): p.detach().clone() for p in flat} if tcfg.ema_decay > 0 else None
    steps_per_epoch = max(1, math.ceil(len(train) / tcfg.batch_size))
    evals = max(1, int(tcfg.evals_per_epoch))
    eval_every = max(1, math.ceil(steps_per_epoch / evals))
    best, best_at, bad, stop = -np.inf, -1.0, 0, False
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    if eval_init:
        best, best_at = float(val_score()), 0.0
        history.append({"epoch": 0.0, "train_loss": float("nan"), "val_score": best})
    epoch = -1
    for epoch in range(max_epochs):
        _set_mode(model, train_modules)
        total, count = 0.0, 0
        for i, batch in enumerate(train.batches(tcfg.batch_size, shuffle=True, generator=generator)):
            loss = loss_fn(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flat, tcfg.grad_clip)
            opt.step()
            if ema is not None:
                _ema_update(flat, ema, tcfg.ema_decay)
            total += float(loss.detach()) * len(batch["label"])
            count += len(batch["label"])
            if (i + 1) % eval_every and i + 1 != steps_per_epoch:
                continue
            if ema is not None:
                _swap(flat, ema)  # evaluate (and keep) the averaged weights
            score = float(val_score())
            improved = score > best + 1e-6
            if improved:
                best, best_at, bad = score, round(epoch + (i + 1) / steps_per_epoch, 3), 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if ema is not None:
                _swap(flat, ema)
            _set_mode(model, train_modules)
            at = round(epoch + (i + 1) / steps_per_epoch, 3)
            history.append({"epoch": at, "train_loss": total / max(count, 1), "val_score": score})
            total, count = 0.0, 0
            if bad >= patience * evals:
                stop = True
                break
        if stop:
            break
    model.load_state_dict(best_state)
    logger.info("[%s] best val %.4f at epoch %.2f (%d epochs run)", tag, best, best_at, epoch + 1)
    return FitResult(float(best), best_at, epoch + 1, history)


@torch.no_grad()
def _ema_update(params: list[nn.Parameter], ema: dict[int, torch.Tensor], decay: float) -> None:
    for p in params:
        ema[id(p)].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


@torch.no_grad()
def _swap(params: list[nn.Parameter], store: dict[int, torch.Tensor]) -> None:
    """Exchange parameter values with ``store`` (calling twice restores the original)."""
    for p in params:
        tmp = p.detach().clone()
        p.copy_(store[id(p)])
        store[id(p)] = tmp


def modality_dropout(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Training-time augmentation: drop present notes / CXR (columns 1, 2) with probability p.

    EHR (column 0) is never dropped; dropped modalities take the model's missing representation.
    """
    if p <= 0:
        return mask
    keep = (torch.rand(mask.shape[0], 2, device=mask.device) >= p).to(mask.dtype)
    return torch.cat([mask[:, :1], mask[:, 1:] * keep], dim=1)


@torch.no_grad()
def predict(model: nn.Module, forward: Forward, data: SplitData, batch_size: int = 1024) -> dict[str, np.ndarray]:
    """Probabilities plus any routing tensors returned by ``forward``; row metadata attached."""
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {}
    for batch in data.batches(batch_size, shuffle=False):
        out = forward(batch)
        chunks.setdefault("prob", []).append(torch.sigmoid(out["logits"]).float().cpu())
        for key in ("gates", "topk_idx", "topk_w"):
            if key in out:
                chunks.setdefault(key, []).append(out[key].float().cpu())
        for key in ("label", "row", "combo"):
            chunks.setdefault(key, []).append(batch[key].cpu())
    return {k: torch.cat(v).numpy() for k, v in chunks.items()}


def val_auroc(model: nn.Module, forward: Forward, data: SplitData, batch_size: int = 1024) -> float:
    pred = predict(model, forward, data, batch_size)
    score = safe_auroc(pred["label"], pred["prob"])
    return 0.0 if np.isnan(score) else score


@torch.no_grad()
def val_bce(model: nn.Module, forward: Forward, data: SplitData, batch_size: int = 1024) -> float:
    """Mean BCE on ``data`` (0 if empty); used for small pretraining subsets."""
    if len(data) == 0:
        return 0.0
    model.eval()
    total = 0.0
    for batch in data.batches(batch_size, shuffle=False):
        total += float(F.binary_cross_entropy_with_logits(forward(batch)["logits"], batch["label"], reduction="sum"))
    return total / len(data)
