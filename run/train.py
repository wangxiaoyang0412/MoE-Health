"""Train and evaluate one method on one (cohort, task, seed).

Examples (from the project root):
    .venv/bin/python run/train.py model=moe_health task=mortality cohort=corrected seed=0
    .venv/bin/python run/train.py model=moe_health +recipe=moe_final task=los seed=1 device=cuda:2
    .venv/bin/python run/train.py model=moe_health model.top_k=1 tag=_top1 exp=ablation

Writes metrics.json (aggregates) and predictions.npz (per-sample predictions: credentialed data, keep it
with the MIMIC files).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_module import DataConfig, DatasetFactory
from src.metrics_module.metrics import best_f1_threshold, binary_metrics
from src.trainer_module import MethodFactory
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

COMBO_NAMES = {1: "E", 3: "ET", 5: "EI", 7: "ETI", 2: "T", 4: "I", 6: "TI", 0: "none"}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _per_combo(pred: dict[str, np.ndarray], threshold: float) -> dict[str, dict[str, float]]:
    return {COMBO_NAMES.get(int(c), str(c)): binary_metrics(pred["label"][pred["combo"] == c],
                                                            pred["prob"][pred["combo"] == c], threshold)
            for c in np.unique(pred["combo"])}


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    start = time.time()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = set_seed(int(cfg.seed))
    device = torch.device(cfg.device)
    dcfg = DataConfig(
        feature_dir=str(cfg.paths.feature_dir), cohort_path=str(Path(cfg.paths.cohort_dir) / "cohort.parquet"),
        cohort=str(cfg.cohort), task=str(cfg.task), seed=int(cfg.seed), cxr_variant=str(cfg.data.cxr_variant),
        use_meds=bool(cfg.data.use_meds), modalities=str(cfg.data.modalities),
        note_variant=str(cfg.data.get("note_variant", "cls")), ehr_delta=bool(cfg.data.get("ehr_delta", False)),
        ehr_stats=bool(cfg.data.get("ehr_stats", False)), ehr_version=str(cfg.data.get("ehr_version", "v1")),
        history=bool(cfg.data.get("history", False)), note_logit=bool(cfg.data.get("note_logit", False)),
    )
    data = DatasetFactory("mimic_mm")(dcfg, device=device)
    result = MethodFactory(str(cfg.model.method))(cfg, data, device, gen)

    threshold = best_f1_threshold(result.val["label"], result.val["prob"])
    metrics = {
        "method": str(cfg.model.method), "model": str(cfg.model.name), "tag": str(cfg.tag),
        "cohort": str(cfg.cohort), "task": str(cfg.task), "seed": int(cfg.seed), "modalities": dcfg.modalities,
        "val": binary_metrics(result.val["label"], result.val["prob"], threshold),
        "test": binary_metrics(result.test["label"], result.test["prob"], threshold),
        "test_per_combo": _per_combo(result.test, threshold),
        "data": data.info, "extra": result.extra, "minutes": round((time.time() - start) / 60, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    np.savez_compressed(out_dir / "predictions.npz",
                        **{f"val_{k}": v for k, v in result.val.items()},
                        **{f"test_{k}": v for k, v in result.test.items()})
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    t = metrics["test"]
    logger.info("TEST %s %s/%s seed %d: AUROC %.4f AUPRC %.4f F1 %.4f (%.1f min)", cfg.model.name, cfg.cohort,
                cfg.task, cfg.seed, t["auroc"], t["auprc"], t["f1"], metrics["minutes"])


if __name__ == "__main__":
    main()
