"""In-memory multimodal dataset aligned to cohort.parquet.

All normalisation statistics come from the training split only:
- continuous EHR variables: clip to train quantiles, z-score, forward-fill in time,
  unobserved-before-first-measurement -> 0 (train mean); an observation mask and
  the binary procedure/medication channels are concatenated;
- age: z-score; categorical static fields: integer codes (0 = UNKNOWN);
- ClinicalBERT / DenseNet features: per-dimension z-score over rows that have the modality.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data_module.ehr_spec import PROCEDURE_ITEMS
from src.data_module.ehr_stats import ehr_summary_features
from src.data_module.registry import register_dataset

logger = logging.getLogger(__name__)

__all__ = ["DataConfig", "SplitData", "TaskData", "ffill_time", "hours_since_last", "load_task_data",
           "prepare_ehr"]

LABELS = {
    "corrected": {"mortality": "y_mortality", "los": "y_los", "readmission": "y_readmission"},
    "faithful": {
        "mortality": "y_mortality_faithful",
        "los": "y_los_faithful",
        "readmission": "y_readmission_faithful",
    },
}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DataConfig:
    feature_dir: str
    cohort_path: str
    cohort: str = "corrected"  # corrected | faithful
    task: str = "mortality"  # mortality | los | readmission
    seed: int = 0
    cxr_variant: str = "first"  # first (paper) | mean
    use_meds: bool = True
    modalities: str = "ETI"  # Table 2: modalities outside this set are masked for every sample
    note_variant: str = "cls"  # cls (paper Eq. 7) | tok (mean of token states)
    ehr_delta: bool = False  # add hours-since-last-measurement channels
    ehr_stats: bool = False  # HAIM-style 48 h summary statistics as an extra EHR-encoder input
    ehr_version: str = "v1"  # v1 (HAIM variable list) | v2 (extended ICU variable set)
    history: bool = False  # + prior-utilisation / admission-type codes as static categoricals
    clip_low: float = 0.005
    clip_high: float = 0.995


@dataclass
class SplitData:
    """Row-aligned tensors of one split (kept on the training device)."""

    tensors: dict[str, torch.Tensor]

    def __len__(self) -> int:
        return int(self.tensors["label"].shape[0])

    def subset(self, mask: torch.Tensor) -> SplitData:
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        return SplitData({k: v[idx] for k, v in self.tensors.items()})

    def batches(
        self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        device = self.tensors["label"].device
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size].to(device)
            yield {k: v[idx] for k, v in self.tensors.items()}


@dataclass
class TaskData:
    train: SplitData
    val: SplitData
    test: SplitData
    ts_dim: int
    static_cardinalities: list[int]
    note_dim: int
    cxr_dim: int
    stats_dim: int = 0
    info: dict[str, Any] = field(default_factory=dict)


def ffill_time(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 1 of an [N, T, V] array (leading NaNs stay NaN)."""
    n, t, v = x.shape
    idx = np.where(~np.isnan(x), np.arange(t)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return x[np.arange(n)[:, None, None], idx, np.arange(v)[None, None, :]]


def hours_since_last(observed: np.ndarray) -> np.ndarray:
    """[N, T, V] hours since the variable was last observed, scaled to [0, 1] (1 = never observed)."""
    n, t, v = observed.shape
    hours = np.arange(t, dtype=np.float32)[None, :, None]
    last = np.where(observed, hours, -np.inf)
    np.maximum.accumulate(last, axis=1, out=last)
    delta = np.where(np.isinf(last), float(t), hours - last)
    return (delta / t).astype(np.float32)


def prepare_ehr(
    values: np.ndarray, active: np.ndarray, train_rows: np.ndarray, clip_low: float, clip_high: float
) -> np.ndarray:
    """[N, T, 2V + P] float32: normalised forward-filled values, observation mask, active flags."""
    observed = ~np.isnan(values)
    tr = values[train_rows]
    has = (~np.isnan(tr)).any(axis=(0, 1))
    lo = np.zeros(values.shape[2], dtype=np.float32)
    hi = np.zeros(values.shape[2], dtype=np.float32)
    mean = np.zeros(values.shape[2], dtype=np.float32)
    std = np.ones(values.shape[2], dtype=np.float32)
    for j in np.nonzero(has)[0]:
        col = tr[:, :, j][~np.isnan(tr[:, :, j])]
        lo[j], hi[j] = np.quantile(col, [clip_low, clip_high])
        clipped = np.clip(col, lo[j], hi[j])
        mean[j], std[j] = clipped.mean(), max(float(clipped.std()), 1e-6)
    z = (np.clip(values, lo, hi) - mean) / std
    z[:, :, ~has] = np.nan
    z = np.nan_to_num(ffill_time(z), nan=0.0)
    return np.concatenate([z, observed, active], axis=2).astype(np.float32)


def _check_order(feat: Path, order_file: str, cohort: pd.DataFrame) -> None:
    """Raise if a feature file was written for a different cohort row order (stale extraction)."""
    if not np.array_equal(np.load(feat / order_file), cohort["hadm_id"].to_numpy()):
        raise ValueError(f"{order_file} does not match cohort.parquet; re-run the feature extraction")


def _standardize(x: np.ndarray, present: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    ref = x[train_rows][present[train_rows]]
    mean, std = ref.mean(axis=0), ref.std(axis=0) + 1e-6
    out = (x - mean) / std
    out[~present] = 0.0
    return out.astype(np.float32)


@register_dataset("mimic_mm")
def load_task_data(cfg: DataConfig, device: str | torch.device = "cpu") -> TaskData:
    """Build train/val/test tensors for one (cohort, task, seed)."""
    feat = Path(cfg.feature_dir)
    cohort = pd.read_parquet(cfg.cohort_path)
    if not np.array_equal(np.load(feat / "hadm_order.npy"), cohort["hadm_id"].to_numpy()):
        raise ValueError("feature arrays are not aligned with cohort.parquet")
    for count_file, col in (("note_count.npy", "n_note"), ("cxr_count.npy", "n_cxr")):
        if (feat / count_file).exists() and not np.array_equal(np.load(feat / count_file), cohort[col].to_numpy()):
            raise ValueError(f"{count_file} does not match cohort.{col}; re-run the feature extraction")
    label_col = LABELS[cfg.cohort][cfg.task]
    keep = cohort[label_col].notna().to_numpy()  # read-only under pandas copy-on-write
    if cfg.cohort == "corrected":
        keep = keep & (cohort["in_corrected"].to_numpy() == 1)
    split = cohort[f"split_s{cfg.seed}"].to_numpy()
    rows = np.nonzero(keep)[0]
    pos = {s: np.nonzero(split[rows] == s)[0] for s in SPLITS}  # positions within ``rows``

    suffix = "" if cfg.ehr_version == "v1" else f"_{cfg.ehr_version}"
    values = np.load(feat / f"ehr_values{suffix}.npy", mmap_mode="r")[rows]
    active = np.load(feat / "ehr_active.npy", mmap_mode="r")[rows]
    meta = json.loads((feat / "ehr_meta.json").read_text())
    if not cfg.use_meds:  # interval channels are procedures first, then medications
        active = active[:, :, : len(PROCEDURE_ITEMS)]
    values = np.asarray(values, dtype=np.float32)
    ehr = prepare_ehr(values, np.asarray(active), pos["train"], cfg.clip_low, cfg.clip_high)
    if cfg.ehr_delta:
        ehr = np.concatenate([ehr, hours_since_last(~np.isnan(values))], axis=2)

    age = np.load(feat / "static_num.npy")[rows]
    age = ((age - age[pos["train"]].mean()) / (age[pos["train"]].std() + 1e-6)).astype(np.float32)
    static_cat = np.load(feat / "static_cat.npy")[rows]
    history_cards: list[int] = []
    if cfg.history:
        _check_order(feat, "history_hadm.npy", cohort)
        hmeta = json.loads((feat / "history_meta.json").read_text())
        static_cat = np.concatenate([static_cat, np.load(feat / "history_cat.npy")[rows]], axis=1)
        history_cards = [len(hmeta["vocab"][c]) for c in hmeta["columns"]]
    has_note = cohort["has_note"].to_numpy()[rows].astype(bool)
    has_cxr = cohort["has_cxr"].to_numpy()[rows].astype(bool)
    note_file = "note_emb.npy" if cfg.note_variant == "cls" else "note_emb_tok.npy"
    note = _standardize(np.load(feat / note_file, mmap_mode="r")[rows], has_note, pos["train"])
    cxr_file = "cxr_emb.npy" if cfg.cxr_variant == "first" else "cxr_emb_mean.npy"
    cxr = _standardize(np.load(feat / cxr_file, mmap_mode="r")[rows], has_cxr, pos["train"])
    mask = np.stack([np.ones(len(rows)), has_note, has_cxr], axis=1).astype(np.float32)
    mask *= np.array([float(m in cfg.modalities) for m in "ETI"], dtype=np.float32)
    combo = (mask * np.array([1, 2, 4])).sum(axis=1).astype(np.int64)
    label = cohort[label_col].to_numpy()[rows].astype(np.float32)

    arrays = {"ehr_ts": ehr, "static_num": age, "static_cat": static_cat, "note": note, "cxr": cxr,
              "mask": mask, "combo": combo, "label": label, "row": rows.astype(np.int64)}
    n_interval = int(active.shape[2])
    if cfg.ehr_stats:
        st = ehr_summary_features(ehr, int(values.shape[2]), n_interval)
        mu, sd = st[pos["train"]].mean(axis=0), st[pos["train"]].std(axis=0) + 1e-6
        arrays["ehr_stats"] = ((st - mu) / sd).astype(np.float32)
    splits = {s: SplitData({k: torch.from_numpy(np.ascontiguousarray(v[pos[s]])).to(device)
                            for k, v in arrays.items()}) for s in SPLITS}
    cards = [len(meta["vocab"][c]) for c in meta["static_categorical"]] + history_cards
    info = {"label": label_col, **{f"n_{s}": int(len(pos[s])) for s in SPLITS},
            "pos_rate_train": float(label[pos["train"]].mean()), "n_continuous": int(values.shape[2]),
            "n_interval": n_interval}
    logger.info("task data %s/%s seed %d: %s", cfg.cohort, cfg.task, cfg.seed, info)
    stats_dim = int(arrays["ehr_stats"].shape[1]) if "ehr_stats" in arrays else 0
    return TaskData(splits["train"], splits["val"], splits["test"], ts_dim=ehr.shape[2],
                    static_cardinalities=cards, note_dim=note.shape[1], cxr_dim=cxr.shape[1],
                    stats_dim=stats_dim, info=info)
