"""load_task_data on synthetic feature files laid out like outputs/features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_module.dataset import DataConfig, load_task_data
from src.data_module.ehr_spec import INTERVAL_VARS, MED_ITEMS, STATIC_CATEGORICAL


def _write(root: Path, n: int = 60) -> DataConfig:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    split = np.array(["train"] * 40 + ["val"] * 10 + ["test"] * 10)
    cohort = pd.DataFrame({
        "hadm_id": np.arange(100, 100 + n), "has_note": rng.integers(0, 2, n), "has_cxr": rng.integers(0, 2, n),
        "in_corrected": np.r_[np.ones(n - 5, int), np.zeros(5, int)], "y_mortality": rng.integers(0, 2, n),
        "y_readmission": np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 2, n)), "split_s0": split,
    })
    cohort.to_parquet(root / "cohort.parquet", index=False)
    vals = rng.normal(size=(n, 48, 3)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.5] = np.nan
    np.save(root / "ehr_values.npy", vals)
    np.save(root / "ehr_active.npy", rng.integers(0, 2, (n, 48, len(INTERVAL_VARS))).astype(np.uint8))
    np.save(root / "static_num.npy", rng.normal(60, 10, (n, 1)).astype(np.float32))
    np.save(root / "static_cat.npy", rng.integers(0, 2, (n, len(STATIC_CATEGORICAL))))
    np.save(root / "note_emb.npy", rng.normal(size=(n, 6)).astype(np.float32))
    np.save(root / "cxr_emb.npy", rng.normal(size=(n, 5)).astype(np.float32))
    np.save(root / "hadm_order.npy", cohort["hadm_id"].to_numpy())
    meta = {"continuous_vars": ["a", "b", "c"], "interval_vars": INTERVAL_VARS,
            "static_categorical": STATIC_CATEGORICAL, "vocab": {c: ["UNKNOWN", "x"] for c in STATIC_CATEGORICAL}}
    (root / "ehr_meta.json").write_text(json.dumps(meta))
    np.save(root / "history_cat.npy", rng.integers(0, 3, (n, 2)))
    np.save(root / "history_hadm.npy", cohort["hadm_id"].to_numpy())
    (root / "history_meta.json").write_text(json.dumps({"columns": ["h1", "h2"],
                                                        "vocab": {"h1": ["U", "a", "b"], "h2": ["U", "a", "b"]}}))
    return DataConfig(feature_dir=str(root), cohort_path=str(root / "cohort.parquet"))


def test_load_task_data_shapes_and_filters(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "f")
    data = load_task_data(cfg)
    n_all = len(data.train) + len(data.val) + len(data.test)
    assert n_all == 55  # 5 rows outside the corrected cohort are dropped
    assert data.ts_dim == 3 * 2 + len(INTERVAL_VARS)
    x = data.train.tensors["ehr_ts"].numpy()
    assert np.isfinite(x).all()
    assert data.train.tensors["mask"][:, 0].eq(1).all()

    readm = load_task_data(DataConfig(**{**cfg.__dict__, "task": "readmission", "use_meds": False}))
    assert len(readm.train) + len(readm.val) + len(readm.test) < n_all  # NaN labels dropped
    assert readm.ts_dim == 3 * 2 + len(INTERVAL_VARS) - len(MED_ITEMS)

    masked = load_task_data(DataConfig(**{**cfg.__dict__, "modalities": "TI"}))
    assert masked.train.tensors["mask"][:, 0].eq(0).all()

    stats = load_task_data(DataConfig(**{**cfg.__dict__, "ehr_stats": True, "ehr_delta": True}))
    assert stats.stats_dim == 3 * 11 + 3 + len(INTERVAL_VARS)  # stats per variable, observed fraction, active
    assert stats.ts_dim == 3 * 3 + len(INTERVAL_VARS)  # + hours-since-last channels
    assert torch.isfinite(stats.train.tensors["ehr_stats"]).all()

    hist = load_task_data(DataConfig(**{**cfg.__dict__, "history": True}))
    assert hist.static_cardinalities[-2:] == [3, 3]
    assert hist.train.tensors["static_cat"].shape[1] == len(STATIC_CATEGORICAL) + 2


def test_history_codes_are_checked_against_the_cohort(tmp_path: Path) -> None:
    cfg = DataConfig(**{**_write(tmp_path / "f").__dict__, "history": True})
    feat = Path(cfg.feature_dir)
    np.save(feat / "history_hadm.npy", np.load(feat / "history_hadm.npy")[::-1])  # stale row order
    with pytest.raises(ValueError, match="history_hadm"):
        load_task_data(cfg)
