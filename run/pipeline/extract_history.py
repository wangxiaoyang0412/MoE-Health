"""Prior-utilisation + admission-type features (known at admission) aligned to cohort.parquet.

Outputs (paths.feature_dir):
    history_cat.npy     [N, 6] int64 codes (0 = UNKNOWN), columns in history_meta.json
    history_hadm.npy    [N] hadm_id row order (checked against cohort.parquet when loading)
    history_meta.json   column names and vocabularies
    history_report.md   aggregates only: bin counts and train-split readmission rate per bin
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_module.history import HISTORY_COLUMNS, HISTORY_VOCAB, admission_categoricals, prior_utilization
from src.utils.report import df_to_markdown

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    feat = Path(cfg.paths.feature_dir)
    mimic = Path(cfg.paths.mimiciv)
    cohort = pd.read_parquet(Path(cfg.paths.cohort_dir) / "cohort.parquet")
    admissions = pd.read_csv(mimic / "core" / "admissions.csv", usecols=["subject_id", "hadm_id", "admittime",
                                                                         "dischtime"])
    icustays = pd.read_csv(mimic / "icu" / "icustays.csv", usecols=["subject_id", "intime", "outtime"])

    hist = prior_utilization(cohort, admissions, icustays)
    adm_codes, adm_vocab = admission_categoricals(cohort)
    codes = np.concatenate([hist[list(HISTORY_VOCAB)].to_numpy(dtype=np.int64), adm_codes], axis=1)
    np.save(feat / "history_cat.npy", codes)
    np.save(feat / "history_hadm.npy", cohort["hadm_id"].to_numpy())
    meta ={"columns": HISTORY_COLUMNS, "vocab": {**HISTORY_VOCAB, **adm_vocab}}
    (feat / "history_meta.json").write_text(json.dumps(meta, indent=2))

    train = (cohort["in_corrected"] == 1) & (cohort["split_s0"] == "train") & cohort["y_readmission"].notna()
    lines = ["# History feature report (aggregates only)", ""]
    for j, col in enumerate(HISTORY_COLUMNS[:4]):
        df = pd.DataFrame({"code": codes[:, j], "readm": cohort["y_readmission"], "train": train})
        rows = []
        for code, label in enumerate(meta["vocab"][col]):
            sel = df["code"] == code
            if sel.any():
                tr = df[sel & df["train"]]
                rows.append({"bin": label, "n": int(sel.sum()), "train readmission rate": float(tr["readm"].mean())})
        lines += [f"## {col}", "", df_to_markdown(pd.DataFrame(rows)), ""]
    lines.append(f"admission_type categories: {len(adm_vocab['admission_type']) - 1}; "
                 f"admission_location categories: {len(adm_vocab['admission_location']) - 1}")
    (feat / "history_report.md").write_text("\n".join(lines) + "\n")
    logger.info("history features %s written", codes.shape)


if __name__ == "__main__":
    main()
