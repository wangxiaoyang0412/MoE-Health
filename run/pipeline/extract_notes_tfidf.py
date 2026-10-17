"""TF-IDF note features aligned to cohort.parquet, fit per split seed on training admissions only.

Outputs (paths.feature_dir):
    note_emb_tfidf_s{seed}.npy                  [N, svd_dim] TF-IDF + SVD embedding (zeros without notes)
    note_tfidf_logit_{cohort}_{task}_s{seed}.npy [N] logistic-regression logit (out-of-fold on train rows)
    tfidf_hadm.npy                              [N] hadm_id row order (checked when loading)
    note_tfidf_report.md                        aggregates only
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from src.data_module.dataset import LABELS
from src.data_module.note_tfidf import admission_documents, crossfit_logits, fit_tfidf, svd_embedding

logger = logging.getLogger(__name__)

TASKS = ("mortality", "los", "readmission")


def _cell(cohort: pd.DataFrame, x, seed: int, name: str, task: str, folds: int, c: float) -> tuple:
    label = cohort[LABELS[name][task]].to_numpy()
    eligible = ~np.isnan(label) & (cohort["has_note"].to_numpy() == 1)
    if name == "corrected":
        eligible &= cohort["in_corrected"].to_numpy() == 1
    split = cohort[f"split_s{seed}"].to_numpy()
    rows = {s: np.nonzero(eligible & (split == s))[0] for s in ("train", "val", "test")}
    y = np.nan_to_num(label).astype(int)
    z = crossfit_logits(x, y, rows["train"], np.r_[rows["val"], rows["test"]], cohort["subject_id"].to_numpy(),
                        folds, c)
    aucs = {s: float(roc_auc_score(y[r], z[r])) for s, r in rows.items()}
    return name, task, seed, z, aucs


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    t = cfg.tfidf
    feat = Path(cfg.paths.feature_dir)
    feat.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(Path(cfg.paths.cohort_dir) / "cohort.parquet")
    notes = pd.read_parquet(Path(cfg.paths.cohort_dir) / "notes_window.parquet", columns=["hadm_id", "text"])
    docs = admission_documents(cohort["hadm_id"].to_numpy(), notes)
    has_doc = (docs != "") & (cohort["has_note"].to_numpy() == 1)
    if not np.array_equal(has_doc, cohort["has_note"].to_numpy() == 1):
        raise ValueError("admissions with notes and non-empty documents disagree")
    np.save(feat / "tfidf_hadm.npy", cohort["hadm_id"].to_numpy())

    lines = ["# TF-IDF note features (aggregates only)", "", "| seed | vocabulary | SVD explained variance |",
             "|---|---|---|"]
    results = []
    for seed in (int(s) for s in cfg.cohort.seeds):
        fit_rows = np.nonzero((cohort[f"split_s{seed}"].to_numpy() == "train") & has_doc)[0]
        vec, x = fit_tfidf(docs, fit_rows, int(t.ngram_max), int(t.min_df), int(t.max_features))
        emb, explained = svd_embedding(x, fit_rows, has_doc, int(t.svd_dim), seed)
        np.save(feat / f"note_emb_tfidf_s{seed}.npy", emb)
        lines.append(f"| {seed} | {len(vec.vocabulary_)} | {explained:.3f} |")
        cells = itertools.product(("corrected", "faithful"), TASKS)
        results += Parallel(n_jobs=int(t.n_jobs))(delayed(_cell)(cohort, x, seed, name, task, int(t.folds),
                                                                 float(t.c)) for name, task in cells)
        for name, task, s, z, _ in results[-6:]:
            np.save(feat / f"note_tfidf_logit_{name}_{task}_s{s}.npy", z)
        logger.info("seed %d done", seed)

    lines += ["", "Logit AUROC on admissions with notes (train = out-of-fold), mean over seeds:", "",
              "| cohort | task | train (OOF) | val | test |", "|---|---|---|---|---|"]
    df = pd.DataFrame([{"cohort": n, "task": k, **a} for n, k, _, _, a in results])
    for (name, task), g in df.groupby(["cohort", "task"], sort=False):
        lines.append(f"| {name} | {task} | {g['train'].mean():.3f} | {g['val'].mean():.3f} | {g['test'].mean():.3f} |")
    (feat / "note_tfidf_report.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote TF-IDF features: %d seeds, %d logit files", len(cfg.cohort.seeds), len(results))


if __name__ == "__main__":
    main()
