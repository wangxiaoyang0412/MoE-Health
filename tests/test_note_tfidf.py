"""TF-IDF note features on synthetic reports: train-only fitting, zeros without notes, out-of-fold logits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_module.note_tfidf import admission_documents, crossfit_logits, fit_tfidf, svd_embedding


def _docs(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    words = np.array(["edema", "effusion", "clear", "stable", "line", "tube", "normal", "opacity"])
    docs = np.array([" ".join(rng.choice(words, 12)) + (" consolidation worsening" if yi else " unremarkable")
                     for yi in y], dtype=object)
    return docs, y


def test_admission_documents_join_salient_sections() -> None:
    notes = pd.DataFrame({"hadm_id": [1, 1, 3], "text": ["FINDINGS: edema.\nTECHNIQUE: AP.", "IMPRESSION: worse.",
                                                         "no headings here"]})
    docs = admission_documents(np.array([1, 2, 3]), notes)
    assert docs[0] == "FINDINGS: edema. IMPRESSION: worse."
    assert docs[1] == "" and docs[2] == "no headings here"


def test_svd_embedding_fits_on_train_rows_and_zeros_missing() -> None:
    docs, _ = _docs()
    docs[:10] = ""
    has = docs != ""
    fit_rows = np.arange(10, 120)
    vec, x = fit_tfidf(docs, fit_rows, ngram_max=2, min_df=2, max_features=1000)
    assert set(vec.vocabulary_) <= {w for d in docs[fit_rows] for w in d.split()} | {
        f"{a} {b}" for d in docs[fit_rows] for a, b in zip(d.split(), d.split()[1:], strict=False)}
    emb, explained = svd_embedding(x, fit_rows, has, dim=4, seed=0)
    assert emb.shape == (200, 4) and np.all(emb[:10] == 0) and 0 < explained <= 1


def test_crossfit_logits_are_out_of_fold_and_predictive() -> None:
    docs, y = _docs()
    _, x = fit_tfidf(docs, np.arange(120), ngram_max=1, min_df=1, max_features=1000)
    train, other = np.arange(120), np.arange(120, 190)
    groups = np.repeat(np.arange(60), 2)
    groups = np.r_[groups, np.arange(60, 140)]
    z = crossfit_logits(x, y, train, other, groups, folds=5, c=1.0)
    assert np.all(z[190:] == 0)
    assert roc_auc_score(y[train], z[train]) > 0.9 and roc_auc_score(y[other], z[other]) > 0.9
