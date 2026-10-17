"""TF-IDF note features: a per-seed TF-IDF + SVD embedding and cross-fitted logistic-regression logits.

Everything that sees labels or fits a vocabulary uses training admissions only. Training rows get
out-of-fold logits (folds grouped by patient); validation / test rows get the logit of a model fit
on all training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.data_module.note_features import salient_text

__all__ = ["admission_documents", "crossfit_logits", "fit_tfidf", "svd_embedding"]


def admission_documents(hadm_ids: np.ndarray, notes: pd.DataFrame) -> np.ndarray:
    """Salient sections of all in-window reports per admission, joined; "" when there is none."""
    clean = notes.assign(clean=[salient_text(t)[0] for t in notes["text"]])
    joined = clean.groupby("hadm_id")["clean"].agg(" ".join)
    return pd.Series(hadm_ids).map(joined).fillna("").to_numpy(dtype=object)


def fit_tfidf(docs: np.ndarray, fit_rows: np.ndarray, ngram_max: int, min_df: int,
              max_features: int) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Vocabulary and IDF from ``docs[fit_rows]``; returns the vectorizer and all documents transformed."""
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_features=max_features, sublinear_tf=True)
    vec.fit(docs[fit_rows])
    return vec, vec.transform(docs).tocsr()


def svd_embedding(x: sparse.csr_matrix, fit_rows: np.ndarray, has_doc: np.ndarray, dim: int,
                  seed: int) -> tuple[np.ndarray, float]:
    """[N, dim] TruncatedSVD of the TF-IDF rows (fit on ``fit_rows``); zeros for admissions without notes."""
    svd = TruncatedSVD(dim, random_state=seed).fit(x[fit_rows])
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[has_doc] = svd.transform(x[has_doc]).astype(np.float32)
    return out, float(svd.explained_variance_ratio_.sum())


def crossfit_logits(x: sparse.csr_matrix, y: np.ndarray, train_rows: np.ndarray, other_rows: np.ndarray,
                    groups: np.ndarray, folds: int, c: float) -> np.ndarray:
    """[N] logits: out-of-fold for ``train_rows`` (GroupKFold on ``groups``), full-train fit for ``other_rows``.

    Rows in neither set stay 0.
    """
    out = np.zeros(x.shape[0], dtype=np.float32)
    for fit_idx, pred_idx in GroupKFold(folds).split(train_rows, groups=groups[train_rows]):
        lr = LogisticRegression(C=c, max_iter=3000).fit(x[train_rows[fit_idx]], y[train_rows[fit_idx]])
        out[train_rows[pred_idx]] = lr.decision_function(x[train_rows[pred_idx]])
    lr = LogisticRegression(C=c, max_iter=3000).fit(x[train_rows], y[train_rows])
    if len(other_rows):
        out[other_rows] = lr.decision_function(x[other_rows])
    return out
