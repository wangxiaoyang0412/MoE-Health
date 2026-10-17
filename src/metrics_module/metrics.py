"""Binary classification metrics: AUROC, AUPRC and F1 at a validation-chosen threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

__all__ = ["best_f1_threshold", "binary_metrics", "safe_auroc"]


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present."""
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability threshold that maximises F1 (chosen on validation, applied to test)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more entry than thresholds; drop the last point.
    return float(thresholds[int(np.nanargmax(f1[:-1]))]) if len(thresholds) else 0.5


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """AUROC, AUPRC, F1 at ``threshold`` and at 0.5, plus the positive rate."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    two_class = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two_class else float("nan"),
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, p >= 0.5, zero_division=0)),
        "threshold": float(threshold),
    }
