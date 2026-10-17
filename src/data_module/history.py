"""Prior-utilisation features known at admission time (optional static inputs).

Only admissions / ICU stays that started and ended before the current admittime are used, so
nothing after the prediction time leaks in (also for rows with corrupted times, e.g. discharge
before admission). Counts and gaps are binned into categorical codes
(0 = UNKNOWN, as for the other static fields) and fed through the static embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["HISTORY_COLUMNS", "HISTORY_VOCAB", "admission_categoricals", "prior_utilization"]

DAY = pd.Timedelta(days=1)

# column -> ordered bin labels; code = 1 + bin index
HISTORY_VOCAB: dict[str, list[str]] = {
    "prior_adm_365d": ["UNKNOWN", "0", "1", "2", "3-4", "5+"],
    "prior_adm_all": ["UNKNOWN", "0", "1", "2-3", "4-7", "8+"],
    "days_since_discharge": ["UNKNOWN", "none", "<=30", "31-90", "91-365", ">365"],
    "prior_icu_365d": ["UNKNOWN", "0", "1", "2+"],
}
HISTORY_COLUMNS = [*HISTORY_VOCAB, "admission_type", "admission_location"]


def _bin(values: pd.Series, edges: list[float]) -> np.ndarray:
    """Codes 1..len(edges)+1 for right-closed bins (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    return (np.searchsorted(np.asarray(edges, dtype=float), values.to_numpy(dtype=float), side="left") + 1).astype(
        np.int64
    )


def prior_utilization(cohort: pd.DataFrame, admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Binned prior admissions / ICU stays per cohort admission (same row order as ``cohort``)."""
    cur = cohort[["subject_id", "hadm_id", "admittime"]].copy()
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].rename(
        columns={"hadm_id": "prev_hadm", "admittime": "prev_admit", "dischtime": "prev_disch"}
    )
    for col in ("prev_admit", "prev_disch"):
        adm[col] = pd.to_datetime(adm[col])
    pairs = cur.merge(adm, on="subject_id", how="inner")
    prior = pairs[(pairs["prev_hadm"] != pairs["hadm_id"]) & (pairs["prev_admit"] < pairs["admittime"])
                  & (pairs["prev_disch"] < pairs["admittime"])]
    n_all = prior.groupby("hadm_id").size()
    n_365 = prior[prior["prev_disch"] >= prior["admittime"] - 365 * DAY].groupby("hadm_id").size()
    gap_days = (cur.set_index("hadm_id")["admittime"] - prior.groupby("hadm_id")["prev_disch"].max()) / DAY

    icu = icustays[["subject_id", "intime", "outtime"]].copy()
    for col in ("intime", "outtime"):
        icu[col] = pd.to_datetime(icu[col])
    ip = cur.merge(icu, on="subject_id", how="inner")
    ip = ip[(ip["intime"] < ip["admittime"]) & (ip["outtime"] < ip["admittime"])
            & (ip["outtime"] >= ip["admittime"] - 365 * DAY)]
    n_icu = ip.groupby("hadm_id").size()

    h = cur["hadm_id"]
    out = pd.DataFrame(index=cohort.index)
    out["prior_adm_365d"] = _bin(h.map(n_365).fillna(0), [0, 1, 2, 4])
    out["prior_adm_all"] = _bin(h.map(n_all).fillna(0), [0, 1, 3, 7])
    gap = h.map(gap_days)
    out["days_since_discharge"] = np.where(gap.isna(), 1, _bin(gap.fillna(0), [30, 90, 365]) + 1)
    out["prior_icu_365d"] = _bin(h.map(n_icu).fillna(0), [0, 1])
    return out


def admission_categoricals(cohort: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Codes for admission_type / admission_location (known at admission; 0 = UNKNOWN)."""
    codes, vocab = [], {}
    for col in ("admission_type", "admission_location"):
        values = cohort[col].astype("object").where(cohort[col].notna(), None)
        cats = sorted({str(v) for v in values if v is not None})
        vocab[col] = ["UNKNOWN", *cats]
        lookup = {c: i + 1 for i, c in enumerate(cats)}
        codes.append(np.array([lookup.get(str(v), 0) if v is not None else 0 for v in values], dtype=np.int64))
    return np.stack(codes, axis=1), vocab
