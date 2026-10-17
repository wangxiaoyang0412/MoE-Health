"""Prior-utilisation features on synthetic admissions (only stays that ended before admittime count)."""

from __future__ import annotations

import pandas as pd

from src.data_module.history import HISTORY_VOCAB, admission_categoricals, prior_utilization

T0 = pd.Timestamp("2150-06-01")


def test_prior_utilization_counts_only_past_stays() -> None:
    d = pd.Timedelta(days=1)
    cohort = pd.DataFrame({"subject_id": [1, 2, 3], "hadm_id": [10, 20, 30], "admittime": [T0, T0, T0]})
    admissions = pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 2, 3],
        "hadm_id": [10, 11, 12, 13, 20, 31],
        "admittime": [T0, T0 - 400 * d, T0 - 60 * d, T0 + 5 * d, T0, T0 - 3 * d],
        # 11: >365 d ago, 12: 50 d ago, 13: future (ignored); subject 3: still in hospital at admittime (ignored)
        "dischtime": [T0 + 3 * d, T0 - 390 * d, T0 - 50 * d, T0 + 9 * d, T0 + 2 * d, T0 + 1 * d],
    })
    icustays = pd.DataFrame({"subject_id": [1, 1], "intime": [T0 - 59 * d, T0 + 1 * d],
                             "outtime": [T0 - 55 * d, T0 + 2 * d]})
    out = prior_utilization(cohort, admissions, icustays)
    label = {c: [HISTORY_VOCAB[c][v] for v in out[c]] for c in HISTORY_VOCAB}
    assert label["prior_adm_365d"] == ["1", "0", "0"]
    assert label["prior_adm_all"] == ["2-3", "0", "0"]
    assert label["days_since_discharge"] == ["31-90", "none", "none"]
    assert label["prior_icu_365d"] == ["1", "0", "0"]


def test_prior_utilization_ignores_future_stays_with_corrupted_times() -> None:
    d = pd.Timedelta(days=1)
    cohort = pd.DataFrame({"subject_id": [1], "hadm_id": [10], "admittime": [T0]})
    # admission 11 starts after the index admission but has a discharge time before it
    admissions = pd.DataFrame({"subject_id": [1, 1], "hadm_id": [10, 11], "admittime": [T0, T0 + 2 * d],
                               "dischtime": [T0 + 3 * d, T0 - 1 * d]})
    icustays = pd.DataFrame({"subject_id": [1], "intime": [T0 + 1 * d], "outtime": [T0 - 1 * d]})
    out = prior_utilization(cohort, admissions, icustays)
    label = {c: [HISTORY_VOCAB[c][v] for v in out[c]] for c in HISTORY_VOCAB}
    assert label["prior_adm_all"] == ["0"]
    assert label["days_since_discharge"] == ["none"]
    assert label["prior_icu_365d"] == ["0"]


def test_admission_categoricals() -> None:
    df = pd.DataFrame({"admission_type": ["URGENT", None], "admission_location": ["EMERGENCY ROOM", "TRANSFER"]})
    codes, vocab = admission_categoricals(df)
    assert codes.shape == (2, 2) and codes[1, 0] == 0
    assert vocab["admission_type"] == ["UNKNOWN", "URGENT"]
