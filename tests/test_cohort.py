"""Unit tests for cohort rules on small synthetic frames (no MIMIC data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_module.cohort import (
    CohortConfig,
    add_labels,
    add_modality_flags,
    assign_splits,
    parse_cxr_time,
    readmission_any,
    readmission_within_table,
    select_cxr_in_window,
    select_notes_in_window,
)

CFG = CohortConfig()
T0 = pd.Timestamp("2150-01-01 00:00:00")


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 11, 20],
            "admittime": [T0, T0 + pd.Timedelta(days=20), T0],
            "dischtime": [T0 + pd.Timedelta(days=10), T0 + pd.Timedelta(days=22), T0 + pd.Timedelta(hours=30)],
        }
    )


def test_parse_cxr_time_zero_pads_hours() -> None:
    t = parse_cxr_time(pd.Series([21500101]), pd.Series([40024.5]))
    assert t.iloc[0] == pd.Timestamp("2150-01-01 04:00:24.5")


def test_cxr_window_rules_and_first_image() -> None:
    meta = pd.DataFrame(
        {
            "dicom_id": ["a", "b", "c", "d", "e"],
            "subject_id": [1, 1, 1, 1, 2],
            "study_id": [100, 101, 102, 103, 200],
            # a: 1 h after admit (keep, first); b: 2 h (keep); c: 49 h (drop); d: lateral (drop)
            "StudyDate": [21500101, 21500101, 21500103, 21500101, 21500101],
            "StudyTime": [10000.0, 20000.0, 10000.0, 30000.0, 10000.0],
            "ViewCodeSequence_CodeMeaning": ["antero-posterior"] * 3 + ["lateral", "antero-posterior"],
        }
    )
    out = select_cxr_in_window(meta, _cohort(), CFG)
    adm10 = out[out.hadm_id == 10]
    assert list(adm10.dicom_id) == ["a", "b"]
    assert list(adm10.is_first) == [True, False]
    assert adm10.path.iloc[0] == "files/p1/p1/s100/a.jpg"
    assert set(out[out.hadm_id == 20].dicom_id) == {"e"}


def test_note_window_bounds_and_dropna() -> None:
    notes = pd.DataFrame(
        {
            "note_id": ["n0", "n1", "n2", "n3"],
            "subject_id": [1, 1, 1, 2],
            "hadm_id": [10, 10, 10, 20],
            "note_type": ["RR"] * 4,
            "note_seq": [1, 2, 3, 4],
            # n0 at admit (delta 0 -> drop), n1 at 48 h (keep), n2 missing storetime (drop),
            # n3 after discharge at 31 h (drop)
            "charttime": [T0, T0 + pd.Timedelta(hours=48), T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=31)],
            "storetime": [T0, T0, None, T0],
            "text": ["x", "y", "z", "w"],
        }
    )
    out = select_notes_in_window(notes, _cohort(), CFG)
    assert list(out.note_id) == ["n1"]


def test_modality_flags_and_combo_names() -> None:
    cxr = pd.DataFrame({"hadm_id": [10, 10]})
    notes = pd.DataFrame({"hadm_id": [10, 11]})
    out = add_modality_flags(_cohort(), cxr, notes)
    assert list(out.combo) == ["ETI", "ET", "E"]
    assert list(out.n_cxr) == [2, 0, 0]


def test_readmission_definitions() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [10, 11, 20, 30],
            "admittime": [T0, T0 + pd.Timedelta(days=20), T0, T0],
            "dischtime": [T0 + pd.Timedelta(days=10), T0 + pd.Timedelta(days=22), T0 + pd.Timedelta(days=2),
                          T0 + pd.Timedelta(days=2)],
        }
    )
    all_adm = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [10, 11, 21, 31],
            # subject 2: next admission 31 days after discharge -> 0; subject 3: overlapping -> 0
            "admittime": [T0, T0 + pd.Timedelta(days=20), T0 + pd.Timedelta(days=33), T0 + pd.Timedelta(days=1)],
        }
    )
    assert list(readmission_any(df, all_adm, 30)) == [1, 0, 0, 0]
    assert list(readmission_within_table(df, 30)) == [1, 0, 0, 0]


def test_old_readmission_counts_overlaps() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 11],
            "admittime": [T0, T0 + pd.Timedelta(days=1)],
            "dischtime": [T0 + pd.Timedelta(days=5), T0 + pd.Timedelta(days=6)],
        }
    )
    assert list(readmission_within_table(df, 30)) == [1, 0]


def test_corrected_membership_and_death_exclusion() -> None:
    df = _cohort().assign(
        deathtime=[pd.NaT, pd.NaT, T0 + pd.Timedelta(hours=30)],
        discharge_location=["HOME", "HOME", "DIED"],
        hospital_expire_flag=[0, 0, 1],
    )
    df["los_h"] = (df.dischtime - df.admittime) / pd.Timedelta(hours=1)
    out = add_labels(df, df[["subject_id", "hadm_id", "admittime"]], CFG)
    assert list(out.in_corrected) == [1, 1, 0]
    assert list(out.y_mortality) == [0, 0, 1]
    assert list(out.y_mortality_faithful) == [0, 0, 1]
    assert np.isnan(out.y_readmission.iloc[2])
    assert list(out.y_los) == [1, 0, 0]


def test_splits_are_patient_level_and_roughly_80_10_10() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(
        {
            "subject_id": rng.integers(0, 2500, n),
            "combo": rng.choice(["E", "ET", "EI", "ETI"], n),
            "y_mortality_faithful": rng.integers(0, 2, n),
        }
    )
    out = assign_splits(df, CohortConfig(seeds=(0, 1)))
    for s in (0, 1):
        assert out.groupby("subject_id")[f"split_s{s}"].nunique().max() == 1
        frac = out[f"split_s{s}"].value_counts(normalize=True)
        assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["test"] - 0.1) < 0.03
    assert (out.split_s0 != out.split_s1).any()
