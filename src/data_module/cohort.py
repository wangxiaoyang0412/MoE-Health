"""Cohort, modality availability, labels and splits for MoE-Health.

The cohort is every hospital admission in the HAIM key-id table (ICU stays of
patients with at least one MIMIC-CXR study), built on MIMIC-IV v1.0.

Two label sets are produced:

- ``*_faithful``: the definitions used in the paper
  (death = discharge_location DIED, readmission searched inside the cohort table).
- corrected (no suffix): in-hospital mortality from ``hospital_expire_flag``,
  readmission searched in the full admissions table with in-hospital deaths
  excluded, and only admissions still in hospital at the 48 h prediction time
  (``in_corrected``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

__all__ = [
    "MODALITY_LETTERS",
    "CohortConfig",
    "add_labels",
    "add_modality_flags",
    "assign_splits",
    "attach_admission_info",
    "load_key_admissions",
    "parse_cxr_time",
    "readmission_any",
    "readmission_within_table",
    "select_cxr_in_window",
    "select_notes_in_window",
]

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
# Modality order used across the project: EHR, clinical notes (text), CXR (image).
MODALITY_LETTERS = ("E", "T", "I")


@dataclass(frozen=True)
class CohortConfig:
    """Cohort construction parameters (mirrors ``run/conf/pipeline.yaml``)."""

    window_hours: float = 48.0
    cxr_view: str = "antero-posterior"
    readmission_days: int = 30
    long_los_hours: float = 168.0
    min_los_hours: float = 48.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_folds: int = 10


def load_key_admissions(key_ids: pd.DataFrame) -> pd.DataFrame:
    """Unique (subject_id, hadm_id) pairs from the HAIM key-id table."""
    keys = key_ids[["subject_id", "hadm_id"]].dropna().astype("int64").drop_duplicates()
    return keys.sort_values(["subject_id", "hadm_id"]).reset_index(drop=True)


def attach_admission_info(
    keys: pd.DataFrame, admissions: pd.DataFrame, patients: pd.DataFrame
) -> pd.DataFrame:
    """Join admission times, outcome fields and demographics onto the key admissions."""
    adm = admissions.copy()
    for col in ("admittime", "dischtime", "deathtime"):
        adm[col] = pd.to_datetime(adm[col])
    df = keys.merge(adm, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    if df["admittime"].isna().any():
        missing = int(df["admittime"].isna().sum())
        raise ValueError(f"{missing} key admissions are absent from admissions.csv")
    pat = patients[["subject_id", "gender", "anchor_age", "anchor_year"]]
    df = df.merge(pat, on="subject_id", how="left", validate="many_to_one")
    df["los_h"] = (df["dischtime"] - df["admittime"]) / HOUR
    # Age at admission (anchor_age is the age in anchor_year).
    df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    return df


def parse_cxr_time(study_date: pd.Series, study_time: pd.Series) -> pd.Series:
    """Combine MIMIC-CXR StudyDate (YYYYMMDD) and StudyTime (HHMMSS.f) into timestamps."""
    date = pd.to_datetime(study_date.astype("int64").astype(str), format="%Y%m%d")
    t = study_time.astype(float)
    hours = (t // 10000).astype("int64")
    minutes = ((t % 10000) // 100).astype("int64")
    seconds = t % 100
    return (
        date
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )


def select_cxr_in_window(
    metadata: pd.DataFrame, cohort: pd.DataFrame, cfg: CohortConfig
) -> pd.DataFrame:
    """AP images taken after admission, before discharge and within the window.

    Window rule of the paper: admittime < t < dischtime and t - admittime < window.
    ``is_first`` marks the earliest image per admission (the paper's input image).
    """
    meta = metadata.loc[metadata["ViewCodeSequence_CodeMeaning"] == cfg.cxr_view].copy()
    meta[["subject_id", "study_id"]] = meta[["subject_id", "study_id"]].astype("int64")
    meta["cxrtime"] = parse_cxr_time(meta["StudyDate"], meta["StudyTime"])
    times = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]]
    m = meta.merge(times, on="subject_id", how="inner")
    m["delta_h"] = (m["cxrtime"] - m["admittime"]) / HOUR
    keep = (
        (m["cxrtime"] > m["admittime"])
        & (m["cxrtime"] < m["dischtime"])
        & (m["delta_h"] < cfg.window_hours)
    )
    m = m.loc[keep].sort_values(["hadm_id", "cxrtime", "dicom_id"])
    m["is_first"] = ~m.duplicated("hadm_id")
    m["path"] = (
        "files/p" + m["subject_id"].astype(str).str[:2]
        + "/p" + m["subject_id"].astype(str)
        + "/s" + m["study_id"].astype(str)
        + "/" + m["dicom_id"] + ".jpg"
    )
    cols = ["subject_id", "hadm_id", "dicom_id", "study_id", "cxrtime", "delta_h", "is_first", "path"]
    return m[cols].reset_index(drop=True)


def select_notes_in_window(
    notes: pd.DataFrame, cohort: pd.DataFrame, cfg: CohortConfig
) -> pd.DataFrame:
    """Radiology reports with 0 < t - admittime <= window and t < dischtime.

    Rows with any missing field are dropped first, as in the paper's pipeline.
    """
    n = notes.dropna().copy()
    n[["subject_id", "hadm_id"]] = n[["subject_id", "hadm_id"]].astype("int64")
    n["charttime"] = pd.to_datetime(n["charttime"])
    times = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]]
    n = n.merge(times, on=["subject_id", "hadm_id"], how="inner")
    n["delta_h"] = (n["charttime"] - n["admittime"]) / HOUR
    keep = (n["delta_h"] > 0) & (n["delta_h"] <= cfg.window_hours) & (n["charttime"] < n["dischtime"])
    n = n.loc[keep].sort_values(["hadm_id", "charttime", "note_id"])
    cols = ["subject_id", "hadm_id", "note_id", "charttime", "delta_h", "text"]
    return n[[c for c in cols if c in n.columns]].reset_index(drop=True)


def add_modality_flags(
    cohort: pd.DataFrame, cxr: pd.DataFrame, notes: pd.DataFrame
) -> pd.DataFrame:
    """Add has_*/n_* availability columns and the combination name (e.g. 'ETI')."""
    df = cohort.copy()
    n_cxr = cxr.groupby("hadm_id").size()
    n_note = notes.groupby("hadm_id").size()
    df["n_cxr"] = df["hadm_id"].map(n_cxr).fillna(0).astype("int64")
    df["n_note"] = df["hadm_id"].map(n_note).fillna(0).astype("int64")
    df["has_ehr"] = 1  # demographics always exist; the paper reports EHR availability as 100%
    df["has_note"] = (df["n_note"] > 0).astype("int64")
    df["has_cxr"] = (df["n_cxr"] > 0).astype("int64")
    flags = df[["has_ehr", "has_note", "has_cxr"]].to_numpy()
    df["combo"] = ["".join(L for L, f in zip(MODALITY_LETTERS, row, strict=True) if f) for row in flags]
    return df


def readmission_within_table(df: pd.DataFrame, days: int) -> pd.Series:
    """Readmission as defined in the paper: next admission *inside this table* within ``days``.

    gap = next admittime - dischtime, label 1 if gap <= days (negative gaps from
    overlapping stays also count), last admission -> 0.
    """
    d = df[["subject_id", "admittime", "dischtime"]].sort_values(["subject_id", "admittime"])
    nxt = d.groupby("subject_id")["admittime"].shift(-1)
    gap = nxt - d["dischtime"]
    label = (gap.notna() & (gap <= days * DAY)).astype("int64")
    return label.reindex(df.index)


def readmission_any(df: pd.DataFrame, all_admissions: pd.DataFrame, days: int) -> pd.Series:
    """1 if the patient has any admission starting in (dischtime, dischtime + days]."""
    allx = all_admissions[["subject_id", "hadm_id", "admittime"]].copy()
    allx["admittime"] = pd.to_datetime(allx["admittime"])
    allx = allx.rename(columns={"hadm_id": "next_hadm", "admittime": "next_admit"})
    pairs = df[["subject_id", "hadm_id", "dischtime"]].merge(allx, on="subject_id", how="inner")
    gap = pairs["next_admit"] - pairs["dischtime"]
    hit = pairs.loc[(pairs["next_hadm"] != pairs["hadm_id"]) & (gap > pd.Timedelta(0)) & (gap <= days * DAY)]
    return df["hadm_id"].isin(set(hit["hadm_id"])).astype("int64")


def add_labels(df: pd.DataFrame, all_admissions: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Add faithful and corrected labels plus the corrected-cohort membership flag."""
    out = df.copy()
    out["y_mortality_faithful"] = (out["discharge_location"] == "DIED").astype("int64")
    out["y_los_faithful"] = (out["los_h"] > cfg.long_los_hours).astype("int64")
    out["y_readmission_faithful"] = readmission_within_table(out, cfg.readmission_days)

    died_early = out["deathtime"].notna() & (
        (out["deathtime"] - out["admittime"]) / HOUR < cfg.min_los_hours
    )
    out["in_corrected"] = ((out["los_h"] >= cfg.min_los_hours) & ~died_early).astype("int64")
    out["y_mortality"] = out["hospital_expire_flag"].fillna(0).astype("int64")
    out["y_los"] = (out["los_h"] > cfg.long_los_hours).astype("int64")
    readm = readmission_any(out, all_admissions, cfg.readmission_days).astype("float64")
    readm[out["y_mortality"] == 1] = np.nan  # the dead cannot be readmitted
    out["y_readmission"] = readm
    return out


def assign_splits(df: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Patient-level stratified 80/10/10 splits, one column ``split_s{seed}`` per seed.

    Strata = modality combination x faithful mortality label. Fold 0 is test, fold 1
    is validation, the rest is train, so no subject appears in two splits. The
    corrected cohort is a row subset and inherits the same subject assignment.
    """
    out = df.copy()
    strata = pd.factorize(out["combo"] + "_" + out["y_mortality_faithful"].astype(str))[0]
    groups = out["subject_id"].to_numpy()
    for seed in cfg.seeds:
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
        fold = np.full(len(out), -1, dtype=np.int64)
        for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(out)), strata, groups)):
            fold[test_idx] = k
        out[f"split_s{seed}"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    return out
