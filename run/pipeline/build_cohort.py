"""Build the cohort tables (faithful + corrected) and the in-window CXR / note indices.

Outputs (under paths.cohort_dir; credentialed data, never commit them):
    cohort.parquet         one row per admission: times, demographics, flags, labels, splits
    cxr_window.parquet     AP images in the 48 h window (``is_first`` = paper's input image)
    notes_window.parquet   radiology reports in the 48 h window, with text
    cohort_report.md       aggregate counts only
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data_module.cohort import (
    CohortConfig,
    add_labels,
    add_modality_flags,
    assign_splits,
    attach_admission_info,
    load_key_admissions,
    select_cxr_in_window,
    select_notes_in_window,
)
from src.utils.report import df_to_markdown, prevalence_table

logger = logging.getLogger(__name__)

CXR_COLS = ["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime", "ViewCodeSequence_CodeMeaning"]
FAITHFUL_LABELS = ["y_mortality_faithful", "y_los_faithful", "y_readmission_faithful"]
CORRECTED_LABELS = ["y_mortality", "y_los", "y_readmission"]


def _cohort_config(cfg: DictConfig) -> CohortConfig:
    c = cfg.cohort
    return CohortConfig(
        window_hours=float(c.window_hours),
        cxr_view=str(c.cxr_view),
        readmission_days=int(c.readmission_days),
        long_los_hours=float(c.long_los_hours),
        min_los_hours=float(c.min_los_hours),
        seeds=tuple(int(s) for s in c.seeds),
        n_folds=int(c.n_folds),
    )


def _gate(cohort: pd.DataFrame, expected: DictConfig) -> list[str]:
    """Compare the faithful cohort with paper Fig. 3; return mismatch messages."""
    got = cohort["combo"].value_counts().to_dict()
    got["n"] = len(cohort)
    return [f"{k}: expected {int(v)}, got {got.get(k, 0)}" for k, v in expected.items() if got.get(k, 0) != int(v)]


def _split_leak_check(cohort: pd.DataFrame, seeds: tuple[int, ...]) -> list[str]:
    msgs = []
    for s in seeds:
        n_split_per_subject = cohort.groupby("subject_id")[f"split_s{s}"].nunique()
        if (n_split_per_subject > 1).any():
            msgs.append(f"seed {s}: {(n_split_per_subject > 1).sum()} subjects in >1 split")
    return msgs


def _report(cohort: pd.DataFrame, cxr: pd.DataFrame, notes: pd.DataFrame, gate_msgs: list[str]) -> str:
    corr = cohort[cohort["in_corrected"] == 1]
    lines = ["# Cohort report (aggregates only)", ""]
    lines.append("Gate (paper Fig. 3): " + ("PASS" if not gate_msgs else "FAIL - " + "; ".join(gate_msgs)))
    for name, df in [("faithful", cohort), ("corrected", corr)]:
        combos = df["combo"].value_counts().rename_axis("combo").reset_index(name="n")
        lines += ["", f"## {name}: N={len(df)}, subjects={df['subject_id'].nunique()}", "", df_to_markdown(combos)]
    lines += ["", "## Label prevalence", "", "faithful labels on faithful cohort:", ""]
    lines.append(df_to_markdown(prevalence_table(cohort, FAITHFUL_LABELS, "split_s0")))
    lines += ["", "corrected labels on corrected cohort:", ""]
    lines.append(df_to_markdown(prevalence_table(corr, CORRECTED_LABELS, "split_s0")))
    per_adm_note = notes.groupby("hadm_id").size()
    per_adm_cxr = cxr.groupby("hadm_id").size()
    lines += [
        "",
        "## Window contents",
        f"- radiology reports in window: {len(notes)} "
        f"(per admission mean {per_adm_note.mean():.2f}, median {per_adm_note.median():.0f}, max {per_adm_note.max()})",
        f"- AP images in window: {len(cxr)} "
        f"(per admission mean {per_adm_cxr.mean():.2f}, max {per_adm_cxr.max()})",
    ]
    return "\n".join(lines) + "\n"


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    ccfg = _cohort_config(cfg)
    out_dir = Path(cfg.paths.cohort_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mimic = Path(cfg.paths.mimiciv)

    keys = load_key_admissions(pd.read_csv(cfg.paths.haim_key_ids))
    logger.info("key admissions: %d", len(keys))
    admissions = pd.read_csv(mimic / "core" / "admissions.csv")
    patients = pd.read_csv(mimic / "core" / "patients.csv")
    cohort = attach_admission_info(keys, admissions, patients)

    meta = pd.read_csv(Path(cfg.paths.cxr_root) / "mimic-cxr-2.0.0-metadata.csv", usecols=CXR_COLS)
    cxr = select_cxr_in_window(meta, cohort, ccfg)
    logger.info("CXR images in window: %d (admissions %d)", len(cxr), cxr["hadm_id"].nunique())

    notes_raw = pd.read_csv(Path(cfg.paths.note_root) / "radiology.csv")
    notes = select_notes_in_window(notes_raw, cohort, ccfg)
    del notes_raw
    logger.info("notes in window: %d (admissions %d)", len(notes), notes["hadm_id"].nunique())

    cohort = add_modality_flags(cohort, cxr, notes)
    cohort = add_labels(cohort, admissions, ccfg)
    cohort = assign_splits(cohort, ccfg)

    gate_msgs = _gate(cohort, cfg.cohort.expected_faithful)
    leak_msgs = _split_leak_check(cohort, ccfg.seeds)
    cohort.to_parquet(out_dir / "cohort.parquet", index=False)
    cxr.to_parquet(out_dir / "cxr_window.parquet", index=False)
    notes.to_parquet(out_dir / "notes_window.parquet", index=False)
    (out_dir / "cohort_report.md").write_text(_report(cohort, cxr, notes, gate_msgs))
    logger.info("wrote %s", out_dir)

    if gate_msgs or leak_msgs:
        raise RuntimeError("cohort checks failed: " + "; ".join(gate_msgs + leak_msgs))
    logger.info("gate PASS; no subject appears in two splits")


if __name__ == "__main__":
    main()
