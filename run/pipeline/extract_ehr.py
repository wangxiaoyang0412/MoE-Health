"""Extract 48 h hourly EHR arrays aligned to cohort.parquet row order.

Outputs (paths.feature_dir):
    ehr_long/*.parquet   long tables (DuckDB)
    ehr_values.npy       [N, 48, V] float32, NaN = unobserved (chart + lab)
    ehr_active.npy       [N, 48, P] uint8 (procedures + medications)
    static_num.npy       [N, 1] float32 (age at admission)
    static_cat.npy       [N, C] int64 (0 = UNKNOWN)
    hadm_order.npy       [N] hadm_id order of all arrays
    ehr_meta.json        variable names + categorical vocabularies
    ehr_report.md        aggregate coverage only
With ehr.version=v2 (extended variable set) only ehr_values_v2.npy, ehr_meta_v2.json and
ehr_report_v2.md are written; the shared v1 arrays are left untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_module.ehr_extract import (
    assemble_continuous,
    assemble_interval,
    encode_static,
    extract_long_tables,
)
from src.data_module.ehr_spec import INTERVAL_VARS, STATIC_CATEGORICAL, continuous_vars
from src.utils.report import df_to_markdown

logger = logging.getLogger(__name__)


def _coverage_report(values: np.ndarray, active: np.ndarray, in_corrected: np.ndarray, variables: list[str]) -> str:
    seen = ~np.isnan(values)
    rows = []
    for j, var in enumerate(variables):
        any_obs = seen[:, :, j].any(axis=1)
        rows.append({"variable": var, "adm with >=1 obs": float(any_obs.mean()),
                     "obs hours / adm": float(seen[:, :, j].sum(axis=1).mean())})
    for j, var in enumerate(INTERVAL_VARS):
        rows.append({"variable": var, "adm with >=1 obs": float(active[:, :, j].any(axis=1).mean()),
                     "obs hours / adm": float(active[:, :, j].sum(axis=1).mean())})
    no_ts = ~(seen.any(axis=(1, 2)) | active.any(axis=(1, 2)))
    lines = [
        "# EHR feature report (aggregates only)",
        "",
        f"- admissions: {len(values)}; with no time-series observation in 48 h: {int(no_ts.sum())} "
        f"(corrected cohort: {int(no_ts[in_corrected == 1].sum())})",
        "",
        df_to_markdown(pd.DataFrame(rows)),
    ]
    return "\n".join(lines) + "\n"


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    version = str(cfg.ehr.get("version", "v1"))
    suffix = "" if version == "v1" else f"_{version}"
    cohort_path = Path(cfg.paths.cohort_dir) / "cohort.parquet"
    feat = Path(cfg.paths.feature_dir)
    long_dir = feat / f"ehr_long{suffix}"
    if cfg.ehr.overwrite or not (long_dir / "interval_long.parquet").exists():
        extract_long_tables(Path(cfg.paths.mimiciv), cohort_path, long_dir, threads=int(cfg.ehr.threads),
                            version=version)

    cohort = pd.read_parquet(cohort_path)
    hadm_order = cohort["hadm_id"].to_numpy()
    variables = continuous_vars(version)
    parts = ["chart_long.parquet", "lab_long.parquet"] + (["output_long.parquet"] if version == "v2" else [])
    cont = pd.concat([pd.read_parquet(long_dir / f) for f in parts], ignore_index=True)
    values = assemble_continuous(cont, hadm_order, variables)
    active = assemble_interval(pd.read_parquet(long_dir / "interval_long.parquet"), hadm_order)
    age, cat, vocab = encode_static(cohort)

    np.save(feat / f"ehr_values{suffix}.npy", values)
    if version == "v1":
        np.save(feat / "ehr_active.npy", active)
        np.save(feat / "static_num.npy", age)
        np.save(feat / "static_cat.npy", cat)
        np.save(feat / "hadm_order.npy", hadm_order)
    elif not np.array_equal(np.load(feat / "hadm_order.npy"), hadm_order):
        raise ValueError("v1 arrays are not aligned with cohort.parquet; run version v1 first")
    meta = {"version": version, "continuous_vars": variables, "interval_vars": INTERVAL_VARS,
            "static_categorical": STATIC_CATEGORICAL, "vocab": vocab}
    (feat / f"ehr_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    report = _coverage_report(values, active, cohort["in_corrected"].to_numpy(), variables)
    (feat / f"ehr_report{suffix}.md").write_text(report)
    logger.info("EHR %s arrays: values %s, active %s, static_cat %s", version, values.shape, active.shape, cat.shape)


if __name__ == "__main__":
    main()
