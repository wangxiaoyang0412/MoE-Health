"""DuckDB EHR extraction on synthetic CSVs that mimic the MIMIC-IV v1.0 layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_module.ehr_extract import (
    assemble_continuous,
    assemble_interval,
    encode_static,
    extract_long_tables,
)
from src.data_module.ehr_spec import CONTINUOUS_VARS, INTERVAL_VARS

T0 = pd.Timestamp("2150-01-01 00:00:00")


def _ts(hours: float) -> str:
    return str(T0 + pd.Timedelta(hours=hours))


def _write_fake_mimic(root: Path) -> Path:
    (root / "icu").mkdir(parents=True)
    (root / "hosp").mkdir()
    pd.DataFrame(
        {
            "subject_id": [1] * 6 + [2],
            "hadm_id": [1] * 6 + [2],
            "stay_id": [9] * 7,
            "charttime": [_ts(0.5), _ts(0.7), _ts(1.0), _ts(48.0), _ts(49.0), _ts(-1.0), _ts(3.2)],
            "itemid": [220045, 220045, 220045, 220045, 220045, 220045, 220277],
            "valuenum": [80.0, 90.0, 400.0, 70.0, 75.0, 60.0, 97.0],
        }
    ).to_csv(root / "icu" / "chartevents.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [1, 1],
            "itemid": [50931, 51981],  # blood glucose (kept), urine glucose (not mapped)
            "charttime": [_ts(2.2), _ts(2.5)],
            "valuenum": [120.0, 50.0],
        }
    ).to_csv(root / "hosp" / "labevents.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [1, 2],
            "itemid": [229351, 224385],  # Foley from before admission; instantaneous intubation
            "starttime": [_ts(-5.0), _ts(10.2)],
            "endtime": [_ts(1.5), _ts(10.2)],
            "statusdescription": ["FinishedRunning", "FinishedRunning"],
        }
    ).to_csv(root / "icu" / "procedureevents.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [2, 2],
            "hadm_id": [2, 2],
            "itemid": [221906, 221906],
            "starttime": [_ts(3.0), _ts(20.0)],
            "endtime": [_ts(5.0), _ts(22.0)],
            "statusdescription": ["FinishedRunning", "Rewritten"],
        }
    ).to_csv(root / "icu" / "inputevents.csv", index=False)
    cohort = pd.DataFrame({"hadm_id": [1, 2], "admittime": [T0, T0]})
    cohort_path = root / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    return cohort_path


def test_extract_and_assemble(tmp_path: Path) -> None:
    cohort_path = _write_fake_mimic(tmp_path / "mimic")
    out = tmp_path / "long"
    extract_long_tables(tmp_path / "mimic", cohort_path, out, threads=2)
    order = np.array([1, 2])
    cont = pd.concat([pd.read_parquet(out / "chart_long.parquet"), pd.read_parquet(out / "lab_long.parquet")])
    values = assemble_continuous(cont, order)
    active = assemble_interval(pd.read_parquet(out / "interval_long.parquet"), order)

    hr = CONTINUOUS_VARS.index("heart_rate")
    assert values[0, 0, hr] == 85.0  # mean of 80 and 90; 400 dropped as implausible
    assert values[0, 47, hr] == 70.0  # t = 48 h is inside the window -> last hour
    assert np.isnan(values[0, 1, hr])  # 400 at 1.0 h dropped
    assert np.sum(~np.isnan(values[0, :, hr])) == 2  # 49 h and -1 h are outside
    assert values[1, 3, CONTINUOUS_VARS.index("spo2")] == 97.0
    assert values[0, 2, CONTINUOUS_VARS.index("glucose")] == 120.0

    foley = INTERVAL_VARS.index("foley_catheter")
    assert list(np.nonzero(active[0, :, foley])[0]) == [0, 1]
    assert list(np.nonzero(active[1, :, INTERVAL_VARS.index("intubation")])[0]) == [10]
    norepi = INTERVAL_VARS.index("norepinephrine")
    assert list(np.nonzero(active[1, :, norepi])[0]) == [3, 4]  # rewritten order ignored


def test_encode_static_unknown_bucket() -> None:
    df = pd.DataFrame(
        {
            "age": [60.0, 70.0],
            "gender": ["M", "F"],
            "ethnicity": ["WHITE", None],
            "insurance": ["Medicare", "Other"],
            "marital_status": [None, "SINGLE"],
            "language": ["ENGLISH", "?"],
        }
    )
    age, cat, vocab = encode_static(df)
    assert age.shape == (2, 1) and cat.shape == (2, 5)
    assert cat[1, 1] == 0 and vocab["ethnicity"][0] == "UNKNOWN"
    assert vocab["gender"] == ["UNKNOWN", "F", "M"]


def test_extended_v2_transforms_and_urine(tmp_path: Path) -> None:
    from src.data_module.ehr_spec import continuous_vars

    root = tmp_path / "mimic"
    cohort_path = _write_fake_mimic(root)
    extra = pd.DataFrame({
        "subject_id": [1, 1, 1], "hadm_id": [1, 1, 1], "stay_id": [9, 9, 9],
        "charttime": [_ts(5.1), _ts(5.2), _ts(6.5)],
        "itemid": [223761, 223835, 223835],  # 98.6 F, FiO2 given as fraction, FiO2 given as %
        "valuenum": [98.6, 0.5, 40.0],
    })
    chart = pd.concat([pd.read_csv(root / "icu" / "chartevents.csv"), extra], ignore_index=True)
    chart.to_csv(root / "icu" / "chartevents.csv", index=False)
    pd.DataFrame({
        "subject_id": [1, 1, 1], "hadm_id": [1, 1, 1], "stay_id": [9, 9, 9],
        "charttime": [_ts(2.1), _ts(2.9), _ts(3.5)], "itemid": [226559, 226560, 226559],
        "value": [100.0, 50.0, 7000.0],  # 7000 mL is implausible and dropped
    }).to_csv(root / "icu" / "outputevents.csv", index=False)
    out = tmp_path / "long_v2"
    extract_long_tables(root, cohort_path, out, threads=2, version="v2")
    variables = continuous_vars("v2")
    long = pd.concat([pd.read_parquet(out / f) for f in ("chart_long.parquet", "lab_long.parquet",
                                                           "output_long.parquet")])
    values = assemble_continuous(long, np.array([1, 2]), variables)
    assert np.isclose(values[0, 5, variables.index("temperature")], 37.0, atol=1e-4)
    assert np.isclose(values[0, 5, variables.index("fio2")], 50.0)
    assert np.isclose(values[0, 6, variables.index("fio2")], 40.0)
    assert values[0, 2, variables.index("urine_output")] == 150.0
    assert np.isnan(values[0, 3, variables.index("urine_output")])
    assert values[0, 0, variables.index("heart_rate")] == 85.0  # v1 variables unchanged
