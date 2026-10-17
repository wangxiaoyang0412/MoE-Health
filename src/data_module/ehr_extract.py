"""DuckDB extraction of 48 h hourly EHR tables and assembly into dense arrays.

Long tables (one row per admission x variable x hour):
    continuous: mean valuenum of chart + lab events with 0 < t - admittime <= 48 h
                (v2 adds unit-normalised chart items and hourly summed urine output)
    interval:   procedures / medications active in the hour, i.e. start < hour end and
                max(end, start + 1 s) > hour start (instantaneous events count once)
Hour index = floor((t - admittime) / 1 h), with t = 48 h mapped to hour 47.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data_module.ehr_spec import (
    CONTINUOUS_VARS,
    INTERVAL_VARS,
    MED_ITEMS,
    OUTPUT_ITEMS,
    PROCEDURE_ITEMS,
    STATIC_CATEGORICAL,
    chart_rows,
    lab_items,
)

logger = logging.getLogger(__name__)

__all__ = [
    "assemble_continuous",
    "assemble_interval",
    "encode_static",
    "extract_long_tables",
]

N_HOURS = 48
HOUR_EXPR = "LEAST(CAST(floor((epoch({t}) - epoch(c.admittime)) / 3600.0) AS INTEGER), 47)"
WINDOW = "{t} > c.admittime AND {t} <= c.admittime + INTERVAL 48 HOUR"


def _values_sql(rows: list[tuple]) -> str:
    return ", ".join("(" + ", ".join(repr(x) for x in r) + ")" for r in rows)


def _register_maps(con: duckdb.DuckDBPyConnection, version: str) -> None:
    chart = chart_rows(version)
    lab = [(i, v) for v, ids in lab_items(version).items() for i in ids]
    proc = [(i, v) for v, ids in PROCEDURE_ITEMS.items() for i in ids]
    med = [(i, v) for v, ids in MED_ITEMS.items() for i in ids]
    out = [(i, v) for v, ids in OUTPUT_ITEMS.items() for i in ids]
    con.execute(f"CREATE TEMP TABLE chart_map AS SELECT * FROM (VALUES {_values_sql(chart)}) "
                "t(itemid, var, lo, hi, transform)")
    con.execute(f"CREATE TEMP TABLE out_map AS SELECT * FROM (VALUES {_values_sql(out)}) t(itemid, var)")
    con.execute(f"CREATE TEMP TABLE lab_map AS SELECT * FROM (VALUES {_values_sql(lab)}) t(itemid, var)")
    con.execute(f"CREATE TEMP TABLE proc_map AS SELECT * FROM (VALUES {_values_sql(proc)}) t(itemid, var)")
    con.execute(f"CREATE TEMP TABLE med_map AS SELECT * FROM (VALUES {_values_sql(med)}) t(itemid, var)")


def _csv(path: Path, types: dict[str, str]) -> str:
    """Lenient RFC-4180 reader: some free-text ``value`` fields contain unbalanced quotes.

    Unparseable lines are skipped and recorded in DuckDB's reject tables (counted in the log).
    """
    spec = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    return (
        f"read_csv('{path}', header=true, types={{{spec}}}, quote='\"', escape='\"', "
        "strict_mode=false, ignore_errors=true, store_rejects=true)"
    )


def _log_rejects(con: duckdb.DuckDBPyConnection, table: str) -> None:
    try:
        row = con.execute("SELECT count(*) FROM reject_errors").fetchone()
        n = row[0] if row else 0
    except duckdb.CatalogException:
        n = 0
    logger.info("%s: %d rejected CSV lines so far (skipped)", table, n)


def extract_long_tables(
    mimic_dir: Path, cohort_path: Path, out_dir: Path, threads: int = 64, version: str = "v1"
) -> None:
    """Write chart/lab/interval (+ v2 output) long tables as parquet into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute(f"CREATE TEMP TABLE c AS SELECT hadm_id, admittime FROM read_parquet('{cohort_path}')")
    _register_maps(con, version)

    ts = {"hadm_id": "BIGINT", "itemid": "BIGINT", "charttime": "TIMESTAMP", "valuenum": "DOUBLE"}
    chart_src = _csv(mimic_dir / "icu" / "chartevents.csv", ts)
    con.execute(f"""
        COPY (
            SELECT hadm_id, var, hour, avg(v) AS value FROM (
                SELECT e.hadm_id, m.var, m.lo, m.hi, {HOUR_EXPR.format(t='e.charttime')} AS hour,
                       CASE m.transform
                           WHEN 'f2c' THEN (e.valuenum - 32) * 5.0 / 9.0
                           WHEN 'frac2pct' THEN CASE WHEN e.valuenum <= 1 THEN e.valuenum * 100 ELSE e.valuenum END
                           ELSE e.valuenum END AS v
                FROM {chart_src} e JOIN c ON e.hadm_id = c.hadm_id JOIN chart_map m ON e.itemid = m.itemid
                WHERE e.valuenum IS NOT NULL AND {WINDOW.format(t='e.charttime')}
            ) WHERE v > lo AND v <= hi
            GROUP BY ALL
        ) TO '{out_dir / "chart_long.parquet"}' (FORMAT parquet)""")
    _log_rejects(con, "chartevents")

    if version == "v2":
        out_src = _csv(mimic_dir / "icu" / "outputevents.csv",
                       {"hadm_id": "BIGINT", "itemid": "BIGINT", "charttime": "TIMESTAMP", "value": "DOUBLE"})
        con.execute(f"""
            COPY (
                SELECT e.hadm_id, m.var, {HOUR_EXPR.format(t='e.charttime')} AS hour, sum(e.value) AS value
                FROM {out_src} e JOIN c ON e.hadm_id = c.hadm_id JOIN out_map m ON e.itemid = m.itemid
                WHERE e.value IS NOT NULL AND e.value >= 0 AND e.value < 5000 AND {WINDOW.format(t='e.charttime')}
                GROUP BY ALL
            ) TO '{out_dir / "output_long.parquet"}' (FORMAT parquet)""")
        _log_rejects(con, "outputevents")

    lab_src = _csv(mimic_dir / "hosp" / "labevents.csv", ts)
    con.execute(f"""
        COPY (
            SELECT e.hadm_id, m.var, {HOUR_EXPR.format(t='e.charttime')} AS hour, avg(e.valuenum) AS value
            FROM {lab_src} e JOIN c ON e.hadm_id = c.hadm_id JOIN lab_map m ON e.itemid = m.itemid
            WHERE e.valuenum IS NOT NULL AND {WINDOW.format(t='e.charttime')}
            GROUP BY ALL
        ) TO '{out_dir / "lab_long.parquet"}' (FORMAT parquet)""")
    _log_rejects(con, "labevents")

    iv = {"hadm_id": "BIGINT", "itemid": "BIGINT", "starttime": "TIMESTAMP", "endtime": "TIMESTAMP",
          "statusdescription": "VARCHAR"}
    parts = []
    for table, mapping, rewritten_filter in [
        ("procedureevents", "proc_map", ""),
        ("inputevents", "med_map", "AND coalesce(e.statusdescription, '') <> 'Rewritten'"),
    ]:
        src = _csv(mimic_dir / "icu" / f"{table}.csv", iv)
        parts.append(f"""
            SELECT DISTINCT e.hadm_id, m.var, h.hour
            FROM {src} e JOIN c ON e.hadm_id = c.hadm_id JOIN {mapping} m ON e.itemid = m.itemid
            CROSS JOIN range(0, {N_HOURS}) h(hour)
            WHERE e.starttime < c.admittime + to_hours(CAST(h.hour AS INTEGER) + 1)
              AND greatest(e.endtime, e.starttime + INTERVAL 1 SECOND)
                  > c.admittime + to_hours(CAST(h.hour AS INTEGER)) {rewritten_filter}""")
    con.execute(f"COPY ({' UNION ALL '.join(parts)}) TO '{out_dir / 'interval_long.parquet'}' (FORMAT parquet)")
    _log_rejects(con, "procedure/inputevents")
    con.close()


def _row_index(hadm_order: np.ndarray, hadm: pd.Series) -> np.ndarray:
    pos = pd.Series(np.arange(len(hadm_order)), index=hadm_order)
    return pos.reindex(hadm.to_numpy()).to_numpy()


def assemble_continuous(
    long_df: pd.DataFrame, hadm_order: np.ndarray, variables: list[str] | None = None
) -> np.ndarray:
    """[N, 48, V] float32 with NaN where a variable is unobserved in that hour."""
    variables = CONTINUOUS_VARS if variables is None else variables
    out = np.full((len(hadm_order), N_HOURS, len(variables)), np.nan, dtype=np.float32)
    var_idx = {v: i for i, v in enumerate(variables)}
    rows = _row_index(hadm_order, long_df["hadm_id"])
    cols = long_df["var"].map(var_idx).to_numpy()
    ok = ~np.isnan(rows.astype(float))
    out[rows[ok].astype(int), long_df["hour"].to_numpy()[ok], cols[ok].astype(int)] = long_df["value"].to_numpy()[ok]
    return out


def assemble_interval(long_df: pd.DataFrame, hadm_order: np.ndarray) -> np.ndarray:
    """[N, 48, P] uint8, 1 if the procedure / medication is active during the hour."""
    out = np.zeros((len(hadm_order), N_HOURS, len(INTERVAL_VARS)), dtype=np.uint8)
    var_idx = {v: i for i, v in enumerate(INTERVAL_VARS)}
    rows = _row_index(hadm_order, long_df["hadm_id"])
    cols = long_df["var"].map(var_idx).to_numpy()
    ok = ~np.isnan(rows.astype(float))
    out[rows[ok].astype(int), long_df["hour"].to_numpy()[ok], cols[ok].astype(int)] = 1
    return out


def encode_static(cohort: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Static features: age [N, 1] float32 and categorical codes [N, C] int64 (0 = UNKNOWN)."""
    age = cohort["age"].astype("float32").to_numpy().reshape(-1, 1)
    codes, vocab = [], {}
    for col in STATIC_CATEGORICAL:
        values = cohort[col].astype("object").where(cohort[col].notna(), None)
        cats = sorted({str(v) for v in values if v is not None})
        vocab[col] = ["UNKNOWN"] + cats
        lookup = {c: i + 1 for i, c in enumerate(cats)}
        codes.append(np.array([lookup.get(str(v), 0) if v is not None else 0 for v in values], dtype=np.int64))
    return age, np.stack(codes, axis=1), vocab
