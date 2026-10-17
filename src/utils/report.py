"""Small helpers for aggregate-only markdown reports (no row-level data)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["df_to_markdown", "prevalence_table"]


def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Render a small DataFrame as a GitHub markdown table without extra dependencies."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [float_fmt.format(v) if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prevalence_table(
    df: pd.DataFrame, label_cols: Sequence[str], split_col: str | None = None
) -> pd.DataFrame:
    """Count and positive rate per label (optionally per split); NaN labels are excluded."""
    groups = [("all", df)] if split_col is None else list(df.groupby(split_col))
    rows = []
    for name, g in groups:
        row: dict[str, object] = {"split": name, "n": len(g)}
        for col in label_cols:
            y = g[col].dropna()
            row[f"{col} n"] = len(y)
            row[f"{col} rate"] = float(y.mean()) if len(y) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
