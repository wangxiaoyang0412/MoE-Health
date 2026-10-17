"""Aggregate metrics.json files into a per-run CSV and Table 1/2/3-style markdown summaries.

Usage (from the project root): .venv/bin/python run/pipeline/collect_results.py [--paths local]
Writes {outputs}/results/{runs.csv, summary.md}. Metrics are aggregates (no patient rows).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger("collect_results")

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
METHOD_ORDER = ["moe_health", "moe_health_paperconfig", "unimodal_ehr", "unimodal_note", "unimodal_cxr"]
REFERENCES = ["moe_health_paperconfig", "unimodal_ehr"]
TASKS = ["mortality", "los", "readmission"]
PAPER_T1 = {"mortality": (0.818, 0.465), "los": (0.794, 0.739), "readmission": (0.643, 0.281)}  # MoE-Health


def load_runs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("*/*/*/*/seed*/metrics.json")):
        m = json.loads(path.read_text())
        row = {"exp": path.relative_to(run_dir).parts[0], "variant": f"{m['model']}{m['tag']}", "model": m["model"],
               "tag": m["tag"], "cohort": m["cohort"], "task": m["task"], "seed": m["seed"],
               "modalities": m["modalities"], "minutes": m["minutes"], "val_auroc": m["val"]["auroc"]}
        row.update({f"test_{k}": v for k, v in m["test"].items()})
        for combo, cm in m.get("test_per_combo", {}).items():
            row[f"auroc_{combo}"] = cm["auroc"]
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(s: pd.Series) -> str:
    s = s.dropna()
    return "-" if s.empty else f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f} (n={len(s)})"


def pivot(df: pd.DataFrame, index: str, metrics: list[str], order: list[str] | None = None) -> str:
    cols = [(t, m) for t in TASKS for m in metrics if not df[df.task == t].empty]
    idx = order or sorted(df[index].unique())
    lines = ["| " + index + " | " + " | ".join(f"{t} {m.replace('test_', '')}" for t, m in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for key in [i for i in idx if i in set(df[index])]:
        sub = df[df[index] == key]
        lines.append(f"| {key} | " + " | ".join(mean_std(sub[sub.task == t][m]) for t, m in cols) + " |")
    return "\n".join(lines)


def paired(main: pd.DataFrame) -> list[str]:
    """Tuned MoE-Health against each reference run on the same split (test AUROC)."""
    lines = ["## MoE-Health (tuned recipe) vs references (paired by seed, test AUROC)", "",
             "| cohort | task | reference | MoE-Health | reference | mean diff | seeds MoE-Health better |",
             "|---|---|---|---|---|---|---|"]
    for (cohort, task), g in main.groupby(["cohort", "task"]):
        moe = g[g.variant == "moe_health"].set_index("seed")["test_auroc"]
        for ref in REFERENCES:
            base = g[g.variant == ref].set_index("seed")["test_auroc"]
            diff = (moe - base).dropna()
            if diff.empty:
                continue
            lines.append(f"| {cohort} | {task} | {ref} | {moe.mean():.3f} | {base.mean():.3f} | "
                         f"{diff.mean():+.3f} | {int((diff > 0).sum())}/{len(diff)} |")
    return lines


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="default", help="config in run/conf/paths/ (e.g. local)")
    paths = OmegaConf.load(CONF_DIR / "paths" / f"{ap.parse_args().paths}.yaml")
    run_dir, out = Path(paths.run_dir), Path(paths.outputs) / "results"
    df = load_runs(run_dir)
    if df.empty:
        logger.warning("no runs found under %s", run_dir)
        return
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    lines = ["# Results summary (test set, mean ± std over seeds)", ""]
    for exp in sorted(df.exp.unique()):
        e = df[df.exp == exp]
        for cohort in sorted(e.cohort.unique()):
            lines += [f"## {exp} / {cohort}", "", pivot(e[e.cohort == cohort], "variant",
                                                         ["test_auroc", "test_f1", "test_auprc"],
                                                         METHOD_ORDER if exp == "main" else None), ""]
    if "main" in set(df.exp):
        main_runs = df[df.exp == "main"]
        lines += paired(main_runs)
        lines += ["", "## MoE-Health in the paper (Table 1, for reference)", "",
                  "| " + " | ".join(f"{t} AUROC | {t} F1" for t in TASKS) + " |", "|---" * 6 + "|",
                  "| " + " | ".join(f"{PAPER_T1[t][0]:.3f} | {PAPER_T1[t][1]:.3f}" for t in TASKS) + " |"]
        combo_cols = [c for c in df.columns if c.startswith("auroc_")]
        moe = main_runs[main_runs.model == "moe_health"]
        lines += ["", "## MoE-Health test AUROC per data combination (mean over seeds)", "",
                  "| variant | cohort | task | " + " | ".join(combo_cols) + " |", "|---" * (len(combo_cols) + 3) + "|"]
        for (variant, cohort, task), g in moe.groupby(["variant", "cohort", "task"]):
            lines.append(f"| {variant} | {cohort} | {task} | " + " | ".join(f"{g[c].mean():.3f}" for c in combo_cols)
                         + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d runs)", out / "summary.md", len(df))


if __name__ == "__main__":
    main()
