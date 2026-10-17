"""Run experiment grids of run/train.py across GPUs (resumable: finished runs are skipped).

Usage (from the project root):
    .venv/bin/python run/pipeline/launch.py main table2 ablation --gpus 0 1 2 3 --jobs-per-gpu 3
    .venv/bin/python run/pipeline/launch.py main --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger("launch")

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
SEEDS = range(5)
TASKS = ("mortality", "los", "readmission")
UNIMODAL = ("ehr", "note", "cxr")
# Paper Table 3 ablations plus variants for design choices the paper leaves open.
ABLATIONS = {
    "_no_missing_emb": ["model.missing=zeros"],
    "_no_pretrain": ["model.pretrain=none"],
    "_generic_experts": ["model.pretrain=generic"],
    "_uniform_gate": ["model.gating=uniform"],
    "_top1": ["model.top_k=1"],
    "_observed4_exact": ["model.experts=observed4", "model.pretrain=exact"],
    "_logit_experts": ["model.expert_output=logit"],
    "_no_balance": ["model.balance=none"],
    "_switch_balance": ["model.balance=switch"],
}


@dataclass(frozen=True)
class Job:
    exp: str
    name: str  # model name as it appears in the output path
    overrides: tuple[str, ...]
    cohort: str
    task: str
    seed: int
    tag: str = ""

    def out_dir(self, run_dir: Path) -> Path:
        return run_dir / self.exp / f"{self.name}{self.tag}" / self.cohort / self.task / f"seed{self.seed}"

    def args(self, device: str, paths: str) -> list[str]:
        return [*self.overrides, f"paths={paths}", f"exp={self.exp}", f"tag={self.tag}", f"cohort={self.cohort}",
                f"task={self.task}", f"seed={self.seed}", f"device={device}"]


def grid(name: str) -> list[Job]:
    jobs: list[Job] = []
    if name == "main":  # Table 1: tuned recipe and paper settings on both cohorts, unimodal references
        for cohort, task, seed in itertools.product(("corrected", "faithful"), TASKS, SEEDS):
            jobs.append(Job("main", "moe_health", ("model=moe_health", "+recipe=moe_final"), cohort, task, seed))
            jobs.append(Job("main", "moe_health", ("model=moe_health",), cohort, task, seed, tag="_paperconfig"))
            if cohort == "corrected":
                jobs += [Job("main", f"unimodal_{m}", ("model=unimodal", f"model.modality={m}", "+recipe=unimodal"),
                             cohort, task, seed) for m in UNIMODAL]
    elif name == "table2":  # Table 2: modality subsets (E+T+I is the tuned run of the main grid)
        for mods, seed in itertools.product(("E", "T", "I", "ET", "EI", "TI"), SEEDS):
            jobs.append(Job("table2", "moe_health", ("model=moe_health", "+recipe=moe_final",
                                                     f"data.modalities={mods}"), "corrected", "los", seed,
                            tag=f"_{mods}"))
    elif name == "ablation":  # Table 3
        for (tag, ov), seed in itertools.product(ABLATIONS.items(), SEEDS):
            jobs.append(Job("ablation", "moe_health", ("model=moe_health", "+recipe=moe_final", *ov), "corrected",
                            "mortality", seed, tag))
    else:
        raise ValueError(f"unknown grid '{name}'")
    return jobs


def free_mb(gpu: str) -> int:
    """Free memory of one GPU in MiB (the GPUs may be shared with other jobs)."""
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return int(out) if out.isdigit() else 0


def worker(q: queue.Queue[Job], gpu: str, failures: list[Job], run_dir: Path, paths: str, min_free_mb: int) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    device = f"cuda:{gpu}"
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        while free_mb(gpu) < min_free_mb:  # never take memory another job may need
            time.sleep(60)
        out_dir = job.out_dir(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "launcher.out", "w") as fh:
            rc = subprocess.call([sys.executable, "run/train.py", *job.args(device, paths)], stdout=fh,
                                 stderr=subprocess.STDOUT, env=env)
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        logger.info("%-8s %s (%s)", status, out_dir.relative_to(run_dir), device)
        if rc != 0:
            failures.append(job)
        q.task_done()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="+")
    ap.add_argument("--paths", default="default", help="config in run/conf/paths/ (e.g. local)")
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--jobs-per-gpu", type=int, default=2)
    ap.add_argument("--min-free-mb", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(OmegaConf.load(CONF_DIR / "paths" / f"{args.paths}.yaml").run_dir)
    jobs = [j for g in args.grids for j in grid(g)]
    todo = [j for j in jobs if not (j.out_dir(run_dir) / "metrics.json").exists()]
    logger.info("%d jobs in grids %s, %d to run", len(jobs), args.grids, len(todo))
    if args.dry_run:
        for j in todo:
            logger.info("would run: %s", " ".join(j.args("cuda:X", args.paths)))
        return
    q: queue.Queue[Job] = queue.Queue()
    for j in todo:
        q.put(j)
    failures: list[Job] = []
    threads = [threading.Thread(target=worker, args=(q, g, failures, run_dir, args.paths, args.min_free_mb))
               for g in args.gpus for _ in range(args.jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.info("done: %d ok, %d failed", len(todo) - len(failures), len(failures))
    for j in failures:
        logger.info("failed: %s", j.out_dir(run_dir))


if __name__ == "__main__":
    main()
