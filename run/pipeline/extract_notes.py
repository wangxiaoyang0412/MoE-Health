"""Encode in-window radiology reports with frozen Bio_ClinicalBERT ([CLS] of the last layer).

Outputs (paths.feature_dir):
    note_cls.npy      [n_notes, 768] per-report [CLS] (notes_window.parquet order)
    note_tok.npy      [n_notes, 768] per-report mean of token states (optional variant)
    note_emb.npy      [N, 768] mean of [CLS] over an admission's reports (paper Eq. 7; zeros if none)
    note_emb_tok.npy  [N, 768] same aggregation for the token-mean variant
    note_count.npy    [N] number of reports per admission
    note_report.md    aggregate statistics only
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_module.note_features import salient_text

logger = logging.getLogger(__name__)


@torch.inference_mode()
def encode(
    texts: list[str], model_name: str, max_length: int, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """[CLS] and attention-masked token-mean embeddings, in length-sorted batches for speed."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).half().to(device).eval()
    order = np.argsort([len(t) for t in texts])
    cls = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    tokmean = np.zeros_like(cls)
    for start in tqdm(range(0, len(order), batch_size), desc="notes", mininterval=30):
        idx = order[start : start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        cls[idx] = hidden[:, 0].cpu().numpy()
        tokmean[idx] = ((hidden * m).sum(1) / m.sum(1)).cpu().numpy()
    return cls, tokmean


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    if cfg.paths.get("hf_home"):
        os.environ.setdefault("HF_HOME", str(cfg.paths.hf_home))
    feat = Path(cfg.paths.feature_dir)
    feat.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(Path(cfg.paths.cohort_dir) / "cohort.parquet", columns=["hadm_id"])
    notes = pd.read_parquet(Path(cfg.paths.cohort_dir) / "notes_window.parquet", columns=["hadm_id", "text"])

    cleaned = [salient_text(t) for t in notes["text"]]
    texts = [t for t, _ in cleaned]
    found = np.array([f for _, f in cleaned])
    cls, tokmean = encode(texts, cfg.notes.model, int(cfg.notes.max_length), int(cfg.notes.batch_size),
                          cfg.notes.device)
    np.save(feat / "note_cls.npy", cls)
    np.save(feat / "note_tok.npy", tokmean)

    pos = pd.Series(np.arange(len(cohort)), index=cohort["hadm_id"].to_numpy())
    rows = pos.reindex(notes["hadm_id"].to_numpy()).to_numpy().astype(np.int64)
    count = np.bincount(rows, minlength=len(cohort)).astype(np.int64)
    for name, per_note in (("note_emb.npy", cls), ("note_emb_tok.npy", tokmean)):
        emb = np.zeros((len(cohort), per_note.shape[1]), dtype=np.float32)
        np.add.at(emb, rows, per_note)
        emb[count > 0] /= count[count > 0, None]
        np.save(feat / name, emb)
    np.save(feat / "note_count.npy", count)

    n_tokens = [len(t.split()) for t in texts]
    report = [
        "# Note feature report (aggregates only)", "",
        f"- reports encoded: {len(texts)}; admissions with >=1 report: {int((count > 0).sum())}",
        f"- reports with FINDINGS/IMPRESSION found: {found.mean():.3f} (others use the full report)",
        f"- words per cleaned report: median {np.median(n_tokens):.0f}, p95 {np.percentile(n_tokens, 95):.0f}",
        f"- embedding norm: mean {np.linalg.norm(cls, axis=1).mean():.2f}; "
        f"per-dim std mean {cls.std(axis=0).mean():.4f}",
        f"- NaN/Inf in embeddings: {int((~np.isfinite(cls)).sum())}",
    ]
    (feat / "note_report.md").write_text("\n".join(report) + "\n")
    logger.info("note features: cls %s, admissions with notes %d", cls.shape, int((count > 0).sum()))


if __name__ == "__main__":
    main()
