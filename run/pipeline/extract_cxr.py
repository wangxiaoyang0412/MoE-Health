"""Encode in-window AP chest X-rays with frozen torchxrayvision DenseNet-121 (CheXpert weights).

Features are the global-average-pooled penultimate activations (``model.features2``).

Outputs (paths.feature_dir):
    cxr_img_emb.npy    [n_images, 1024] (cxr_window.parquet order)
    cxr_emb.npy        [N, 1024] earliest AP image per admission (paper input; zeros if none)
    cxr_emb_mean.npy   [N, 1024] mean over all in-window AP images (optional variant)
    cxr_count.npy      [N] number of in-window AP images
    cxr_report.md      aggregate statistics only
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_module.cxr_images import CXRImages

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    feat = Path(cfg.paths.feature_dir)
    feat.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(Path(cfg.paths.cohort_dir) / "cohort.parquet", columns=["hadm_id"])
    imgs = pd.read_parquet(Path(cfg.paths.cohort_dir) / "cxr_window.parquet", columns=["hadm_id", "is_first", "path"])

    loader = DataLoader(CXRImages(Path(cfg.paths.cxr_root), imgs["path"].tolist()),
                        batch_size=int(cfg.cxr.batch_size), num_workers=int(cfg.cxr.num_workers))
    model = xrv.models.DenseNet(weights=cfg.cxr.weights).to(cfg.cxr.device).eval()
    chunks = []
    with torch.inference_mode():
        for x in tqdm(loader, desc="cxr", mininterval=30):
            chunks.append(model.features2(x.to(cfg.cxr.device)).float().cpu().numpy())
    emb_img = np.concatenate(chunks)
    np.save(feat / "cxr_img_emb.npy", emb_img)

    pos = pd.Series(np.arange(len(cohort)), index=cohort["hadm_id"].to_numpy())
    rows = pos.reindex(imgs["hadm_id"].to_numpy()).to_numpy().astype(np.int64)
    first = np.zeros((len(cohort), emb_img.shape[1]), dtype=np.float32)
    first_mask = imgs["is_first"].to_numpy()
    first[rows[first_mask]] = emb_img[first_mask]
    count = np.bincount(rows, minlength=len(cohort)).astype(np.int64)
    mean = np.zeros_like(first)
    np.add.at(mean, rows, emb_img)
    mean[count > 0] /= count[count > 0, None]
    np.save(feat / "cxr_emb.npy", first)
    np.save(feat / "cxr_emb_mean.npy", mean)
    np.save(feat / "cxr_count.npy", count)

    report = [
        "# CXR feature report (aggregates only)", "",
        f"- images encoded: {len(emb_img)}; admissions with >=1 image: {int((count > 0).sum())}",
        f"- first-image rows filled: {int(first_mask.sum())}",
        f"- feature mean {emb_img.mean():.4f}, per-dim std mean {emb_img.std(axis=0).mean():.4f}",
        f"- NaN/Inf in features: {int((~np.isfinite(emb_img)).sum())}",
    ]
    (feat / "cxr_report.md").write_text("\n".join(report) + "\n")
    logger.info("cxr features: %s, admissions with images %d", emb_img.shape, int((count > 0).sum()))


if __name__ == "__main__":
    main()
