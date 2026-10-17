"""HAIM-style summary statistics of the 48 h EHR series (input of the optional statistics branch
of the MoE-Health EHR encoder)."""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["N_STATS", "ehr_summary_features", "series_stats"]

N_STATS = 11  # max, min, mean, var, mean diff, mean |diff|, max |diff|, sum |diff|, last-first, peaks, trend


def series_stats(z: np.ndarray, obs: np.ndarray, missing: float = 0.0) -> np.ndarray:
    """[N, 11] statistics of each row's observed values in time order.

    Rows without any observation get ``missing`` (default 0);
    difference-based statistics of single-observation rows are 0.
    """
    n, t = z.shape
    order = np.argsort(~obs, axis=1, kind="stable")
    seq = np.take_along_axis(np.where(obs, z, np.nan), order, axis=1)
    cnt = obs.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = seq[:, 1:] - seq[:, :-1]
        ad = np.abs(d)
        last = seq[np.arange(n), np.maximum(cnt - 1, 0)]
        mid, left, right = seq[:, 1:-1], seq[:, :-2], seq[:, 2:]
        peaks = ((mid > left) & (mid > right)).sum(axis=1)
        x = np.where(np.isnan(seq), np.nan, np.arange(t)[None, :])
        xc, yc = x - np.nanmean(x, axis=1, keepdims=True), seq - np.nanmean(seq, axis=1, keepdims=True)
        varx = np.nansum(xc**2, axis=1)
        trend = np.where(varx > 0, np.nansum(xc * yc, axis=1) / np.where(varx > 0, varx, 1), 0.0)
        stats = np.stack([
            np.nanmax(seq, 1), np.nanmin(seq, 1), np.nanmean(seq, 1), np.nanvar(seq, 1),
            np.nanmean(d, 1), np.nanmean(ad, 1), np.nanmax(ad, 1), np.nansum(ad, 1),
            last - seq[:, 0], peaks, trend,
        ], axis=1)
    stats = np.nan_to_num(stats, nan=0.0)
    stats[cnt == 0] = missing
    return stats.astype(np.float32)


def ehr_summary_features(ehr_ts: np.ndarray, n_cont: int, n_interval: int, missing: float = 0.0) -> np.ndarray:
    """[N, 12 V + P]: per-variable statistics, observation fraction, and fraction of hours active.

    ``ehr_ts`` is the prepared tensor [N, T, 2V + P (+ V delta)]: z values, observation mask, active flags.
    """
    z, obs = ehr_ts[:, :, :n_cont], ehr_ts[:, :, n_cont : 2 * n_cont] > 0.5
    act = ehr_ts[:, :, 2 * n_cont : 2 * n_cont + n_interval]
    stats = [series_stats(z[:, :, j], obs[:, :, j], missing) for j in range(n_cont)]
    observed_frac = obs.mean(axis=1).astype(np.float32)
    return np.concatenate([*stats, observed_frac, act.mean(axis=1)], axis=1).astype(np.float32)
