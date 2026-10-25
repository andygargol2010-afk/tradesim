"""Is an agent's result skill or luck? Bootstrap confidence interval of the Sharpe ratio, probability of beating
buy & hold, and how many more days the current performance would need to become statistically significant."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _block_bootstrap(n: int, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indexes of ``n_boot`` circular block-bootstrap resamples of length ``n`` (keeps short autocorrelation)."""
    starts = rng.integers(0, n, size=(n_boot, math.ceil(n / block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx % n


def _days_to_significance(mean: float, sd: float, n: int, z: float = 2.0) -> int | None:
    """Extra days until mean / sd * sqrt(days) reaches ``z`` at the current pace (None if the mean is not positive)."""
    if not (sd > 0) or mean <= 0:
        return None
    return max(0, math.ceil((z * sd / mean) ** 2) - n)


def significance(daily: pd.Series, benchmark: pd.Series | None = None, learning: dict[str, Any] | None = None,
                 n_boot: int = 2000, block: int = 5, seed: int = 0) -> dict[str, Any]:
    r = daily.dropna().to_numpy(dtype=float)
    n = len(r)
    out: dict[str, Any] = {"days": n}
    if n >= 10 and r.std(ddof=1) > 0:
        rng = np.random.default_rng(seed)
        idx = _block_bootstrap(n, n_boot, block, rng)
        samples = r[idx]
        sd = samples.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(252), 0.0)
        out["sharpe"] = round(float(r.mean() / r.std(ddof=1) * math.sqrt(252)), 3)
        out["sharpe_ci95"] = [round(float(np.percentile(sharpe, 2.5)), 3), round(float(np.percentile(sharpe, 97.5)), 3)]
        out["p_sharpe_positive"] = round(float(np.mean(sharpe > 0)), 3)
        out["days_to_significant_sharpe"] = _days_to_significance(float(r.mean()), float(r.std(ddof=1)), n)
        if benchmark is not None:
            common = daily.index.intersection(benchmark.index)
            d = (daily[common] - benchmark[common]).dropna().to_numpy(dtype=float)
            if len(d) >= 10 and d.std(ddof=1) > 0:
                bi = _block_bootstrap(len(d), n_boot, block, rng)
                out["p_beats_buy_hold"] = round(float(np.mean(d[bi].mean(axis=1) > 0)), 3)
                out["excess_return_ann_pct"] = round(float(d.mean()) * 252 * 100, 2)
                out["days_to_significant_vs_buy_hold"] = _days_to_significance(float(d.mean()), float(d.std(ddof=1)),
                                                                               len(d))
    if learning and learning.get("rank_ic_mean") is not None:
        ic, t, days = learning["rank_ic_mean"], learning.get("rank_ic_t"), learning.get("rank_ic_days") or 0
        out["rank_ic"] = {"mean": round(ic, 4), "t": None if t is None else round(t, 2), "days": days}
        if t and ic > 0 and days:
            out["rank_ic"]["days_to_significant"] = max(0, math.ceil((2.0 / t) ** 2 * days) - days)
    return out
