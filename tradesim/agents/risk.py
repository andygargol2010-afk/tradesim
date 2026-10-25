"""Position sizing and portfolio risk shared by the agents.

* Volatility targeting: each position gets a weight proportional to ``1 / sigma`` (equal risk instead of equal money:
  20 % in DOGE is not the same bet as 20 % in KO) and the whole portfolio is scaled to ``target_vol`` a year using
  the 60-day covariance (shrunk towards its diagonal).
* Correlated groups: assets whose 60-day correlation is above ``threshold`` are clustered (complete linkage: every
  pair inside a group is that correlated) and each group, plus crypto as a class, is capped (``group_cap``).
* ``exposure``: overall multiplier (the runner lowers it in drawdowns, see ``Observation.exposure_cap``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tradesim.data.base import DataUnavailableError


def returns_frame(obs: Any, symbols: Iterable[str], n: int = 61) -> pd.DataFrame:
    """Daily log returns of the last ``n`` bars, one column per symbol, aligned by date."""
    cols = {}
    for s in symbols:
        try:
            df = obs.history(s, limit=n)
        except (DataUnavailableError, KeyError, ValueError):
            continue
        c = df["close"].astype(float)
        c.index = pd.DatetimeIndex(c.index).normalize()
        c = c[~c.index.duplicated(keep="last")]
        if len(c) > 10 and (c > 0).all():
            cols[s] = np.log(c).diff()
    return pd.DataFrame(cols).iloc[1:]


def periods_per_year(symbol: str, universe: Any = None) -> int:
    try:
        return 365 if universe.get(symbol).is_24_7 else 252
    except Exception:  # noqa: BLE001
        return 365 if symbol.endswith("-USD") else 252


def annual_vols(rets: pd.DataFrame, universe: Any = None, window: int = 20) -> dict[str, float]:
    out = {}
    for s in rets:
        r = rets[s].dropna().tail(window)
        if len(r) >= 10 and r.std() > 0:
            out[s] = float(r.std() * math.sqrt(periods_per_year(s, universe)))
    return out


def correlation_groups(rets: pd.DataFrame, threshold: float = 0.7, min_obs: int = 30) -> dict[str, int]:
    """Complete-linkage clusters of assets with pairwise correlation above ``threshold``."""
    syms = list(rets.columns)
    corr = rets.corr(min_periods=min_obs).fillna(0.0)
    clusters = [{s} for s in syms]
    while True:
        best, pair = threshold, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                link = min(corr.loc[a, b] for a in clusters[i] for b in clusters[j])
                if link > best:
                    best, pair = link, (i, j)
        if pair is None:
            break
        i, j = pair
        clusters[i] |= clusters.pop(j)
    return {s: k for k, cl in enumerate(clusters) for s in cl}


def apply_caps(weights: dict[str, float], groups: dict[str, Any], group_cap: float,
               crypto: set[str] | None = None, crypto_cap: float | None = None) -> dict[str, float]:
    """Scale down the members of any group (and crypto as a whole) whose total weight exceeds its cap."""
    w = dict(weights)
    buckets: dict[Any, list[str]] = {}
    for s in w:
        buckets.setdefault(("corr", groups.get(s, s)), []).append(s)
    for members in buckets.values():
        tot = sum(w[s] for s in members)
        if len(members) > 1 and tot > group_cap:
            for s in members:
                w[s] *= group_cap / tot
    if crypto and crypto_cap is not None:
        tot = sum(v for s, v in w.items() if s in crypto)
        if tot > crypto_cap:
            for s in w:
                if s in crypto:
                    w[s] *= crypto_cap / tot
    return w


def size_positions(symbols: list[str], rets: pd.DataFrame, universe: Any = None, target_vol: float = 0.15,
                   max_weight: float = 0.25, invested_cap: float = 0.95, group_cap: float = 0.30,
                   crypto_cap: float = 0.30, corr_threshold: float = 0.7, exposure: float = 1.0) -> dict[str, float]:
    """Inverse-volatility weights for ``symbols`` scaled to ``target_vol`` a year, then capped per position, per
    correlated group, for crypto and in total, and multiplied by ``exposure``."""
    vols = annual_vols(rets, universe)
    syms = [s for s in symbols if s in vols]
    if not syms:
        return {}
    raw = np.array([1.0 / vols[s] for s in syms])
    w = raw / raw.sum()
    sub = rets[syms].dropna(how="all").tail(60)
    ann = np.array([math.sqrt(periods_per_year(s, universe)) for s in syms])
    cov = sub.cov(min_periods=10).fillna(0.0).to_numpy() * np.outer(ann, ann)
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))  # shrinkage: noisy correlations from 60 days
    port = math.sqrt(max(float(w @ cov @ w), 1e-12))
    w = w * (target_vol / port)
    weights = {s: float(min(x, max_weight)) for s, x in zip(syms, w)}
    groups = correlation_groups(rets[syms], corr_threshold) if len(syms) > 1 else {}
    crypto = {s for s in syms if periods_per_year(s, universe) == 365}
    weights = apply_caps(weights, groups, group_cap, crypto, crypto_cap)
    tot = sum(weights.values())
    if tot > invested_cap:
        weights = {s: v * invested_cap / tot for s, v in weights.items()}
    return {s: v * max(0.0, min(1.0, exposure)) for s, v in weights.items()}


def drawdown_exposure(drawdown_pct: float, limit_pct: float) -> float:
    """Gradual de-risking: ``1 - (drawdown / limit)^2`` (1 at the peak, 0 at the limit)."""
    if limit_pct <= 0:
        return 1.0
    return max(0.0, 1.0 - (max(drawdown_pct, 0.0) / limit_pct) ** 2)
