"""Market-condition scenarios: deterministic transforms of OHLCV bars.

Scenarios work on real or synthetic data and are seeded, so a stress episode
is reproducible. They operate in log-price space *relative to the previous
close*, so OHLC consistency (``low <= open, close <= high``) is preserved.

Besides prices they can set two optional per-bar columns read by the
execution engine:

* ``spread_mult``: multiplies the modelled bid/ask spread,
* ``liq_mult``: multiplies available volume / book depth.

Example::

    scenario = Scenario.preset("high_volatility")
    stressed = scenario.apply(bars, rng=np.random.default_rng(7))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


def _to_rel(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Return first previous close and an (n, 4) array of log(O,H,L,C / prev_close)."""
    c = df["close"].to_numpy()
    prev = np.concatenate([[df["open"].iloc[0]], c[:-1]])
    rel = np.log(df[["open", "high", "low", "close"]].to_numpy() / prev[:, None])
    return float(prev[0]), rel


def _from_rel(df: pd.DataFrame, first_prev: float, rel: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    log_closes = np.log(first_prev) + np.cumsum(rel[:, 3])
    log_prev = np.concatenate([[np.log(first_prev)], log_closes[:-1]])
    out["open"] = np.exp(log_prev + rel[:, 0])
    out["high"] = np.exp(log_prev + rel[:, 1])
    out["low"] = np.exp(log_prev + rel[:, 2])
    out["close"] = np.exp(log_closes)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def _mult_col(df: pd.DataFrame, col: str, mult: np.ndarray | float) -> None:
    base = df[col].to_numpy() if col in df.columns else np.ones(len(df))
    df[col] = base * mult


class Transform(Protocol):
    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame: ...


@dataclass
class VolatilityScale:
    """Scale all log moves (gaps, ranges, closes) around their mean by ``factor``."""

    factor: float = 2.0

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if len(df) < 2:
            return df
        first, rel = _to_rel(df)
        mu = rel[:, 3].mean()
        rel = rel * self.factor
        rel[:, 3] += mu * (1 - self.factor)  # keep drift unchanged
        out = _from_rel(df, first, rel)
        _mult_col(out, "spread_mult", self.factor ** 0.5)
        return out


@dataclass
class Trend:
    """Add a deterministic drift of ``annual_return`` (log) spread over ``bars_per_year``."""

    annual_return: float = -0.5
    bars_per_year: float = 252.0

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if len(df) < 2:
            return df
        first, rel = _to_rel(df)
        rel += self.annual_return / self.bars_per_year
        return _from_rel(df, first, rel)


@dataclass
class GapInjection:
    """Random opening gaps: with probability ``prob`` per bar, shift the whole bar by
    ``N(0, size)`` in log terms (the gap persists). Stops get filled *through* the gap."""

    prob: float = 0.02
    size: float = 0.06

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if len(df) < 2:
            return df
        first, rel = _to_rel(df)
        mask = rng.random(len(df)) < self.prob
        mask[0] = False
        rel[mask] += rng.normal(0, self.size, size=mask.sum())[:, None]
        out = _from_rel(df, first, rel)
        _mult_col(out, "spread_mult", np.where(mask, 3.0, 1.0))
        return out


@dataclass
class FlashCrash:
    """Intrabar crash with partial recovery: at ``at_fraction`` of the series (or random)
    the low pierces ``depth`` below the open, with volume x5, spreads x10, depth /4."""

    depth: float = 0.15
    at_fraction: float | None = None
    recovery: float = 0.6  # fraction of the drop recovered by the bar's close

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if len(df) < 3:
            return df
        out = df.copy()
        i = int((self.at_fraction if self.at_fraction is not None else rng.uniform(0.2, 0.9)) * (len(df) - 1))
        o = out["open"].iloc[i]
        low = o * (1 - self.depth)
        close = low + (o - low) * self.recovery
        col = out.columns.get_loc
        out.iloc[i, col("low")] = min(out["low"].iloc[i], low)
        out.iloc[i, col("close")] = close
        out.iloc[i, col("high")] = max(out["high"].iloc[i], o, close)
        # Shift the rest of the series so the next open is continuous with the crash close.
        ratio = close / df["close"].iloc[i]
        for c in ("open", "high", "low", "close"):
            out.iloc[i + 1:, col(c)] = df[c].iloc[i + 1:] * ratio
        out.iloc[i, col("volume")] = out["volume"].iloc[i] * 5
        spread = np.ones(len(out))
        liq = np.ones(len(out))
        spread[i] = 10.0
        liq[i] = 0.25
        _mult_col(out, "spread_mult", spread)
        _mult_col(out, "liq_mult", liq)
        return out


@dataclass
class LiquidityDrought:
    """Thin markets: volume x ``volume_factor``, spreads x ``spread_factor``."""

    volume_factor: float = 0.3
    spread_factor: float = 3.0

    def __call__(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        out["volume"] = out["volume"] * self.volume_factor
        _mult_col(out, "spread_mult", self.spread_factor)
        _mult_col(out, "liq_mult", self.volume_factor)
        return out


@dataclass
class Scenario:
    """An ordered list of transforms, applied identically (but with per-symbol rng) to every asset."""

    name: str = "custom"
    transforms: list[Transform] = field(default_factory=list)

    def apply(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df
        for t in self.transforms:
            out = t(out, rng)
        return out

    @classmethod
    def preset(cls, name: str) -> "Scenario":
        presets: dict[str, list[Transform]] = {
            "normal": [],
            "high_volatility": [VolatilityScale(2.0), LiquidityDrought(0.7, 1.5)],
            "crash": [Trend(-0.8), VolatilityScale(1.8), GapInjection(0.03, 0.05)],
            "bull": [Trend(0.6)],
            "gaps": [GapInjection(0.05, 0.08)],
            "flash_crash": [FlashCrash(0.2)],
            "illiquid": [LiquidityDrought(0.2, 4.0)],
        }
        if name not in presets:
            raise KeyError(f"Unknown scenario {name!r}; choose from {sorted(presets)}")
        return cls(name=name, transforms=presets[name])

    @staticmethod
    def presets() -> list[str]:
        return ["normal", "high_volatility", "crash", "bull", "gaps", "flash_crash", "illiquid"]
