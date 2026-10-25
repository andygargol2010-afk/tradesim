"""In-memory market data for a simulation run, with look-ahead-safe features."""

from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe
from tradesim.market.conditions import Scenario
from tradesim.market.liquidity import liquidity_profile

log = logging.getLogger(__name__)


@dataclass
class SymbolSeries:
    """Bars for one symbol as numpy arrays + derived, look-ahead-safe estimates.

    ``sigma[i]`` and ``typical_volume[i]`` only use bars ``< i``, so they are
    known before bar ``i`` trades.
    """

    asset: Asset
    frame: pd.DataFrame
    ts: np.ndarray  # int64 ns, bar open times
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    sigma: np.ndarray
    typical_volume: np.ndarray
    spread_mult: np.ndarray
    liq_mult: np.ndarray
    spread_bps: np.ndarray | None
    sigma_typical: float

    def index_exact(self, ts_ns: int) -> int | None:
        i = int(np.searchsorted(self.ts, ts_ns))
        return i if i < len(self.ts) and self.ts[i] == ts_ns else None

    def index_le(self, ts_ns: int) -> int | None:
        """Index of the last bar with open time <= ts_ns."""
        i = int(np.searchsorted(self.ts, ts_ns, side="right")) - 1
        return i if i >= 0 else None


def build_series(asset: Asset, df: pd.DataFrame, timeframe: Timeframe, lookback: int = 20) -> SymbolSeries:
    prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
    bpy = timeframe.bars_per_year(asset.is_24_7)
    sigma_typical = prof.annual_vol / math.sqrt(bpy)
    close = df["close"].to_numpy(dtype=float)
    logret = np.diff(np.log(close), prepend=np.nan)
    # Volatility from bars strictly before i: rolling std shifted by one.
    sig = pd.Series(logret).rolling(lookback, min_periods=max(5, lookback // 4)).std().shift(1).to_numpy()
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, sigma_typical)
    vol = df["volume"].to_numpy(dtype=float)
    vol_clean = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
    typical = pd.Series(vol_clean).rolling(lookback, min_periods=3).median().shift(1).to_numpy()
    # Fallback when the feed has no volume (FX on Yahoo, indices) or not enough history:
    # tier-typical dollar volume spread over the bars of a day.
    bars_per_day = max(bpy / (365.0 if asset.is_24_7 else 252.0), 1e-9)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    fallback = prof.adv_usd / np.maximum(prev_close, 1e-12) / bars_per_day
    typical = np.where(np.isfinite(typical) & (typical > 0), typical, fallback)
    n = len(df)
    spread_mult = df["spread_mult"].to_numpy(dtype=float) if "spread_mult" in df else np.ones(n)
    liq_mult = df["liq_mult"].to_numpy(dtype=float) if "liq_mult" in df else np.ones(n)
    spread_bps = df["spread_bps"].to_numpy(dtype=float) if "spread_bps" in df else None
    return SymbolSeries(
        asset=asset, frame=df, ts=pd.DatetimeIndex(df.index).as_unit("ns").asi8.astype(np.int64),
        open=df["open"].to_numpy(dtype=float), high=df["high"].to_numpy(dtype=float),
        low=df["low"].to_numpy(dtype=float), close=close, volume=vol, sigma=sig, typical_volume=typical,
        spread_mult=np.nan_to_num(spread_mult, nan=1.0), liq_mult=np.nan_to_num(liq_mult, nan=1.0),
        spread_bps=spread_bps, sigma_typical=sigma_typical,
    )


def _ns(ts: pd.Timestamp) -> int:
    return int(ts.value)


class MarketData:
    """Loads and holds bars for the symbols of one simulation run."""

    def __init__(self, provider: DataProvider, universe: AssetUniverse, timeframe: Timeframe,
                 start: datetime, end: datetime, warmup_bars: int = 50, lookback: int = 20,
                 scenario: Scenario | None = None, seed: int = 0):
        self.provider = provider
        self.universe = universe
        self.timeframe = timeframe
        self.start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        self.warmup_bars = warmup_bars
        self.lookback = lookback
        self.scenario = scenario
        self.seed = seed
        self.series: dict[str, SymbolSeries] = {}
        self.failed: dict[str, str] = {}

    def _fetch_start(self) -> pd.Timestamp:
        # Calendar padding: markets are closed ~1/3 of weekdays-hours; be generous.
        pad = self.timeframe.seconds * self.warmup_bars * (2.2 if self.timeframe.seconds < 86400 else 1.6)
        if self.timeframe.seconds < 86400:
            pad *= 24 / 6.5  # intraday equity bars only exist 6.5h a day
        return (self.start - pd.Timedelta(seconds=pad) - pd.Timedelta(days=4)).floor("s")

    def load(self, symbols: Iterable[str]) -> list[str]:
        """Fetch bars for ``symbols`` not loaded yet. Returns symbols successfully loaded."""
        todo = [s for s in dict.fromkeys(symbols) if s not in self.series and s not in self.failed]
        if not todo:
            return []
        assets = [self.universe.get(s) for s in todo]
        try:
            frames = self.provider.get_bars_many(assets, self.timeframe, self._fetch_start().to_pydatetime(),
                                                 self.end.to_pydatetime())
        except DataUnavailableError as exc:
            frames = {}
            log.warning("bulk load failed: %s", exc)
        loaded = []
        for asset in assets:
            df = frames.get(asset.symbol)
            if df is None or df.empty or len(df) < 2:
                self.failed[asset.symbol] = "no data"
                continue
            if self.scenario is not None and self.scenario.transforms:
                rng = np.random.default_rng([self.seed, zlib.crc32(asset.symbol.encode()), 99])
                in_window = df.index >= self.start
                if in_window.any():
                    # Scenario applies to the episode window only; warm-up history stays untouched.
                    df = pd.concat([df[~in_window], self.scenario.apply(df[in_window], rng)])
            self.series[asset.symbol] = build_series(asset, df, self.timeframe, self.lookback)
            loaded.append(asset.symbol)
        return loaded

    def ensure(self, symbol: str) -> SymbolSeries:
        if symbol not in self.series:
            self.load([symbol])
        s = self.series.get(symbol)
        if s is None:
            raise DataUnavailableError(f"No market data for {symbol}: {self.failed.get(symbol, 'unknown')}")
        return s

    def calendar(self, symbols: Iterable[str] | None = None) -> np.ndarray:
        """Sorted union of bar timestamps (ns) inside [start, end) across ``symbols``."""
        lo, hi = _ns(self.start), _ns(self.end)
        parts = [s.ts[(s.ts >= lo) & (s.ts < hi)] for k, s in self.series.items() if symbols is None or k in symbols]
        if not parts:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(parts))

    def frame_upto(self, symbol: str, ts_ns: int, limit: int | None = None) -> pd.DataFrame:
        """Bars with open time <= ts_ns (i.e. bars completed by the current decision time)."""
        s = self.ensure(symbol)
        i = s.index_le(ts_ns)
        if i is None:
            return s.frame.iloc[0:0]
        lo = 0 if limit is None else max(0, i + 1 - limit)
        return s.frame.iloc[lo: i + 1]


def ns_to_dt(ns: int) -> datetime:
    return pd.Timestamp(ns, tz="UTC").to_pydatetime()


def dt_to_ns(dt: datetime) -> int:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value)


