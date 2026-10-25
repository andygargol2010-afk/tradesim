"""Bar timeframes and OHLCV resampling."""

from __future__ import annotations

from enum import Enum

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        return _SECONDS[self]

    @property
    def pandas_freq(self) -> str:
        return _PANDAS_FREQ[self]

    @classmethod
    def parse(cls, value: "Timeframe | str") -> "Timeframe":
        if isinstance(value, Timeframe):
            return value
        v = value.strip().lower()
        aliases = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "60m": "1h", "1hour": "1h",
                   "4hour": "4h", "240m": "4h", "1day": "1d", "d": "1d", "day": "1d", "1wk": "1w", "w": "1w"}
        return cls(aliases.get(v, v))

    def bars_per_year(self, is_24_7: bool) -> float:
        """Annualisation factor for metrics (252 trading days x 6.5h for US stocks, 365 x 24h for crypto)."""
        if self is Timeframe.D1:
            return 365.0 if is_24_7 else 252.0
        if self is Timeframe.W1:
            return 52.0
        seconds_per_year = 365 * 86400 if is_24_7 else 252 * 6.5 * 3600
        return seconds_per_year / self.seconds


_SECONDS = {
    Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M15: 900, Timeframe.M30: 1800,
    Timeframe.H1: 3600, Timeframe.H4: 14400, Timeframe.D1: 86400, Timeframe.W1: 604800,
}
_PANDAS_FREQ = {
    Timeframe.M1: "1min", Timeframe.M5: "5min", Timeframe.M15: "15min", Timeframe.M30: "30min",
    Timeframe.H1: "1h", Timeframe.H4: "4h", Timeframe.D1: "1D", Timeframe.W1: "W-MON",
}


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider frame to the canonical layout.

    * ``DatetimeIndex`` named ``timestamp`` in UTC, the bar *open* time, sorted, unique.
    * float columns ``open, high, low, close, volume`` (+ optional ``bid, ask`` etc.).
    * Rows with non-positive / missing prices dropped; OHLC made self-consistent.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "adj close" in out.columns and "close" not in out.columns:
        out = out.rename(columns={"adj close": "close"})
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).as_unit("ns")
    out.index = idx.rename("timestamp")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if "volume" not in out.columns:
        out["volume"] = float("nan")
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # Some feeds occasionally report high < max(open, close) etc. Repair rather than drop.
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    extra = [c for c in out.columns if c not in OHLCV_COLUMNS]
    return out[OHLCV_COLUMNS + extra]


def resample_bars(df: pd.DataFrame, timeframe: Timeframe | str) -> pd.DataFrame:
    """Aggregate OHLCV bars to a coarser ``timeframe`` (bars labelled by their open time).

    4h bars are anchored at 00:00 UTC (as on Binance and most crypto venues).
    Empty intervals (market closed) are dropped, not forward-filled.
    """
    tf = Timeframe.parse(timeframe)
    if df.empty:
        return df
    agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in df.columns:
        if c not in agg:
            agg[c] = "last"
    out = df.resample(tf.pandas_freq, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "close"])
    return out
