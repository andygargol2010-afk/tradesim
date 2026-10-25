"""Technical indicators (pandas, causal: value at bar i uses bars <= i only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * sd, "bb_lower": mid - k * sd,
                         "bb_pct_b": (close - (mid - k * sd)) / (2 * k * sd)})


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV ``df`` with a standard indicator set appended."""
    out = df.copy()
    c = out["close"]
    out["ret_1"] = c.pct_change()
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["ema_12"] = ema(c, 12)
    out["ema_26"] = ema(c, 26)
    out["rsi_14"] = rsi(c, 14)
    out["atr_14"] = atr(out, 14)
    out["vol_20"] = realized_vol(c, 20)
    out = pd.concat([out, macd(c), bollinger(c)], axis=1)
    return out
