"""Serve bars from in-memory DataFrames (your own CSVs, cached downloads, or history an agent already has)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe, normalize_bars, resample_bars


class InMemoryProvider(DataProvider):
    """``DataProvider`` over ``{symbol: OHLCV DataFrame}``.

    Frames are normalised once (UTC index = bar open time). Requests for a coarser timeframe
    than ``timeframe`` are resampled; finer ones are refused.

    Example::

        frames = {"AAPL": pd.read_csv("aapl.csv", index_col=0, parse_dates=True)}
        sim = TradingSimulator(provider=InMemoryProvider(frames), symbols=["AAPL"], start=..., end=...)
    """

    name = "memory"

    def __init__(self, frames: dict[str, pd.DataFrame], timeframe: Timeframe | str = Timeframe.D1):
        self.timeframe = Timeframe.parse(timeframe)
        self.frames = {sym: normalize_bars(df) for sym, df in frames.items() if df is not None and len(df)}

    def supports(self, asset: Asset) -> bool:
        return asset.symbol in self.frames

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        df = self.frames.get(asset.symbol)
        if df is None:
            raise DataUnavailableError(f"no in-memory data for {asset.symbol}")
        if tf.seconds < self.timeframe.seconds:
            raise DataUnavailableError(f"in-memory data is {self.timeframe.value}; cannot serve {tf.value}")
        if tf is not self.timeframe:
            df = resample_bars(df, tf)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return df[(df.index >= s) & (df.index < e)]
