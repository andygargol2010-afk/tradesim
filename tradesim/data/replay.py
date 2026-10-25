"""Replay historical (or synthetic) bars as if they were live: dry runs of the live pipeline.

``ReplayLiveProvider`` answers "live" questions at the time given by a clock: quotes are the last
completed bar's close, and bars never extend past the clock. Driving a
:class:`~tradesim.trading.runner.LiveRunner` with it rehearses weeks of daily operation in seconds,
which is how the live system is tested without waiting for real days to pass.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe
from tradesim.types import Quote


class ReplayLiveProvider(DataProvider):
    """Wrap any provider so that ``now`` is ``clock()``: no future bars, quotes from the last completed bar."""

    name = "replay"

    def __init__(self, base: DataProvider, clock: Callable[[], datetime], timeframe: Timeframe | str = "1d"):
        self.base = base
        self.clock = clock
        self.timeframe = Timeframe.parse(timeframe)

    def supports(self, asset: Asset) -> bool:
        return self.base.supports(asset)

    @property
    def supports_live(self) -> bool:
        return True

    def _completed(self, df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        now = pd.Timestamp(self.clock())
        return df[df.index + pd.Timedelta(seconds=tf.seconds) <= now]

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        end = min(pd.Timestamp(end), pd.Timestamp(self.clock())).to_pydatetime()
        return self._completed(self.base.get_bars(asset, tf, start, end), tf)

    def get_quote(self, asset: Asset) -> Quote:
        now = pd.Timestamp(self.clock())
        df = self.get_bars(asset, self.timeframe, (now - pd.Timedelta(days=10)).to_pydatetime(), now.to_pydatetime())
        if df.empty:
            raise DataUnavailableError(f"replay: no completed bar for {asset.symbol} before {now}")
        last = float(df["close"].iloc[-1])
        return Quote(symbol=asset.symbol, timestamp=now.to_pydatetime(), bid=last, ask=last, last=last, modelled=True)
