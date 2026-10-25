"""Data plumbing for live runs: a per-run cache and the default live data source."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError, OrderBookSnapshot, RoutingProvider
from tradesim.data.timeframes import Timeframe
from tradesim.types import Quote

log = logging.getLogger(__name__)


class RunCacheProvider(DataProvider):
    """Memoises a provider for the duration of one run.

    Every account in a run sees exactly the same quotes, books and bars (a fair comparison), and each
    series is downloaded once. Bars are cached per ``(symbol, timeframe)`` as the widest range fetched;
    quotes and books are reused for ``quote_ttl`` seconds. Failures are cached too, so a symbol whose
    feed is down fails fast for every account instead of retrying N times.
    """

    def __init__(self, base: DataProvider, quote_ttl: float = 120.0):
        self.base = base
        self.name = f"run-cache:{base.name}"
        self.quote_ttl = quote_ttl
        self._bars: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = {}
        self._quotes: dict[str, tuple[float, Quote | Exception]] = {}
        self._books: dict[str, tuple[float, OrderBookSnapshot | Exception]] = {}
        self.errors: dict[str, str] = {}

    def supports(self, asset: Asset) -> bool:
        return self.base.supports(asset)

    @property
    def supports_live(self) -> bool:
        return self.base.supports_live

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        key = (asset.symbol, tf.value)
        hit = self._bars.get(key)
        if hit is None or s < hit[0] or e > hit[1] + pd.Timedelta(minutes=10):
            lo = min(s, hit[0]) if hit else s
            hi = max(e, hit[1]) if hit else e
            try:
                df = self.base.get_bars(asset, tf, lo.to_pydatetime(), hi.to_pydatetime())
            except DataUnavailableError as exc:
                self.errors[f"bars:{asset.symbol}"] = str(exc)
                raise
            hit = (lo, hi, df)
            self._bars[key] = hit
        df = hit[2]
        return df[(df.index >= s) & (df.index < e)]

    def _memo(self, store: dict, key: str, fetch):
        cached = store.get(key)
        if cached and time.monotonic() - cached[0] < self.quote_ttl:
            if isinstance(cached[1], Exception):
                raise cached[1]
            return cached[1]
        try:
            value = fetch()
        except DataUnavailableError as exc:
            store[key] = (time.monotonic(), exc)
            raise
        store[key] = (time.monotonic(), value)
        return value

    def get_quote(self, asset: Asset) -> Quote:
        try:
            return self._memo(self._quotes, asset.symbol, lambda: self.base.get_quote(asset))
        except DataUnavailableError as exc:
            self.errors[f"quote:{asset.symbol}"] = str(exc)
            raise

    def get_order_book(self, asset: Asset, depth: int = 50) -> OrderBookSnapshot:
        return self._memo(self._books, asset.symbol, lambda: self.base.get_order_book(asset, depth))


def default_live_provider(crypto_exchange: str | None = "coinbase", stocks: str = "auto") -> DataProvider:
    """Live data with sensible fallbacks.

    * crypto: real bid/ask and L2 books from ``crypto_exchange`` via ccxt (no key), else Yahoo;
    * stocks/ETFs: Alpaca (real bid/ask) when ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` are set and
      ``stocks`` is ``"auto"`` or ``"alpaca"``, else Yahoo Finance (last price, modelled spread).
    """
    providers: list[DataProvider] = []
    if crypto_exchange:
        try:
            from tradesim.data.ccxt_provider import CCXTProvider

            providers.append(CCXTProvider(crypto_exchange))
        except ImportError:
            log.warning("ccxt not installed: crypto falls back to Yahoo")
    if stocks in ("auto", "alpaca") and os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
        try:
            from tradesim.data.alpaca_provider import AlpacaProvider

            providers.append(AlpacaProvider())
        except (ImportError, DataUnavailableError) as exc:
            log.warning("Alpaca data unavailable (%s): stocks fall back to Yahoo", exc)
    elif stocks == "alpaca":
        log.warning("stocks='alpaca' but ALPACA_API_KEY / ALPACA_SECRET_KEY are not set: using Yahoo")
    from tradesim.data.yfinance_provider import YFinanceProvider

    providers.append(YFinanceProvider())
    return RoutingProvider(providers)
