"""Alpaca Market Data (``alpaca-py``): US stocks/ETFs and crypto, with real bid/ask quotes.

Requires a (free) Alpaca account. Keys are read from the environment::

    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...

The free plan serves the IEX feed (``feed="iex"``), roughly 2-3% of US volume:
prices are fine, but volumes are a fraction of the consolidated tape, so the
provider rescales IEX volume by ``iex_volume_scale`` before it reaches the
liquidity model. With a paid SIP subscription use ``feed="sip"`` and scale 1.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe, normalize_bars, resample_bars
from tradesim.types import AssetClass, Quote


class AlpacaProvider(DataProvider):
    name = "alpaca"

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, feed: str = "iex",
                 iex_volume_scale: float = 40.0):
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("pip install alpaca-py  (or: pip install 'tradesim[alpaca]')") from exc
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise DataUnavailableError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (see README).")
        self.stocks = StockHistoricalDataClient(api_key, secret_key)
        self.crypto = CryptoHistoricalDataClient(api_key, secret_key)
        self.feed = feed
        self.volume_scale = iex_volume_scale if feed == "iex" else 1.0

    def supports(self, asset: Asset) -> bool:
        return asset.asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO)

    @staticmethod
    def _alpaca_symbol(asset: Asset) -> str:
        return asset.symbol.replace("-USD", "/USD") if asset.asset_class is AssetClass.CRYPTO else asset.symbol.replace("-", ".")

    @staticmethod
    def _timeframe(tf: Timeframe):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        return {
            Timeframe.M1: TimeFrame(1, TimeFrameUnit.Minute), Timeframe.M5: TimeFrame(5, TimeFrameUnit.Minute),
            Timeframe.M15: TimeFrame(15, TimeFrameUnit.Minute), Timeframe.M30: TimeFrame(30, TimeFrameUnit.Minute),
            Timeframe.H1: TimeFrame(1, TimeFrameUnit.Hour), Timeframe.H4: TimeFrame(1, TimeFrameUnit.Hour),
            Timeframe.D1: TimeFrame(1, TimeFrameUnit.Day), Timeframe.W1: TimeFrame(1, TimeFrameUnit.Week),
        }[tf]

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

        tf = Timeframe.parse(timeframe)
        sym = self._alpaca_symbol(asset)
        try:
            if asset.asset_class is AssetClass.CRYPTO:
                req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=self._timeframe(tf), start=start, end=end)
                raw = self.crypto.get_crypto_bars(req).df
                scale = 1.0
            else:
                req = StockBarsRequest(symbol_or_symbols=sym, timeframe=self._timeframe(tf), start=start, end=end,
                                       adjustment=Adjustment.ALL, feed=self.feed)
                raw = self.stocks.get_stock_bars(req).df
                scale = self.volume_scale
        except Exception as exc:
            raise DataUnavailableError(f"alpaca bars {sym}: {exc}") from exc
        if raw is None or raw.empty:
            raise DataUnavailableError(f"alpaca returned no bars for {sym}")
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.xs(sym, level=0)
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df["volume"] *= scale
        out = normalize_bars(df)
        if tf in (Timeframe.D1, Timeframe.W1):
            out.index = out.index.tz_convert("America/New_York").tz_localize(None).normalize().tz_localize("UTC")
        return resample_bars(out, tf) if tf is Timeframe.H4 else out

    def get_quote(self, asset: Asset) -> Quote:
        from alpaca.data.requests import CryptoLatestQuoteRequest, StockLatestQuoteRequest

        sym = self._alpaca_symbol(asset)
        try:
            if asset.asset_class is AssetClass.CRYPTO:
                q = self.crypto.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=sym))[sym]
            else:
                q = self.stocks.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym, feed=self.feed))[sym]
        except Exception as exc:
            raise DataUnavailableError(f"alpaca quote {sym}: {exc}") from exc
        bid, ask = float(q.bid_price), float(q.ask_price)
        if not (bid > 0 and ask >= bid):
            raise DataUnavailableError(f"alpaca quote {sym}: crossed/empty book ({bid}/{ask})")
        ts = q.timestamp if q.timestamp.tzinfo else q.timestamp.replace(tzinfo=timezone.utc)
        return Quote(symbol=asset.symbol, timestamp=ts, bid=bid, ask=ask, last=(bid + ask) / 2,
                     bid_size=float(q.bid_size), ask_size=float(q.ask_size), modelled=False)

    @property
    def supports_live(self) -> bool:
        return True
