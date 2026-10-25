"""Market-data provider interface."""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from typing import Iterable

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.timeframes import Timeframe
from tradesim.types import Quote

log = logging.getLogger(__name__)


class DataUnavailableError(RuntimeError):
    """Raised when a provider cannot serve a request (unknown symbol, rate limit, no network...)."""


class OrderBookSnapshot:
    """Level-2 book: ``bids`` / ``asks`` as lists of ``(price, size)``, best first."""

    def __init__(self, symbol: str, timestamp: datetime, bids: list[tuple[float, float]], asks: list[tuple[float, float]]):
        self.symbol = symbol
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks

    def to_dict(self, depth: int = 10) -> dict:
        return {"symbol": self.symbol, "timestamp": self.timestamp.isoformat(),
                "bids": self.bids[:depth], "asks": self.asks[:depth]}


class DataProvider(abc.ABC):
    """Source of historical bars and (optionally) live quotes / order books.

    Implementations must return bars in the canonical layout produced by
    :func:`tradesim.data.timeframes.normalize_bars` (UTC index = bar open time).
    """

    name: str = "base"

    @abc.abstractmethod
    def supports(self, asset: Asset) -> bool:
        """Whether this provider can serve ``asset``."""

    @abc.abstractmethod
    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        """Historical OHLCV bars in ``[start, end)``."""

    def get_bars_many(
        self, assets: Iterable[Asset], timeframe: Timeframe, start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]:
        """Batch download. Default: sequential; providers with bulk endpoints override."""
        out: dict[str, pd.DataFrame] = {}
        for asset in assets:
            try:
                out[asset.symbol] = self.get_bars(asset, timeframe, start, end)
            except DataUnavailableError as exc:
                log.warning("%s: no data for %s (%s)", self.name, asset.symbol, exc)
        return out

    def get_quote(self, asset: Asset) -> Quote:
        """Latest top-of-book. Providers without live quotes raise ``DataUnavailableError``."""
        raise DataUnavailableError(f"{self.name} does not provide live quotes")

    def get_order_book(self, asset: Asset, depth: int = 50) -> OrderBookSnapshot:
        raise DataUnavailableError(f"{self.name} does not provide order books")

    @property
    def supports_live(self) -> bool:
        return False


class RoutingProvider(DataProvider):
    """Route each asset to the first provider in ``providers`` that supports it.

    Example: ``RoutingProvider([CCXTProvider("kraken"), YFinanceProvider()])`` serves
    crypto from Kraken and everything else from Yahoo.
    """

    name = "router"

    def __init__(self, providers: list[DataProvider]):
        if not providers:
            raise ValueError("RoutingProvider needs at least one provider")
        self.providers = providers

    def _pick(self, asset: Asset) -> DataProvider:
        for p in self.providers:
            if p.supports(asset):
                return p
        raise DataUnavailableError(f"No provider supports {asset.symbol}")

    def supports(self, asset: Asset) -> bool:
        return any(p.supports(asset) for p in self.providers)

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        return self._pick(asset).get_bars(asset, timeframe, start, end)

    def get_bars_many(self, assets, timeframe, start, end):
        groups: dict[int, list[Asset]] = {}
        for a in assets:
            for i, p in enumerate(self.providers):
                if p.supports(a):
                    groups.setdefault(i, []).append(a)
                    break
        out: dict[str, pd.DataFrame] = {}
        for i, group in groups.items():
            out.update(self.providers[i].get_bars_many(group, timeframe, start, end))
        return out

    def get_quote(self, asset: Asset) -> Quote:
        return self._pick(asset).get_quote(asset)

    def get_order_book(self, asset: Asset, depth: int = 50) -> OrderBookSnapshot:
        return self._pick(asset).get_order_book(asset, depth)

    @property
    def supports_live(self) -> bool:
        return any(p.supports_live for p in self.providers)
