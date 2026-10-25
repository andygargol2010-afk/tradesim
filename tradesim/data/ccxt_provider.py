"""Crypto exchanges via ``ccxt`` (public endpoints need no API key).

Gives real OHLCV, real top-of-book (bid/ask) and real level-2 order books, the
best possible inputs for crypto execution realism.

Exchange notes:

* ``binance``: deep history and books, but geo-blocked in some regions (HTTP 451).
* ``coinbase`` / ``kraken``: good alternatives. Kraken's OHLC endpoint only
  returns the most recent 720 candles per timeframe; use it for live/paper
  trading and recent windows, not for long backtests.
* Prices quoted in USDT/USDC are treated as USD (basis usually < 0.1%).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError, OrderBookSnapshot
from tradesim.data.timeframes import Timeframe, normalize_bars
from tradesim.types import AssetClass, Quote

log = logging.getLogger(__name__)


class CCXTProvider(DataProvider):
    """Args:
        exchange_id: any ccxt id (``binance``, ``coinbase``, ``kraken``, ``okx``, ``bybit``...).
        quote_preference: quote currencies tried in order when mapping ``BTC-USD``.
        api_key / secret: optional; only needed for private endpoints (not used here).
        trust_env: honour ``HTTPS_PROXY`` / ``REQUESTS_CA_BUNDLE`` from the environment
            (ccxt ignores them by default), needed behind corporate proxies.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        quote_preference: tuple[str, ...] = ("USD", "USDT", "USDC"),
        api_key: str | None = None,
        secret: str | None = None,
        page_limit: int = 1000,
        trust_env: bool = True,
    ):
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("pip install ccxt  (or: pip install 'tradesim[crypto]')") from exc
        config = {"enableRateLimit": True, "requests_trust_env": trust_env}
        if api_key:
            config.update({"apiKey": api_key, "secret": secret})
        self.exchange = getattr(ccxt, exchange_id)(config)
        self.name = f"ccxt:{exchange_id}"
        self.quote_preference = quote_preference
        self.page_limit = page_limit
        self._markets: dict | None = None
        self._symbol_map: dict[str, str | None] = {}

    # ------------------------------------------------------------------ mapping
    def _load_markets(self) -> dict:
        if self._markets is None:
            try:
                self._markets = self.exchange.load_markets()
            except Exception as exc:
                raise DataUnavailableError(f"{self.name}: cannot load markets ({exc})") from exc
        return self._markets

    def market_symbol(self, asset: Asset) -> str | None:
        if asset.symbol in self._symbol_map:
            return self._symbol_map[asset.symbol]
        markets = self._load_markets()
        base = asset.symbol.split("-")[0].upper()
        found = None
        for quote in self.quote_preference:
            cand = f"{base}/{quote}"
            m = markets.get(cand)
            if m and m.get("spot", True) and m.get("active", True) is not False:
                found = cand
                break
        self._symbol_map[asset.symbol] = found
        return found

    def supports(self, asset: Asset) -> bool:
        if asset.asset_class is not AssetClass.CRYPTO:
            return False
        try:
            return self.market_symbol(asset) is not None
        except DataUnavailableError:
            return False

    def _require(self, asset: Asset) -> str:
        sym = self.market_symbol(asset)
        if sym is None:
            raise DataUnavailableError(f"{self.name} has no USD-like spot market for {asset.symbol}")
        return sym

    # --------------------------------------------------------------------- API
    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        sym = self._require(asset)
        if tf.value not in (self.exchange.timeframes or {}):
            raise DataUnavailableError(f"{self.name} does not support timeframe {tf.value}")
        since = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows: list[list[float]] = []
        empty_jumps = 0
        while since < end_ms:
            try:
                batch = self.exchange.fetch_ohlcv(sym, tf.value, since=since, limit=self.page_limit)
            except Exception as exc:
                raise DataUnavailableError(f"{self.name} fetch_ohlcv({sym}): {exc}") from exc
            if not batch:
                # Nothing in this window (e.g. the market was listed later): jump ahead a window.
                empty_jumps += 1
                if rows or empty_jumps > 40:
                    break
                since += min(self.page_limit, 300) * tf.seconds * 1000
                continue
            rows.extend(batch)
            last = batch[-1][0]
            if last < since:  # exchange ignored `since` (e.g. Kraken's 720-candle window)
                break
            since = last + tf.seconds * 1000
            if len(batch) < 2:
                break
        if not rows:
            raise DataUnavailableError(f"{self.name}: no {tf.value} bars for {sym}")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        out = normalize_bars(df)
        now = pd.Timestamp.now(tz="UTC")
        out = out[out.index + pd.Timedelta(seconds=tf.seconds) <= now]  # drop the still-forming candle
        return out[(out.index >= pd.Timestamp(start)) & (out.index < pd.Timestamp(end))]

    def get_quote(self, asset: Asset) -> Quote:
        sym = self._require(asset)
        try:
            t = self.exchange.fetch_ticker(sym)
        except Exception as exc:
            raise DataUnavailableError(f"{self.name} fetch_ticker({sym}): {exc}") from exc
        bid, ask, last = t.get("bid"), t.get("ask"), t.get("last") or t.get("close")
        observed = bool(bid and ask)
        ts = datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc) if t.get("timestamp") else datetime.now(timezone.utc)
        return Quote(
            symbol=asset.symbol, timestamp=ts,
            bid=float(bid if observed else last), ask=float(ask if observed else last), last=float(last),
            bid_size=t.get("bidVolume"), ask_size=t.get("askVolume"), modelled=not observed,
        )

    def get_order_book(self, asset: Asset, depth: int = 50) -> OrderBookSnapshot:
        sym = self._require(asset)
        try:
            ob = self.exchange.fetch_order_book(sym, limit=depth)
        except Exception as exc:
            raise DataUnavailableError(f"{self.name} fetch_order_book({sym}): {exc}") from exc
        ts = datetime.fromtimestamp(ob["timestamp"] / 1000, tz=timezone.utc) if ob.get("timestamp") else datetime.now(timezone.utc)
        return OrderBookSnapshot(
            symbol=asset.symbol, timestamp=ts,
            bids=[(float(p), float(s)) for p, s, *_ in ob["bids"]],
            asks=[(float(p), float(s)) for p, s, *_ in ob["asks"]],
        )

    @property
    def supports_live(self) -> bool:
        return True

    def sleep_for_rate_limit(self) -> None:
        time.sleep(self.exchange.rateLimit / 1000)
