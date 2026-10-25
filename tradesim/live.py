"""Live (near real-time) paper trading on real quotes, with the same broker semantics.

The clock is the wall clock. Each :meth:`LivePaperTrader.sync` call:

1. fetches the latest quote (real bid/ask when the provider has it) and, for
   crypto via ccxt, the real L2 order book for every symbol with open orders;
2. executes open orders against that snapshot with the same engine as the
   backtester (market orders sweep the real book; resting limits fill only if
   the last trade prints through them);
3. marks positions at the mid, accrues financing, records equity and checks margin.

US equities only execute during the regular session (09:30-16:00 New York,
Mon-Fri; exchange holidays are not modelled). Orders placed outside it wait
for the open, as at a real broker.

Example::

    from tradesim.data.ccxt_provider import CCXTProvider
    trader = LivePaperTrader(CCXTProvider("kraken"), config=SimulatorConfig(initial_cash=10_000, fees="kraken"))
    trader.place_order("BTC-USD", "buy", notional=500)
    trader.sync()
    print(trader.get_portfolio())
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe
from tradesim.execution.engine import MarketContext
from tradesim.indicators import add_indicators
from tradesim.market.liquidity import liquidity_profile
from tradesim.persistence import read_json, write_json
from tradesim.portfolio.portfolio import Portfolio
from tradesim.simulator import NY, BaseSimulator, SimulatorConfig
from tradesim.types import Fill, Order, OrderStatus, Quote, TimeInForce

log = logging.getLogger(__name__)


def is_market_open(asset: Asset, now: datetime) -> bool:
    session = asset.effective_session
    if session == "24/7":
        return True
    local = now.astimezone(NY)
    if session == "24/5":  # FX: Sunday 17:00 -> Friday 17:00 New York
        wd, minutes = local.weekday(), local.hour * 60 + local.minute
        if wd == 5 or (wd == 4 and minutes >= 17 * 60) or (wd == 6 and minutes < 17 * 60):
            return False
        return True
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


class LivePaperTrader(BaseSimulator):
    """Paper trading against live market data.

    Args:
        provider: a provider with live quotes (``CCXTProvider``, ``AlpacaProvider``,
            ``YFinanceProvider`` or a ``RoutingProvider`` of them).
        history_provider: provider for bars (volatility / volume estimates and
            ``get_historical_data``); defaults to ``provider``.
        quote_ttl: seconds a fetched quote is reused by ``get_price``.
        use_order_book: fetch L2 books for execution when the provider supports it.
    """

    def __init__(self, provider: DataProvider, universe: AssetUniverse | None = None,
                 config: SimulatorConfig | None = None, history_provider: DataProvider | None = None,
                 quote_ttl: float = 2.0, use_order_book: bool = True):
        super().__init__(universe, provider, config)
        self.history_provider = history_provider or provider
        self.quote_ttl = quote_ttl
        self.use_order_book = use_order_book
        self._quotes: dict[str, tuple[float, Quote]] = {}
        self._stats: dict[str, tuple[float, float, float]] = {}  # symbol -> (fetched_at, sigma_bar, typical_volume)
        self._last_sync: datetime | None = None
        self._record_equity(self.now)

    @property
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    # --------------------------------------------------------------- market
    def _quote(self, asset: Asset) -> Quote:
        cached = self._quotes.get(asset.symbol)
        if cached and time.monotonic() - cached[0] < self.quote_ttl:
            return cached[1]
        q = self.provider.get_quote(asset)
        if q.modelled or q.bid <= 0 or q.ask < q.bid:
            # No real top of book: model the spread around the last trade, as in backtests.
            sigma, typical = self._bar_stats(asset)
            prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
            ctx = MarketContext(asset=asset, timestamp=q.timestamp, open=q.last, high=q.last, low=q.last,
                                close=q.last, volume=float("nan"), sigma=sigma,
                                sigma_typical=prof.annual_vol / math.sqrt(self.timeframe.bars_per_year(asset.is_24_7)),
                                typical_volume=typical, live=True)
            bid, ask = self.engine.touch(ctx, q.last, at_entry=False)
            q = Quote(symbol=q.symbol, timestamp=q.timestamp, bid=bid, ask=ask, last=q.last, modelled=True)
        self._quotes[asset.symbol] = (time.monotonic(), q)
        return q

    def _bar_stats(self, asset: Asset) -> tuple[float, float]:
        """Per-bar volatility and typical volume from recent bars (refreshed every 15 min)."""
        cached = self._stats.get(asset.symbol)
        if cached and time.monotonic() - cached[0] < 900:
            return cached[1], cached[2]
        prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
        bpy = self.timeframe.bars_per_year(asset.is_24_7)
        sigma = prof.annual_vol / math.sqrt(bpy)
        last_price = None
        typical = float("nan")
        try:
            end = self.now
            start = end - timedelta(seconds=self.timeframe.seconds * 120) - timedelta(days=4)
            bars = self.history_provider.get_bars(asset, self.timeframe, start, end).tail(60)
            if len(bars) >= 10:
                lr = np.diff(np.log(bars["close"].to_numpy()))
                sigma = float(np.std(lr[-20:], ddof=1)) or sigma
                vol = bars["volume"].replace(0, np.nan).dropna()
                if len(vol) >= 5:
                    typical = float(vol.tail(20).median())
                last_price = float(bars["close"].iloc[-1])
        except DataUnavailableError as exc:
            log.debug("bar stats for %s unavailable: %s", asset.symbol, exc)
        if not math.isfinite(typical):
            bars_per_day = bpy / (365.0 if asset.is_24_7 else 252.0)
            price = last_price or self.provider.get_quote(asset).last
            typical = prof.adv_usd / max(price, 1e-12) / bars_per_day
        self._stats[asset.symbol] = (time.monotonic(), sigma, typical)
        return sigma, typical

    def get_historical_data(self, symbol: str, timeframe: Timeframe | str | None = None, limit: int = 100,
                            indicators: bool = False) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe) if timeframe else self.timeframe
        asset = self.universe.get(symbol)
        extra = 60 if indicators else 0
        pad = 2.0 if asset.is_24_7 else 5.0
        end = self.now
        start = end - timedelta(seconds=tf.seconds * (limit + extra) * pad) - timedelta(days=4)
        df = self.history_provider.get_bars(asset, tf, start, end)
        now_ns = pd.Timestamp(end).value
        df = df[df.index.asi8 + tf.seconds * 1_000_000_000 <= now_ns][["open", "high", "low", "close", "volume"]]
        if indicators:
            df = add_indicators(df)
        return df.tail(limit)

    # -------------------------------------------------------------- execution
    def sync(self) -> list[dict[str, Any]]:
        """Execute open orders against fresh quotes / books, mark to market, record equity."""
        now = self.now
        fills: list[Fill] = []
        for sym in list(dict.fromkeys(self.orders[i].symbol for i in self.open_order_ids)):
            asset = self.universe.get(sym)
            if not is_market_open(asset, now):
                continue
            try:
                ctx = self._live_context(asset)
            except DataUnavailableError as exc:
                log.warning("skip %s this sync: %s", sym, exc)
                continue
            for o in self._open_orders(sym):
                if o.time_in_force is TimeInForce.DAY and o.created_at is not None \
                        and o.created_at.astimezone(NY).date() < now.astimezone(NY).date():
                    self._close_order(o, OrderStatus.EXPIRED, "DAY order expired")
            fills.extend(self._execute_symbol(asset, ctx))
        for pos in self.portfolio.open_positions():
            try:
                q = self._quote(self.universe.get(pos.symbol))
                self.portfolio.mark(pos.symbol, q.mid)
            except DataUnavailableError as exc:
                log.warning("cannot mark %s: %s", pos.symbol, exc)
        if self._last_sync is not None:
            self.portfolio.accrue_financing((now - self._last_sync).total_seconds())
        self._last_sync = now
        self._record_equity(now)
        self._check_margin()
        return [f.to_dict() for f in fills]

    def _live_context(self, asset: Asset) -> MarketContext:
        self._quotes.pop(asset.symbol, None)  # always use a fresh quote for execution
        q = self._quote(asset)
        sigma, typical = self._bar_stats(asset)
        book = None
        if self.use_order_book:
            try:
                book = self.provider.get_order_book(asset)
            except DataUnavailableError:
                book = None
        prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
        return MarketContext(
            asset=asset, timestamp=q.timestamp, open=q.last, high=q.last, low=q.last, close=q.last,
            volume=float("nan"), sigma=sigma,
            sigma_typical=prof.annual_vol / math.sqrt(self.timeframe.bars_per_year(asset.is_24_7)),
            typical_volume=typical, bid=None if q.modelled else q.bid, ask=None if q.modelled else q.ask,
            book=book, live=True,
        )

    def run(self, agent: Callable[["LivePaperTrader"], None], interval_seconds: float = 60.0,
            max_iterations: int | None = None) -> None:
        """Loop: ``agent(self)``, ``sync()``, sleep. Stop with Ctrl+C or ``max_iterations``."""
        it = 0
        try:
            while max_iterations is None or it < max_iterations:
                started = time.monotonic()
                agent(self)
                self.sync()
                it += 1
                time.sleep(max(0.0, interval_seconds - (time.monotonic() - started)))
        except KeyboardInterrupt:
            log.info("live loop interrupted")

    # ------------------------------------------------------------ persistence
    def save_state(self, path: str | Path) -> None:
        write_json(path, {
            "version": 1, "kind": "live", "run_id": self.run_id, "config": self.config.to_dict(),
            "portfolio": self.portfolio.state(), "orders": [o.to_dict() for o in self.orders.values()],
            "open_order_ids": self.open_order_ids, "order_id_counter": self.order_ids.counter,
            "fill_id_counter": self.fill_ids.counter, "rng_exec": self.rng_exec.bit_generator.state,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None, "bankrupt": self.bankrupt,
        })

    @classmethod
    def load_state(cls, path: str | Path, provider: DataProvider, universe: AssetUniverse | None = None,
                   config: SimulatorConfig | None = None, **kw: Any) -> "LivePaperTrader":
        st = read_json(path)
        trader = cls(provider, universe, config or SimulatorConfig.from_dict(st["config"]), **kw)
        trader.run_id = st["run_id"]
        trader.portfolio = Portfolio.from_state(st["portfolio"])
        trader.orders = {d["id"]: Order.from_dict(d) for d in st["orders"]}
        trader.open_order_ids = list(st["open_order_ids"])
        trader.order_ids.counter = st["order_id_counter"]
        trader.fill_ids.counter = st["fill_id_counter"]
        trader.rng_exec.bit_generator.state = st["rng_exec"]
        trader._last_sync = datetime.fromisoformat(st["last_sync"]) if st.get("last_sync") else None
        trader.bankrupt = st["bankrupt"]
        return trader
