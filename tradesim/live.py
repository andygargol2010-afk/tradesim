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
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from dateutil.easter import easter

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
from tradesim.types import Fill, Order, OrderStatus, OrderType, Quote, TimeInForce

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def nyse_holidays(year: int) -> frozenset[date]:
    """NYSE full-day closures for ``year`` (rules in force since 2022; early closes are not modelled).

    Saturday holidays are observed on Friday and Sunday holidays on Monday, except New Year's Day
    falling on a Saturday, which is not observed (NYSE rule 7.2).
    """
    def nth_weekday(month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last_weekday(month: int, weekday: int) -> date:
        d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d: date, saturday_rolls_back: bool = True) -> date | None:
        if d.weekday() == 5:
            return d - timedelta(days=1) if saturday_rolls_back else None
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    days = [
        observed(date(year, 1, 1), saturday_rolls_back=False),
        nth_weekday(1, 0, 3),  # Martin Luther King Jr. Day
        nth_weekday(2, 0, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        last_weekday(5, 0),  # Memorial Day
        observed(date(year, 6, 19)),  # Juneteenth
        observed(date(year, 7, 4)),
        nth_weekday(9, 0, 1),  # Labor Day
        nth_weekday(11, 3, 4),  # Thanksgiving
        observed(date(year, 12, 25)),
    ]
    return frozenset(d for d in days if d is not None)


def is_market_open(asset: Asset, now: datetime) -> bool:
    """Whether ``asset`` trades at ``now``: 24/7 crypto, 24/5 FX, NYSE regular session (holidays included)."""
    session = asset.effective_session
    if session == "24/7":
        return True
    local = now.astimezone(NY)
    if session == "24/5":  # FX: Sunday 17:00 -> Friday 17:00 New York
        wd, minutes = local.weekday(), local.hour * 60 + local.minute
        if wd == 5 or (wd == 4 and minutes >= 17 * 60) or (wd == 6 and minutes < 17 * 60):
            return False
        return True
    if local.weekday() >= 5 or local.date() in nyse_holidays(local.year):
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
        clock: returns the current UTC time (default: the wall clock); inject one for tests / replays.
        symbols: the symbols agents may trade (what ``observe()`` reports).
        path_timeframes: intraday bars used to replay what happened to resting stop / limit orders
            between two syncs (first available wins); empty disables the replay.

    Resting orders (stop-loss, take-profit, limits) are checked on every :meth:`sync` against the
    intraday bars since the previous check, bar by bar, so they fill at the price and time the market
    actually reached them (gaps included), as a broker's resting order would, even if the account is
    only synced a few times a day.
    """

    def __init__(self, provider: DataProvider, universe: AssetUniverse | None = None,
                 config: SimulatorConfig | None = None, history_provider: DataProvider | None = None,
                 quote_ttl: float = 2.0, use_order_book: bool = True,
                 clock: Callable[[], datetime] | None = None, symbols: list[str] | None = None,
                 path_timeframes: tuple[str, ...] = ("5m", "1h")):
        self._clock = clock
        super().__init__(universe, provider, config)
        self.history_provider = history_provider or provider
        self.quote_ttl = quote_ttl
        self.use_order_book = use_order_book
        self.symbols: list[str] = list(symbols or [])
        self._quotes: dict[str, tuple[float, Quote]] = {}
        self._stats: dict[str, tuple[float, float, float]] = {}  # symbol -> (fetched_at, sigma_bar, typical_volume)
        self._last_sync: datetime | None = None
        self.path_timeframes = tuple(path_timeframes)
        self._path_checked: dict[str, str] = {}  # symbol -> ISO time up to which resting orders were replayed
        self._record_equity(self.now)

    @property
    def now(self) -> datetime:
        return self._clock() if getattr(self, "_clock", None) else datetime.now(timezone.utc)

    def is_open(self, symbol: str) -> bool:
        """Whether ``symbol``'s market is open right now."""
        return is_market_open(self.universe.get(symbol), self.now)

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
    def _settle(self) -> None:
        if self.open_order_ids:
            self.sync()

    def sync(self) -> list[dict[str, Any]]:
        """Execute open orders against fresh quotes / books, mark to market, record equity."""
        now = self.now
        fills: list[Fill] = []
        for sym in list(dict.fromkeys(self.orders[i].symbol for i in self.open_order_ids)):
            asset = self.universe.get(sym)
            fills.extend(self._replay_resting(asset, now))
            if not is_market_open(asset, now) or not self._open_orders(sym):
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
        self.protect_positions()
        return [f.to_dict() for f in fills]

    def _replay_resting(self, asset: Asset, now: datetime) -> list[Fill]:
        """Walk the intraday bars since the last check through the resting (non-market) orders."""
        sym = asset.symbol
        resting = [o for o in self._open_orders(sym) if o.order_type is not OrderType.MARKET
                   and o.created_at is not None and o.created_at < now]
        if not resting or not self.path_timeframes:
            return []
        first = min(o.created_at for o in resting)
        checked = self._path_checked.get(sym)
        start = max(first, datetime.fromisoformat(checked)) if checked else first
        bars, tf = None, None
        for name in self.path_timeframes:
            try:
                tf = Timeframe.parse(name)
                bars = self.history_provider.get_bars(asset, tf, start - timedelta(seconds=tf.seconds), now)
                break
            except (DataUnavailableError, ValueError) as exc:
                log.debug("no %s bars for %s: %s", name, sym, exc)
        if bars is None or tf is None or bars.empty:
            return []
        step = pd.Timedelta(seconds=tf.seconds)
        bars = bars[(bars.index >= pd.Timestamp(start)) & (bars.index + step <= pd.Timestamp(now))]
        if bars.empty:
            return []
        sigma_day, typical_day = self._bar_stats(asset)
        session = 86400.0 if asset.is_24_7 else 6.5 * 3600
        frac = min(1.0, tf.seconds / session)
        prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
        sigma_typ = prof.annual_vol / math.sqrt(self.timeframe.bars_per_year(asset.is_24_7)) * math.sqrt(frac)
        fills: list[Fill] = []
        for ts, row in bars.iterrows():
            end = (ts + step).to_pydatetime()
            orders = [o for o in self._open_orders(sym) if o.order_type is not OrderType.MARKET
                      and o.created_at is not None and o.created_at <= ts.to_pydatetime()]
            if not orders:
                continue
            vol = float(row.get("volume", float("nan")))
            ctx = MarketContext(asset=asset, timestamp=end, open=float(row["open"]), high=float(row["high"]),
                                low=float(row["low"]), close=float(row["close"]),
                                volume=vol if vol > 0 else float("nan"), sigma=sigma_day * math.sqrt(frac),
                                sigma_typical=sigma_typ, typical_volume=typical_day * frac)
            fills.extend(self._execute_symbol(asset, ctx, orders))
        self._path_checked[sym] = (bars.index[-1] + step).isoformat()
        return fills

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
    def state(self) -> dict[str, Any]:
        """Everything needed to resume this account (JSON-serialisable)."""
        return {
            "version": 1, "kind": "live", "run_id": self.run_id, "config": self.config.to_dict(),
            "portfolio": self.portfolio.state(), "orders": [o.to_dict() for o in self.orders.values()],
            "open_order_ids": self.open_order_ids, "order_id_counter": self.order_ids.counter,
            "fill_id_counter": self.fill_ids.counter, "rng_exec": self.rng_exec.bit_generator.state,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None, "bankrupt": self.bankrupt,
            "path_checked": self._path_checked,
        }

    def restore(self, st: dict[str, Any]) -> None:
        """Load what :meth:`state` returned into this instance."""
        self.run_id = st["run_id"]
        self.portfolio = Portfolio.from_state(st["portfolio"])
        self.orders = {d["id"]: Order.from_dict(d) for d in st["orders"]}
        self.open_order_ids = list(st["open_order_ids"])
        self.order_ids.counter = st["order_id_counter"]
        self.fill_ids.counter = st["fill_id_counter"]
        self.rng_exec.bit_generator.state = st["rng_exec"]
        self._last_sync = datetime.fromisoformat(st["last_sync"]) if st.get("last_sync") else None
        self.bankrupt = st["bankrupt"]
        self._path_checked = dict(st.get("path_checked") or {})

    def save_state(self, path: str | Path) -> None:
        write_json(path, self.state())

    @classmethod
    def load_state(cls, path: str | Path, provider: DataProvider, universe: AssetUniverse | None = None,
                   config: SimulatorConfig | None = None, **kw: Any) -> "LivePaperTrader":
        st = read_json(path)
        trader = cls(provider, universe, config or SimulatorConfig.from_dict(st["config"]), **kw)
        trader.restore(st)
        return trader
