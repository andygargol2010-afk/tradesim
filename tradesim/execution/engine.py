"""Order execution engine.

The engine fills open orders against one *market context* per symbol: a
historical bar (backtest mode) or a live quote / L2 snapshot (paper mode).
Both go through the same code path; a live snapshot is just a bar whose
OHLC collapse to the last trade, with observed bid/ask and optional book.

Fill rules (designed to be conservative, i.e. to avoid flattering the agent):

* **No look-ahead.** Orders submitted after bar *t* closed are first
  evaluated on bar *t+1*. Volatility and typical-volume estimates used for
  spreads and impact come only from bars before the one being traded.
* **Intrabar path.** Within a bar, price is assumed to travel
  ``open -> low -> high -> close`` for an up bar and ``open -> high -> low -> close``
  for a down bar (the standard pessimistic-but-neutral convention).
* **Market orders** fill at the modelled ask (bid) around the bar open,
  plus market impact from sweeping the book, plus optional fixed/noise slippage.
* **Stops** trigger when the path touches the stop. If the bar *gaps*
  through the stop, the trigger price is the open (you get the gap, as in reality).
* **Limit orders** that are marketable on arrival take liquidity up to their
  limit (taker fee). Otherwise they rest and fill *at the limit* (maker fee)
  only if the path trades **through** the limit; a mere touch fills with
  probability ``limit_touch_fill_prob`` (queue-position uncertainty).
* **Capacity.** At most ``participation_rate`` of the bar's volume (or
  visible L2 depth) can be filled per bar, shared by all orders on that
  symbol and side. Unfilled quantity keeps working (GTC/DAY) or is
  cancelled (IOC). FOK fills completely or not at all.
* **Prices** are rounded to the tick size against the trader; quantities to
  the lot step toward zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Callable, Literal

import numpy as np

from tradesim.assets.universe import Asset
from tradesim.data.base import OrderBookSnapshot
from tradesim.execution.fees import FeeModel
from tradesim.market.liquidity import L2Book, SpreadModel, SqrtImpactBook, liquidity_profile
from tradesim.portfolio.portfolio import Portfolio
from tradesim.types import (
    ZERO, D, Fill, IdGenerator, Liquidity, Order, OrderStatus, OrderType, Side, TimeInForce,
    quantize_down, quantize_up,
)

SlippageModelName = Literal["sqrt_impact", "fixed_bps", "none"]


@dataclass
class ExecutionConfig:
    """Execution realism knobs.

    Attributes:
        slippage_model: ``"sqrt_impact"`` (spread + square-root market impact, default),
            ``"fixed_bps"`` (spread + constant ``fixed_slippage_bps``), or ``"none"``
            (fill at the reference price, no spread, no capacity limit; for ablations only).
        fixed_slippage_bps: extra adverse slippage added to every taker fill (all models).
        impact_coefficient: ``Y`` in ``impact = Y * sigma * sqrt(Q / V)``; 0.5-1.0 is typical.
        slippage_noise_bps: std-dev of random (symmetric) execution noise, for robustness training.
        participation_rate: max fraction of a bar's volume the agent can trade.
        limit_touch_fill_prob: probability a resting limit fills when price only touches it.
        spread: bid/ask spread model used when the feed has no quotes.
        use_observed_quotes: prefer real bid/ask when available (live mode).
        use_l2_book: sweep a real L2 book when available (live crypto).
    """

    slippage_model: SlippageModelName = "sqrt_impact"
    fixed_slippage_bps: float = 0.0
    impact_coefficient: float = 1.0
    slippage_noise_bps: float = 0.0
    participation_rate: float = 0.10
    limit_touch_fill_prob: float = 0.3
    spread: SpreadModel = field(default_factory=SpreadModel)
    use_observed_quotes: bool = True
    use_l2_book: bool = True


@dataclass
class MarketContext:
    """Everything the engine may know when executing on one symbol at one time step."""

    asset: Asset
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float  # units traded during the bar (nan if unknown)
    sigma: float  # per-bar log-return volatility estimated from *previous* bars
    sigma_typical: float  # long-run per-bar volatility for this asset class/tier
    typical_volume: float  # typical units per bar estimated from previous bars
    spread_mult: float = 1.0
    liq_mult: float = 1.0
    spread_bps_observed: float | None = None  # e.g. from a dataset with average quoted spread
    bid: float | None = None  # observed top of book (live)
    ask: float | None = None
    book: OrderBookSnapshot | None = None
    live: bool = False

    def path(self) -> list[float]:
        if self.live or (self.high == self.low):
            return [self.open] if self.open == self.close else [self.open, self.close]
        if self.close >= self.open:
            return [self.open, self.low, self.high, self.close]
        return [self.open, self.high, self.low, self.close]


# max quantity the account may still trade for this order at this price (None = unlimited)
MaxQtyFn = Callable[[Order, Asset, Decimal], Decimal | None]


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig, fee_model: FeeModel, rng: np.random.Generator,
                 fill_ids: IdGenerator | None = None):
        self.config = config
        self.fee_model = fee_model
        self.rng = rng
        self.fill_ids = fill_ids or IdGenerator("F")

    # ------------------------------------------------------------ microstructure
    def half_spread_frac(self, ctx: MarketContext) -> float:
        if self.config.slippage_model == "none":
            return 0.0
        prof = liquidity_profile(ctx.asset.asset_class, ctx.asset.liquidity_tier)
        if ctx.spread_bps_observed is not None and math.isfinite(ctx.spread_bps_observed):
            bps = ctx.spread_bps_observed * ctx.spread_mult
        else:
            bps = self.config.spread.spread_bps(prof, ctx.sigma, ctx.sigma_typical, ctx.spread_mult)
        return bps / 2.0 / 1e4

    def touch(self, ctx: MarketContext, price: float, at_entry: bool) -> tuple[float, float]:
        """Best bid/ask around reference ``price``. Observed quotes are used at the entry point."""
        if (at_entry and self.config.use_observed_quotes and ctx.bid is not None and ctx.ask is not None
                and 0 < ctx.bid <= ctx.ask and self.config.slippage_model != "none"):
            return ctx.bid, ctx.ask
        hs = self.half_spread_frac(ctx)
        tick = float(ctx.asset.tick_for_price(price))
        half = max(price * hs, tick / 2.0) if hs > 0 else 0.0
        return price - half, price + half

    def _book(self, ctx: MarketContext, bid: float, ask: float, at_entry: bool):
        cfg = self.config
        if at_entry and cfg.use_l2_book and ctx.book is not None and ctx.book.bids and ctx.book.asks \
                and cfg.slippage_model != "none":
            return L2Book(ctx.book)
        sigma = ctx.sigma if cfg.slippage_model == "sqrt_impact" else 0.0
        volume = ctx.typical_volume * ctx.liq_mult
        return SqrtImpactBook(bid=bid, ask=ask, sigma_bar=sigma, volume_bar=volume,
                              impact_coefficient=cfg.impact_coefficient)

    def capacity(self, ctx: MarketContext) -> float:
        """Max units fillable per side in this context."""
        if self.config.slippage_model == "none":
            return math.inf
        caps = []
        vol = ctx.volume if (ctx.volume and math.isfinite(ctx.volume) and ctx.volume > 0) else ctx.typical_volume
        if vol and math.isfinite(vol) and vol > 0 and not ctx.live:
            caps.append(self.config.participation_rate * vol * ctx.liq_mult)
        if ctx.live and ctx.book is None and ctx.typical_volume and math.isfinite(ctx.typical_volume):
            caps.append(self.config.participation_rate * ctx.typical_volume * ctx.liq_mult)
        return min(caps) if caps else math.inf

    # ------------------------------------------------------------------- driver
    def execute(self, orders: list[Order], ctx: MarketContext, portfolio: Portfolio,
                max_qty: MaxQtyFn | None = None) -> list[Fill]:
        """Try to fill ``orders`` (all for ``ctx.asset``) in priority order. Books fills into ``portfolio``."""
        path = ctx.path()
        cap = self.capacity(ctx)
        l2_depth = None
        if ctx.book is not None and self.config.use_l2_book:
            l2_depth = {Side.BUY: sum(s for _, s in ctx.book.asks), Side.SELL: sum(s for _, s in ctx.book.bids)}
        consumed = {Side.BUY: 0.0, Side.SELL: 0.0}
        fills: list[Fill] = []
        for order in orders:
            if not order.status.is_open:
                continue
            side_cap = cap if l2_depth is None else min(cap, l2_depth[order.side])
            fills.extend(self._execute_order(order, ctx, path, consumed, side_cap, portfolio, max_qty))
            if order.status.is_open and order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
                order.status = OrderStatus.CANCELLED
                order.reject_reason = order.reject_reason or "IOC/FOK remainder cancelled"
                order.updated_at = ctx.timestamp
        return fills

    def _stop_trigger(self, order: Order, path: list[float]) -> list[float] | None:
        """Return the remaining path starting at the trigger price, or None if not triggered."""
        stop = float(order.stop_price)  # type: ignore[arg-type]
        up = order.side is Side.BUY
        if (up and path[0] >= stop) or (not up and path[0] <= stop):
            return path  # gapped through: trigger at the open
        for i in range(len(path) - 1):
            nxt = path[i + 1]
            if (up and nxt >= stop) or (not up and nxt <= stop):
                return [stop] + path[i + 1:]
        return None

    def _execute_order(self, order: Order, ctx: MarketContext, path: list[float], consumed: dict[Side, float],
                       cap: float, portfolio: Portfolio, max_qty: MaxQtyFn | None) -> list[Fill]:
        sub = path
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not order.triggered:
            trig = self._stop_trigger(order, path)
            if trig is None:
                return []
            order.triggered = True
            order.updated_at = ctx.timestamp
            sub = trig
        at_entry = sub is path  # observed quotes / L2 only describe the start of the step

        if order.order_type in (OrderType.MARKET, OrderType.STOP):
            fill = self._take(order, ctx, sub[0], consumed, cap, portfolio, max_qty, limit=None, at_entry=at_entry)
            return [fill] if fill else []

        # LIMIT or triggered STOP_LIMIT
        limit = float(order.limit_price)  # type: ignore[arg-type]
        fills: list[Fill] = []
        bid, ask = self.touch(ctx, sub[0], at_entry)
        marketable = (order.side is Side.BUY and ask <= limit) or (order.side is Side.SELL and bid >= limit)
        if marketable:
            fill = self._take(order, ctx, sub[0], consumed, cap, portfolio, max_qty, limit=limit, at_entry=at_entry)
            if fill:
                fills.append(fill)
            if not order.status.is_open or order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
                return fills
        # Passive: the (rest of the) order sits at its limit and fills only if trades print
        # through it later in this step. In live mode the "path" is just the last trade.
        if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
            return fills
        rest = sub[1:] if marketable else sub
        if rest and self._through(order, rest, ctx):
            fill = self._passive(order, ctx, limit, consumed, cap, portfolio, max_qty)
            if fill:
                fills.append(fill)
        return fills

    def _through(self, order: Order, rest: list[float], ctx: MarketContext) -> bool:
        limit = float(order.limit_price)  # type: ignore[arg-type]
        tick = float(ctx.asset.tick_for_price(limit))
        if order.side is Side.BUY:
            extreme = min(rest)
            if extreme < limit - tick / 2:
                return True
            touched = extreme <= limit + tick / 2
        else:
            extreme = max(rest)
            if extreme > limit + tick / 2:
                return True
            touched = extreme >= limit - tick / 2
        return touched and self.rng.random() < self.config.limit_touch_fill_prob

    # ---------------------------------------------------------------- filling
    def _size(self, order: Order, ctx: MarketContext, avail: float, ref_price: float, portfolio: Portfolio,
              max_qty: MaxQtyFn | None) -> Decimal:
        step = ctx.asset.effective_qty_step
        qty = order.remaining
        if math.isfinite(avail):
            qty = min(qty, quantize_down(D(max(avail, 0.0)), step))
        if max_qty is not None:
            allowed = max_qty(order, ctx.asset, D(ref_price))
            if allowed is not None:
                qty = min(qty, quantize_down(max(allowed, ZERO), step))
        return quantize_down(qty, step)

    def _take(self, order: Order, ctx: MarketContext, ref: float, consumed: dict[Side, float], cap: float,
              portfolio: Portfolio, max_qty: MaxQtyFn | None, limit: float | None, at_entry: bool) -> Fill | None:
        side = order.side
        bid, ask = self.touch(ctx, ref, at_entry)
        book = self._book(ctx, bid, ask, at_entry)
        q0 = consumed[side]
        avail = cap - q0
        if limit is not None:
            avail = min(avail, book.max_qty_within(side, limit, q0) - q0)
        # Worst marginal price this fill could reach: used to size against buying power.
        est_price = ask if side is Side.BUY else bid
        if isinstance(book, SqrtImpactBook):
            depth = float(order.remaining) if not math.isfinite(avail) else min(avail, float(order.remaining))
            est_price = book.marginal_price(side, q0 + max(depth, 0.0))
        qty = self._size(order, ctx, avail, est_price, portfolio, max_qty)
        if order.time_in_force is TimeInForce.FOK and qty < order.remaining:
            order.reject_reason = "FOK: insufficient liquidity or buying power"
            return None
        if qty <= 0:
            return None
        q1 = q0 + float(qty)
        price = book.avg_price(side, q0, q1)
        cfg = self.config
        adverse = cfg.fixed_slippage_bps / 1e4
        noise = self.rng.normal(0.0, cfg.slippage_noise_bps / 1e4) if cfg.slippage_noise_bps > 0 else 0.0
        price *= (1 + adverse) if side is Side.BUY else (1 - adverse)
        price *= math.exp(noise)
        if limit is not None:
            price = min(price, limit) if side is Side.BUY else max(price, limit)
        tick = ctx.asset.tick_for_price(price)
        dprice = quantize_up(D(price), tick) if side is Side.BUY else quantize_down(D(price), tick)
        if limit is not None:
            dlimit = D(limit)
            dprice = min(dprice, dlimit) if side is Side.BUY else max(dprice, dlimit)
        if dprice <= 0:
            return None
        if max_qty is not None:
            # Re-check affordability at the final (tick-rounded, slipped) price; a smaller fill
            # would only get a better price, so keeping `dprice` stays conservative.
            allowed = max_qty(order, ctx.asset, dprice)
            if allowed is not None and qty > allowed:
                qty = quantize_down(max(allowed, ZERO), ctx.asset.effective_qty_step)
                if qty <= 0:
                    return None
                q1 = q0 + float(qty)
        consumed[side] = q1
        return self._book_fill(order, ctx, qty, dprice, Liquidity.TAKER, D(ref), portfolio)

    def _passive(self, order: Order, ctx: MarketContext, limit: float, consumed: dict[Side, float], cap: float,
                 portfolio: Portfolio, max_qty: MaxQtyFn | None) -> Fill | None:
        side = order.side
        avail = cap - consumed[side]
        qty = self._size(order, ctx, avail, limit, portfolio, max_qty)
        if qty <= 0:
            return None
        consumed[side] += float(qty)
        price = D(order.limit_price)  # type: ignore[arg-type]
        return self._book_fill(order, ctx, qty, price, Liquidity.MAKER, price, portfolio)

    def _book_fill(self, order: Order, ctx: MarketContext, qty: Decimal, price: Decimal, liquidity: Liquidity,
                   ref: Decimal, portfolio: Portfolio) -> Fill:
        schedule = self.fee_model.schedule_for(ctx.asset.asset_class)
        fee = schedule.compute(order.side, qty, price, liquidity, order.fees_paid, first_fill=order.filled_quantity == 0)
        slippage_cost = ((price - ref) * qty * order.side.sign).quantize(Decimal("0.00000001")).normalize()
        fill = Fill(
            id=self.fill_ids(), order_id=order.id, symbol=order.symbol, side=order.side, quantity=qty, price=price,
            fee=fee, liquidity=liquidity, timestamp=ctx.timestamp, slippage_cost=slippage_cost,
        )
        booked, _ = portfolio.apply_fill(fill)
        new_filled = order.filled_quantity + qty
        order.avg_fill_price = (order.avg_fill_price * order.filled_quantity + price * qty) / new_filled
        order.filled_quantity = new_filled
        order.fees_paid += fee
        order.status = OrderStatus.FILLED if order.remaining <= 0 else OrderStatus.PARTIALLY_FILLED
        order.updated_at = ctx.timestamp
        return booked
