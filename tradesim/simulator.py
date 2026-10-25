"""Trading simulator: broker semantics on top of market data + execution engine.

Two clocks share one implementation of orders, validation, accounting and the
agent-facing API (:class:`BaseSimulator`):

* :class:`TradingSimulator` - historical replay (backtesting / RL training).
* :class:`tradesim.live.LivePaperTrader` - wall-clock paper trading on live quotes.

Timing model (historical)
-------------------------
The simulator sits at the **close of bar k** (``now``). The agent sees every
bar up to and including bar k, places orders, and calls :meth:`step`. Orders
are then executed against bar k+1 (open first, then the intrabar path), the
portfolio is marked at the close of bar k+1, and the clock advances. An order
can therefore never trade at a price that was known when it was placed,
which is how live trading works too.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.base import DataProvider
from tradesim.data.store import MarketData, dt_to_ns, ns_to_dt
from tradesim.data.timeframes import Timeframe, resample_bars
from tradesim.execution.engine import ExecutionConfig, ExecutionEngine, MarketContext
from tradesim.execution.fees import FeeModel
from tradesim.indicators import add_indicators
from tradesim.market.conditions import Scenario
from tradesim.market.liquidity import SpreadModel
from tradesim.persistence import EventLog, TradeStore, read_json, write_json
from tradesim.portfolio.metrics import compute_metrics
from tradesim.portfolio.portfolio import MarginConfig, Portfolio
from tradesim.types import (
    ZERO, AssetClass, D, Fill, IdGenerator, Order, OrderStatus, OrderType, Quote, Side, TimeInForce,
    quantize_down,
)

if TYPE_CHECKING:  # pragma: no cover
    from tradesim.agents.base import Observation

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


@dataclass
class SimulatorConfig:
    """All knobs of a simulation run.

    Attributes:
        initial_cash: starting USD balance.
        timeframe: bar size the clock advances by (``1m``, ``5m``, ``15m``, ``1h``, ``4h``, ``1d``...).
        fees: fee preset name (see ``FeeModel.presets()``) or a ``FeeModel``.
        execution: slippage / spread / liquidity realism (see ``ExecutionConfig``).
        margin: leverage, shorting and financing (default: cash account, long only).
        seed: master seed; fixes episode sampling, execution noise and scenarios.
        warmup_bars: history loaded before ``start`` (indicators, volatility estimates).
        vol_lookback: bars used for the volatility / typical-volume estimates.
        scenario: market-condition preset name or ``Scenario`` applied to the episode window.
        max_position_pct: optional pre-trade cap on a single position's weight (0-100).
        risk_free_annual: used by Sharpe / Sortino.
        log_dir: if set, writes ``events.jsonl`` and ``trades.sqlite`` there.
    """

    initial_cash: Decimal | float | str = 10_000
    timeframe: Timeframe | str = Timeframe.D1
    fees: str | FeeModel = "alpaca"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    seed: int = 42
    warmup_bars: int = 60
    vol_lookback: int = 20
    scenario: str | Scenario | None = None
    max_position_pct: float | None = None
    risk_free_annual: float = 0.0
    log_dir: str | None = None

    def fee_model(self) -> FeeModel:
        return self.fees if isinstance(self.fees, FeeModel) else FeeModel.preset(self.fees)

    def to_dict(self) -> dict[str, Any]:
        ex = asdict(self.execution)
        return {
            "initial_cash": str(self.initial_cash),
            "timeframe": Timeframe.parse(self.timeframe).value,
            "fees": self.fees if isinstance(self.fees, str) else self.fees.name,
            "execution": ex,
            "margin": {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self.margin).items()},
            "seed": self.seed,
            "warmup_bars": self.warmup_bars,
            "vol_lookback": self.vol_lookback,
            "scenario": self.scenario if isinstance(self.scenario, str) or self.scenario is None else self.scenario.name,
            "max_position_pct": self.max_position_pct,
            "risk_free_annual": self.risk_free_annual,
            "log_dir": self.log_dir,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimulatorConfig":
        ex = dict(d["execution"])
        ex["spread"] = SpreadModel(**ex["spread"])
        margin = MarginConfig(**{k: (v if isinstance(v, bool) else D(v)) for k, v in d["margin"].items()})
        return cls(
            initial_cash=D(d["initial_cash"]), timeframe=Timeframe.parse(d["timeframe"]), fees=d["fees"],
            execution=ExecutionConfig(**ex), margin=margin, seed=d["seed"], warmup_bars=d["warmup_bars"],
            vol_lookback=d["vol_lookback"], scenario=d["scenario"], max_position_pct=d["max_position_pct"],
            risk_free_annual=d["risk_free_annual"], log_dir=d["log_dir"],
        )


class OrderRejected(Exception):
    pass


def _session_date(asset: Asset, ts: datetime) -> Any:
    """Trading date used for DAY-order expiry (New York date for US sessions, UTC otherwise)."""
    if asset.effective_session == "XNYS":
        return ts.astimezone(NY).date()
    return ts.date()


class BaseSimulator:
    """Broker core shared by historical and live simulators.

    Subclasses provide the clock (:attr:`now`), quotes (:meth:`_quote`) and
    history access (:meth:`get_historical_data`).
    """

    def __init__(self, universe: AssetUniverse | None, provider: DataProvider, config: SimulatorConfig | None = None):
        self.universe = universe or AssetUniverse.default()
        self.provider = provider
        self.config = config or SimulatorConfig()
        self.timeframe = Timeframe.parse(self.config.timeframe)
        self.fee_model = self.config.fee_model()
        self._init_run_state()

    # ------------------------------------------------------------------ state
    def _init_run_state(self) -> None:
        cfg = self.config
        seeds = np.random.SeedSequence(cfg.seed).spawn(3)
        self.rng_exec = np.random.default_rng(seeds[0])
        self.rng_episode = np.random.default_rng(seeds[1])
        self.order_ids = IdGenerator("O")
        self.fill_ids = IdGenerator("F")
        self.engine = ExecutionEngine(cfg.execution, self.fee_model, self.rng_exec, self.fill_ids)
        self.portfolio = Portfolio(cfg.initial_cash, cfg.margin)
        self.orders: dict[str, Order] = {}
        self.open_order_ids: list[str] = []
        self.run_id = uuid.uuid4().hex[:12]
        self.events = EventLog(Path(cfg.log_dir) / "events.jsonl" if cfg.log_dir else None)
        self.store = TradeStore(Path(cfg.log_dir) / "trades.sqlite", self.run_id) if cfg.log_dir else None
        self._bp_starved: set[str] = set()
        self.bankrupt = False

    # -------------------------------------------------------- abstract hooks
    @property
    def now(self) -> datetime:  # pragma: no cover - abstract
        raise NotImplementedError

    def _quote(self, asset: Asset) -> Quote:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_historical_data(self, symbol: str, timeframe: Timeframe | str | None = None, limit: int = 100,
                            indicators: bool = False) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    # ============================================================ agent API
    def get_available_assets(self, asset_class: str | None = None, group: str | None = None, query: str | None = None,
                             max_liquidity_tier: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """List tradable assets, optionally filtered (``asset_class`` in equity/etf/crypto/forex)."""
        assets = self.universe.filter(asset_class=asset_class, group=group, query=query,
                                      max_liquidity_tier=max_liquidity_tier)
        rows = [{"symbol": a.symbol, "name": a.name, "asset_class": a.asset_class.value, "groups": list(a.groups),
                 "sector": a.sector, "liquidity_tier": a.liquidity_tier, "fractionable": a.fractionable,
                 "shortable": a.shortable} for a in assets]
        return rows[:limit] if limit else rows

    def get_asset(self, symbol: str) -> dict[str, Any]:
        return self.universe.get(symbol).to_dict()

    def get_price(self, symbol: str) -> dict[str, Any]:
        """Latest quote: ``bid``, ``ask``, ``mid``, ``last``, ``spread_bps``, ``timestamp`` (+ ``modelled`` flag)."""
        return self._quote(self.universe.get(symbol)).to_dict()

    def get_portfolio(self) -> dict[str, Any]:
        """Cash, equity, PnL, exposure, drawdown and open positions."""
        out = self.portfolio.summary()
        out["timestamp"] = self.now.isoformat()
        out["open_orders"] = len(self.open_order_ids)
        return out

    def get_positions(self) -> list[dict[str, Any]]:
        eq = self.portfolio.equity
        return [p.to_dict(eq if eq > 0 else None) for p in self.portfolio.open_positions()]

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self.orders[order_id].to_dict()

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [self.orders[i].to_dict() for i in self.open_order_ids if symbol is None or self.orders[i].symbol == symbol]

    def get_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        return [o.to_dict() for o in list(self.orders.values())[-limit:]]

    def get_fills(self, limit: int = 50) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.portfolio.fills[-limit:]]

    def get_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        """Closed round trips (with net PnL)."""
        return [t.to_dict() for t in self.portfolio.closed_trades[-limit:]]

    def get_trade_history(self, kind: str = "fills", symbol: str | None = None,
                          as_dataframe: bool = False) -> list[dict[str, Any]] | pd.DataFrame:
        """Complete history of the episode.

        Args:
            kind: ``"fills"`` = every execution (buys and sells, partial fills included, with
                price, fee, maker/taker, slippage cost and realised PnL); ``"closed"`` = closed
                round trips (entry/exit price, net PnL after fees) used for win rate etc.
            symbol: only this symbol.
            as_dataframe: return a ``pandas.DataFrame`` (numeric columns as floats) instead of dicts.
        """
        if kind == "fills":
            rows = [f.to_dict() for f in self.portfolio.fills if symbol is None or f.symbol == symbol]
            time_cols, num_cols = ["timestamp"], ["quantity", "price", "notional", "fee", "slippage_cost", "realized_pnl"]
        elif kind == "closed":
            rows = [t.to_dict() for t in self.portfolio.closed_trades if symbol is None or t.symbol == symbol]
            time_cols, num_cols = ["opened_at", "closed_at"], ["quantity", "entry_price", "exit_price", "pnl", "fees"]
        else:
            raise ValueError("kind must be 'fills' or 'closed'")
        if not as_dataframe:
            return rows
        df = pd.DataFrame(rows)
        for c in num_cols:
            if c in df:
                df[c] = df[c].astype(float)
        for c in time_cols:
            if c in df:
                df[c] = pd.to_datetime(df[c], utc=True)
        return df

    # ======================================================== agent loop API
    def observe(self) -> "Observation":
        """Snapshot for ``Agent.act``: prices, cash, equity, positions (see :class:`Observation`)."""
        from tradesim.agents.base import Observation

        symbols = list(getattr(self, "symbols", []) or [p.symbol for p in self.portfolio.open_positions()])
        prices = {}
        for s in symbols:
            try:
                prices[s] = self._quote(self.universe.get(s)).last
            except Exception:  # noqa: BLE001 - no price yet (before first bar / feed hiccup)
                prices[s] = float("nan")
        return Observation(
            timestamp=self.now, step=0, steps_remaining=0, symbols=symbols, prices=prices,
            cash=float(self.portfolio.cash), equity=float(self.portfolio.equity),
            positions={p.symbol: p.quantity for p in self.portfolio.open_positions()},
            open_orders=len(self.open_order_ids), _sim=self,
        )

    def recent_closes(self, symbol: str, limit: int = 100) -> np.ndarray:
        """Last ``limit`` completed closes of ``symbol`` as a numpy array."""
        return self.get_historical_data(symbol, limit=limit)["close"].to_numpy(dtype=float)

    def detect_patterns(self, symbol: str, limit: int = 300, timeframe: Timeframe | str | None = None):
        """Technical patterns on the last ``limit`` completed bars (see :mod:`tradesim.patterns`)."""
        from tradesim.patterns import detect_patterns

        return detect_patterns(self.get_historical_data(symbol, timeframe, limit), symbol)

    def apply_action(self, action: Any) -> list[dict[str, Any]]:
        """Execute an agent action (see :mod:`tradesim.agents.base` for the accepted forms).

        Returns the resulting order dicts; rejected orders carry ``status == "rejected"``
        and a ``reject_reason`` instead of raising, so agents can learn from them.
        """
        from tradesim.agents.base import CancelOrders, ClosePosition, OrderIntent, TargetWeights

        if action is None or (isinstance(action, str) and action.lower() == "hold"):
            return []
        if isinstance(action, (list, tuple)):
            out: list[dict[str, Any]] = []
            for a in action:
                out.extend(self.apply_action(a))
            return out
        if isinstance(action, OrderIntent):
            return [self.place_order(**action.kwargs())]
        if isinstance(action, TargetWeights):
            return self.target_weights(action.weights)
        if isinstance(action, ClosePosition):
            res = self.close_position(action.symbol)
            return [res] if res else []
        if isinstance(action, CancelOrders):
            self.cancel_all_orders(action.symbol)
            return []
        if isinstance(action, dict):
            if "weights" in action:
                return self.target_weights(action["weights"])
            if "close" in action:
                res = self.close_position(action["close"])
                return [res] if res else []
            return [self.place_order(**action)]
        raise TypeError(f"Unsupported action type {type(action).__name__}: {action!r}")

    # ============================================================ reporting
    def print_portfolio(self) -> None:
        """Print cash, equity and open positions (fractional quantity, average price, unrealised PnL)."""
        from tradesim.reporting import format_portfolio

        print(format_portfolio(self.get_portfolio()))

    def print_summary(self, title: str | None = None) -> None:
        """Print a readable summary of the episode's performance metrics."""
        from tradesim.reporting import format_summary

        print(format_summary(self.get_metrics(), title=title or f"Episode summary ({self.now:%Y-%m-%d %H:%M} UTC)"))

    def plot_results(self, symbol: str | None = None, path: str | Path | None = None, show: bool | None = None,
                     benchmark: bool = True):
        """Plot equity curve, drawdown and (optionally) buy/sell points on ``symbol``'s price.

        Saves to ``path`` if given; shows the window when ``show`` is true (default: only when
        no path is given and the matplotlib backend is interactive). Returns the figure.
        """
        from tradesim.reporting import plot_results

        return plot_results(self, symbol=symbol, path=path, show=show, benchmark=benchmark)

    def place_order(
        self,
        symbol: str,
        side: str | Side,
        quantity: float | str | Decimal | None = None,
        order_type: str | OrderType = OrderType.MARKET,
        limit_price: float | str | Decimal | None = None,
        stop_price: float | str | Decimal | None = None,
        time_in_force: str | TimeInForce = TimeInForce.GTC,
        notional: float | str | Decimal | None = None,
        percent_of_equity: float | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Submit an order. Size with exactly one of ``quantity`` (units, fractional OK),
        ``notional`` (USD) or ``percent_of_equity`` (0-100).

        Returns the order as a dict; ``status == "rejected"`` with ``reject_reason`` if invalid.
        The order executes on the next :meth:`step` (historical) or :meth:`sync` (live).
        """
        order = Order(symbol=symbol, side=Side.BUY, quantity=ZERO, id=self.order_ids(), created_at=self.now, tag=tag)
        try:
            self._fill_in_order(order, side, quantity, order_type, limit_price, stop_price, time_in_force,
                                notional, percent_of_equity)
        except (OrderRejected, ValueError, KeyError) as exc:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(exc).strip("'\"")
            order.updated_at = self.now
            self.orders[order.id] = order
            self.events.emit("order_rejected", self.now, order=order.to_dict())
            self._persist_order(order)
            return order.to_dict()
        self.orders[order.id] = order
        self.open_order_ids.append(order.id)
        self.events.emit("order_submitted", self.now, order=order.to_dict())
        self._persist_order(order)
        return order.to_dict()

    def buy(self, symbol: str, quantity: float | str | Decimal | None = None, **kw: Any) -> dict[str, Any]:
        return self.place_order(symbol, Side.BUY, quantity, **kw)

    def sell(self, symbol: str, quantity: float | str | Decimal | None = None, **kw: Any) -> dict[str, Any]:
        return self.place_order(symbol, Side.SELL, quantity, **kw)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"Unknown order {order_id}")
        if order.status.is_open:
            self._close_order(order, OrderStatus.CANCELLED, "cancelled by agent")
        return order.to_dict()

    def cancel_all_orders(self, symbol: str | None = None) -> int:
        ids = [i for i in self.open_order_ids if symbol is None or self.orders[i].symbol == symbol]
        for i in ids:
            self._close_order(self.orders[i], OrderStatus.CANCELLED, "cancelled by agent")
        return len(ids)

    def close_position(self, symbol: str, tag: str | None = "close_position") -> dict[str, Any] | None:
        """Cancel the symbol's open orders and submit a market order flattening the position."""
        self.cancel_all_orders(symbol)
        qty = self.portfolio.quantity(symbol)
        if qty == 0:
            return None
        side = Side.SELL if qty > 0 else Side.BUY
        return self.place_order(symbol, side, abs(qty), tag=tag)

    def close_all_positions(self) -> list[dict[str, Any]]:
        out = []
        for pos in list(self.portfolio.open_positions()):
            res = self.close_position(pos.symbol)
            if res:
                out.append(res)
        return out

    def target_weights(self, weights: dict[str, float], min_trade_notional: float = 1.0,
                       tag: str | None = "rebalance") -> list[dict[str, Any]]:
        """Rebalance to target portfolio weights (fractions of equity; negative = short).

        Sells are submitted before buys so freed cash funds the purchases on the same bar.
        """
        eq = float(self.portfolio.equity)
        orders: list[tuple[str, Side, Decimal]] = []
        symbols = set(weights) | {p.symbol for p in self.portfolio.open_positions()}
        for sym in sorted(symbols):
            asset = self.universe.get(sym)
            q = self._quote(asset)
            price = q.mid
            target_qty = D(weights.get(sym, 0.0) * eq / price) if price > 0 else ZERO
            delta = target_qty - self.portfolio.quantity(sym)
            if abs(float(delta) * price) < min_trade_notional:
                continue
            qty = quantize_down(abs(delta), asset.effective_qty_step)
            if qty > 0:
                orders.append((sym, Side.BUY if delta > 0 else Side.SELL, qty))
        self.cancel_all_orders()
        results = [self.place_order(s, side, q, tag=tag) for s, side, q in orders if side is Side.SELL]
        if any(side is Side.SELL for _, side, _ in orders):
            self._settle()  # live: let the sells execute so their proceeds can fund the buys
        results += [self.place_order(s, side, q, tag=tag) for s, side, q in orders if side is Side.BUY]
        return results

    def _settle(self) -> None:
        """Execute what can execute right now. Backtests: nothing (orders fill on the next bar);
        live accounts override it to sync with the market."""

    def get_metrics(self) -> dict[str, Any]:
        """Sharpe, Sortino, max drawdown, CAGR, win rate, profit factor, fees..."""
        curve = self.portfolio.equity_curve
        if not curve:
            return {}
        idx = pd.DatetimeIndex([p.timestamp for p in curve])
        equity = pd.Series([float(p.equity) for p in curve], index=idx)
        exposure = pd.Series([float(p.gross_exposure) for p in curve], index=idx)
        held = {f.symbol for f in self.portfolio.fills} or set(getattr(self, "symbols", []) or [])
        all_crypto = bool(held) and all(self.universe.get(s).is_24_7 for s in held if s in self.universe)
        ppy = self.timeframe.bars_per_year(all_crypto)
        turnover = sum(float(f.notional) for f in self.portfolio.fills)
        m = compute_metrics(equity, self.portfolio.closed_trades, ppy, self.config.risk_free_annual,
                            float(self.portfolio.total_fees), turnover, exposure)
        m["total_slippage_cost"] = float(self.portfolio.total_slippage)
        m["total_financing"] = float(self.portfolio.total_financing)
        m["turnover_notional"] = turnover
        m["num_fills"] = len(self.portfolio.fills)
        m["open_positions"] = len(self.portfolio.open_positions())
        m["start"] = curve[0].timestamp.isoformat()
        m["end"] = curve[-1].timestamp.isoformat()
        return m

    # =============================================================== orders
    def _fill_in_order(self, order: Order, side, quantity, order_type, limit_price, stop_price, tif,
                       notional, percent_of_equity) -> None:
        if self.bankrupt:
            raise OrderRejected("account is bankrupt")
        asset = self.universe.get(order.symbol)
        if not asset.tradable:
            raise OrderRejected(f"{asset.symbol} is not tradable")
        order.side = Side(side.lower()) if isinstance(side, str) else side
        order.order_type = OrderType(order_type.lower()) if isinstance(order_type, str) else order_type
        order.time_in_force = TimeInForce(tif.lower()) if isinstance(tif, str) else tif
        if sum(x is not None for x in (quantity, notional, percent_of_equity)) != 1:
            raise OrderRejected("specify exactly one of quantity, notional, percent_of_equity")

        quote = self._quote(asset)
        # Prices: must be positive; snapped to the tick grid (brokers reject sub-tick prices).
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if limit_price is None:
                raise OrderRejected(f"{order.order_type.value} order requires limit_price")
            order.limit_price = self._snap(asset, D(limit_price))
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if stop_price is None:
                raise OrderRejected(f"{order.order_type.value} order requires stop_price")
            order.stop_price = self._snap(asset, D(stop_price))
        for p in (order.limit_price, order.stop_price):
            if p is not None and p <= 0:
                raise OrderRejected("prices must be positive")

        ref = order.limit_price or order.stop_price or D(quote.ask if order.side is Side.BUY else quote.bid)
        if ref <= 0:
            raise OrderRejected("no valid reference price")
        if quantity is not None:
            qty = D(quantity)
        elif notional is not None:
            qty = D(notional) / ref
        else:
            pct = float(percent_of_equity)
            if not 0 < pct <= 100 * float(max(self.config.margin.max_leverage, Decimal(1))):
                raise OrderRejected("percent_of_equity must be in (0, 100 x max_leverage]")
            qty = self.portfolio.equity * D(pct) / 100 / ref
        if qty <= 0:
            raise OrderRejected("quantity must be positive")
        step = asset.effective_qty_step
        qty = quantize_down(qty, step)
        if not asset.fractionable and qty != qty.to_integral_value():
            qty = qty.to_integral_value(rounding="ROUND_DOWN")
        if qty < asset.effective_min_qty or qty <= 0:
            raise OrderRejected(f"quantity below minimum ({asset.effective_min_qty}) after rounding to lot step {step}")
        order.quantity = qty

        # Position / shorting checks against the projected position (incl. other open orders).
        held = self.portfolio.quantity(order.symbol)
        pending = sum((o.remaining * o.side.sign for o in self._open_orders(order.symbol)), ZERO)
        projected = held + pending
        new = projected + qty * order.side.sign
        closes_position = new == 0
        can_short = self.config.margin.allow_short and asset.shortable
        if new < 0 and not can_short:
            avail = max(projected, ZERO)
            raise OrderRejected(f"insufficient position to sell {qty} {asset.symbol} (available {avail}); "
                                f"short selling is {'not allowed for this asset' if self.config.margin.allow_short else 'disabled'}")
        notional_value = qty * ref
        if not closes_position and notional_value < asset.effective_min_notional:
            raise OrderRejected(f"order notional {notional_value:.2f} below minimum {asset.effective_min_notional}")

        # Buying power for the exposure-increasing part (estimated with taker fee + spread buffer).
        increase = max(abs(new) - abs(projected), ZERO)
        if increase > 0:
            sched = self.fee_model.schedule_for(asset.asset_class)
            buffer = D(1) + sched.taker_pct + D(self.config.execution.fixed_slippage_bps / 1e4) + D("0.002")
            need = increase * ref * buffer + sched.fixed_per_order + sched.min_per_order + increase * sched.per_share
            bp = self.portfolio.buying_power(self._reserved_notional())
            if need > bp:
                raise OrderRejected(f"insufficient buying power: need ~{need:.2f}, available {bp:.2f}")
            if self.config.max_position_pct is not None:
                eq = self.portfolio.equity
                if eq > 0 and abs(new) * ref / eq * 100 > D(self.config.max_position_pct):
                    raise OrderRejected(f"position would exceed max_position_pct={self.config.max_position_pct}%")

    @staticmethod
    def _snap(asset: Asset, price: Decimal) -> Decimal:
        tick = asset.tick_for_price(float(price))
        return (price / tick).to_integral_value() * tick

    def _open_orders(self, symbol: str | None = None) -> list[Order]:
        return [self.orders[i] for i in self.open_order_ids if symbol is None or self.orders[i].symbol == symbol]

    def _reserved_notional(self) -> Decimal:
        """Notional committed to open orders that increase exposure (blocks double-spending cash)."""
        reserved = ZERO
        by_symbol: dict[str, list[Order]] = {}
        for o in self._open_orders():
            by_symbol.setdefault(o.symbol, []).append(o)
        for sym, orders in by_symbol.items():
            projected = self.portfolio.quantity(sym)
            for o in orders:
                new = projected + o.remaining * o.side.sign
                inc = max(abs(new) - abs(projected), ZERO)
                if inc > 0:
                    price = o.limit_price or o.stop_price
                    if price is None:
                        q = self._quote(self.universe.get(sym))
                        price = D(q.ask if o.side is Side.BUY else q.bid)
                    reserved += inc * price
                projected = new
        return reserved

    def _max_fill_qty(self, order: Order, asset: Asset, price: Decimal) -> Decimal | None:
        """Called by the engine at fill time: what the account can actually afford / deliver now."""
        held = self.portfolio.quantity(order.symbol)
        can_short = self.config.margin.allow_short and asset.shortable
        if order.side is Side.SELL and not can_short:
            allowed = max(held, ZERO)
            if allowed < order.remaining:
                self._bp_starved.add(order.id)
            return allowed
        reducing = held < 0 if order.side is Side.BUY else held > 0
        reduce_qty = abs(held) if reducing else ZERO
        sched = self.fee_model.schedule_for(asset.asset_class)
        rate = sched.taker_pct + D(self.config.execution.fixed_slippage_bps / 1e4)
        bp = self.portfolio.buying_power() - sched.fixed_per_order - sched.min_per_order
        unit_cost = price * (1 + rate) + sched.per_share
        inc_max = max(bp, ZERO) / unit_cost if unit_cost > 0 else ZERO
        allowed = reduce_qty + inc_max
        if allowed < order.remaining:
            self._bp_starved.add(order.id)
        return allowed

    def _close_order(self, order: Order, status: OrderStatus, reason: str | None = None) -> None:
        order.status = status
        order.reject_reason = order.reject_reason or reason
        order.updated_at = self.now
        if order.id in self.open_order_ids:
            self.open_order_ids.remove(order.id)
        self.events.emit(f"order_{status.value}", self.now, order=order.to_dict())
        self._persist_order(order)

    def _execute_symbol(self, asset: Asset, ctx: MarketContext) -> list[Fill]:
        orders = self._open_orders(asset.symbol)
        if not orders:
            return []
        self._bp_starved.clear()
        n_closed = len(self.portfolio.closed_trades)
        fills = self.engine.execute(orders, ctx, self.portfolio, self._max_fill_qty)
        for f in fills:
            self.events.emit("fill", self.now, fill=f.to_dict())
            if self.store:
                self.store.add_fill(f.to_dict())
        if self.store:
            for t in self.portfolio.closed_trades[n_closed:]:
                self.store.add_closed_trade(t.to_dict())
        for o in orders:
            if not o.status.is_open:
                if o.id in self.open_order_ids:
                    self.open_order_ids.remove(o.id)
                self.events.emit(f"order_{o.status.value}", self.now, order=o.to_dict())
                self._persist_order(o)
            elif o.id in self._bp_starved and (o.order_type is OrderType.MARKET or
                                               (o.order_type is OrderType.STOP and o.triggered)):
                # A market order the account can no longer afford (price moved) is cancelled, like a broker would.
                self._close_order(o, OrderStatus.CANCELLED, "insufficient buying power / position at execution")
        return fills

    def _check_margin(self) -> None:
        pf = self.portfolio
        if pf.equity <= 0 and pf.gross_exposure > 0:
            self.bankrupt = True
        if pf.is_margin_call():
            self.events.emit("margin_call", self.now, equity=str(pf.equity), gross_exposure=str(pf.gross_exposure))
            log.warning("Margin call at %s: equity %s, gross exposure %s -> liquidating", self.now, pf.equity,
                        pf.gross_exposure)
            self.cancel_all_orders()
            for pos in pf.open_positions():
                side = Side.SELL if pos.quantity > 0 else Side.BUY
                order = Order(symbol=pos.symbol, side=side, quantity=abs(pos.quantity), id=self.order_ids(),
                              created_at=self.now, tag="liquidation")
                self.orders[order.id] = order
                self.open_order_ids.append(order.id)
                self.events.emit("order_submitted", self.now, order=order.to_dict())

    def _persist_order(self, order: Order) -> None:
        if self.store:
            self.store.upsert_order(order.to_dict())

    def _record_equity(self, ts: datetime) -> None:
        p = self.portfolio.record_equity(ts)
        if self.store:
            self.store.add_equity(p.timestamp, p.equity, p.cash, p.gross_exposure, p.net_exposure)
            self.store.commit()

    def close(self) -> None:
        self.events.close()
        if self.store:
            self.store.close()


class TradingSimulator(BaseSimulator):
    """Historical / backtesting simulator.

    Example::

        sim = TradingSimulator(provider=SyntheticProvider(seed=1),
                               config=SimulatorConfig(initial_cash=10_000, timeframe="1d"),
                               symbols=["AAPL", "BTC-USD"], start="2023-01-01", end="2024-01-01")
        sim.reset()
        sim.place_order("BTC-USD", "buy", 0.003)
        sim.step()
        print(sim.get_portfolio())

    ``symbols`` / ``start`` / ``end`` given here are the defaults for :meth:`reset`, so the
    simulator can be handed straight to :func:`tradesim.practice.evaluate_agent`.
    """

    def __init__(self, universe: AssetUniverse | None = None, provider: DataProvider | None = None,
                 config: SimulatorConfig | None = None, symbols: Iterable[str] | None = None,
                 start: datetime | str | None = None, end: datetime | str | None = None):
        if provider is None:
            from tradesim.data.synthetic import SyntheticProvider

            provider = SyntheticProvider(seed=(config or SimulatorConfig()).seed)
            log.warning("No provider given: using SyntheticProvider (synthetic prices, not real market data).")
        super().__init__(universe, provider, config)
        self.data: MarketData | None = None
        self.symbols: list[str] = list(dict.fromkeys(symbols)) if symbols else []
        self.default_start: datetime | str = start or "2023-01-01"
        self.default_end: datetime | str = end or "2024-01-01"
        self._calendar = np.array([], dtype=np.int64)
        self._k = 0
        self._start_k = 0
        self._end_k = 0
        self._aux_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # --------------------------------------------------------------- episode
    def reset(self, symbols: Iterable[str] | None = None, start: datetime | str | None = None,
              end: datetime | str | None = None, seed: int | None = None, start_offset: int | None = None,
              episode_length: int | None = None, random_start: bool = False) -> dict[str, Any]:
        """Start a new episode: fresh portfolio, data loaded for ``symbols`` over ``[start, end)``.

        Args:
            symbols / start / end: default to the previous episode's (or the constructor's).
            seed: master seed for this episode (execution noise, scenarios, random start).
            start_offset: start this many bars into the window (earlier bars are visible history).
            episode_length: stop after this many bars (default: run to the end of the window).
            random_start: draw the start bar at random (seeded) so that consecutive episodes
                cover different market periods; requires ``episode_length``.

        Data is reused when the window, scenario (and seed, if a scenario is set) are unchanged,
        so repeated resets for training / evaluation are cheap.
        """
        if seed is not None:
            self.config.seed = seed
        if self.data is not None:
            self.close()
        self._init_run_state()
        if symbols is not None:
            self.symbols = list(dict.fromkeys(symbols))
        if not self.symbols:
            raise ValueError("reset() needs at least one symbol")
        self.default_start = start or self.default_start
        self.default_end = end or self.default_end
        start = pd.Timestamp(self.default_start)
        end = pd.Timestamp(self.default_end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        scenario = self.config.scenario
        if isinstance(scenario, str):
            scenario = Scenario.preset(scenario)
        reuse = (self.data is not None and self.data.start == start and self.data.end == end
                 and getattr(self.data.scenario, "name", None) == getattr(scenario, "name", None)
                 and (scenario is None or self.data.seed == self.config.seed))
        if not reuse:
            self.data = MarketData(self.provider, self.universe, self.timeframe, start, end,
                                   self.config.warmup_bars, self.config.vol_lookback, scenario, self.config.seed)
        loaded = self.data.load(self.symbols)
        missing = [s for s in self.symbols if s not in self.data.series]
        if missing:
            log.warning("No data for %d symbols (dropped): %s", len(missing), missing[:10])
            self.symbols = [s for s in self.symbols if s in self.data.series]
        if not self.symbols:
            raise RuntimeError("No market data could be loaded for any requested symbol")
        self._calendar = self.data.calendar(self.symbols)
        if len(self._calendar) < 2:
            raise RuntimeError("Not enough bars in the requested window")
        last = len(self._calendar) - 1
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        if start_offset is None:
            start_offset = 0
            if random_start:
                if episode_length is None:
                    raise ValueError("random_start=True needs episode_length")
                start_offset = int(self.rng_episode.integers(0, max(last - episode_length, 0) + 1))
        if not 0 <= start_offset < last:
            raise ValueError(f"start_offset {start_offset} outside [0, {last - 1}]")
        self._k = self._start_k = start_offset
        self._end_k = last if episode_length is None else min(start_offset + episode_length, last)
        self._mark_all(self._calendar[self._k])
        self._record_equity(self.now)
        self.events.emit("reset", self.now, symbols=len(self.symbols), loaded=len(loaded),
                         bars=int(len(self._calendar)), config=self.config.to_dict(), run_id=self.run_id)
        return self.get_portfolio()

    @property
    def now(self) -> datetime:
        """Decision time: the close of the current bar."""
        if not len(self._calendar):
            return datetime(1970, 1, 1, tzinfo=ZoneInfo("UTC"))
        return ns_to_dt(int(self._calendar[self._k])) + timedelta(seconds=self.timeframe.seconds)

    @property
    def current_bar_time(self) -> datetime:
        return ns_to_dt(int(self._calendar[self._k]))

    @property
    def done(self) -> bool:
        return self.bankrupt or self._k >= self._end_k

    @property
    def num_bars(self) -> int:
        return int(len(self._calendar))

    @property
    def step_index(self) -> int:
        return self._k

    @property
    def steps_remaining(self) -> int:
        return max(self._end_k - self._k, 0)

    @property
    def episode_step(self) -> int:
        """Bars elapsed since the start of the current episode."""
        return self._k - self._start_k

    def observe(self) -> "Observation":
        """Snapshot for ``Agent.act`` (cheap even with 1000+ symbols: reads the price arrays)."""
        from tradesim.agents.base import Observation

        assert self.data is not None, "call reset() first"
        ts_ns = int(self._calendar[self._k])
        prices: dict[str, float] = {}
        for sym in self.symbols:
            series = self.data.series[sym]
            i = series.index_le(ts_ns)
            prices[sym] = float(series.close[i]) if i is not None else float("nan")
        return Observation(
            timestamp=self.now, step=self.episode_step, steps_remaining=self.steps_remaining,
            symbols=list(self.symbols), prices=prices, cash=float(self.portfolio.cash),
            equity=float(self.portfolio.equity),
            positions={p.symbol: p.quantity for p in self.portfolio.open_positions()},
            open_orders=len(self.open_order_ids), _sim=self,
        )

    def recent_closes(self, symbol: str, limit: int = 100) -> np.ndarray:
        """Last ``limit`` completed closes as a numpy array, read straight from the price arrays."""
        assert self.data is not None, "call reset() first"
        series = self.data.ensure(symbol)
        i = series.index_le(int(self._calendar[self._k]))
        if i is None:
            return np.array([], dtype=float)
        return series.close[max(0, i + 1 - limit): i + 1]

    def price_series(self, symbol: str, episode_only: bool = True) -> pd.Series:
        """Close prices of ``symbol`` up to now (from the episode start if ``episode_only``)."""
        assert self.data is not None
        lo = int(self._calendar[self._start_k]) if episode_only else None
        df = self.data.frame_upto(symbol, int(self._calendar[self._k]))
        if lo is not None:
            df = df[df.index.asi8 >= lo]
        return df["close"]

    def step(self, n: int = 1) -> list[dict[str, Any]]:
        """Advance ``n`` bars, executing open orders on each. Returns the fills that happened."""
        fills: list[Fill] = []
        for _ in range(n):
            if self.done:
                break
            prev_ns = int(self._calendar[self._k])
            self._k += 1
            ts_ns = int(self._calendar[self._k])
            fills.extend(self._process_bar(ts_ns))
            self.portfolio.accrue_financing((ts_ns - prev_ns) / 1e9)
            self._mark_all(ts_ns)
            self._record_equity(self.now)
            self._check_margin()
        return [f.to_dict() for f in fills]

    def run(self, agent: Callable[["TradingSimulator"], None], max_steps: int | None = None) -> dict[str, Any]:
        """Call ``agent(sim)`` at every bar, then step, until the episode ends. Returns metrics."""
        steps = 0
        while not self.done and (max_steps is None or steps < max_steps):
            agent(self)
            self.step()
            steps += 1
        return self.get_metrics()

    def _process_bar(self, ts_ns: int) -> list[Fill]:
        assert self.data is not None
        bar_time = ns_to_dt(ts_ns)
        fills: list[Fill] = []
        symbols = list(dict.fromkeys(self.orders[i].symbol for i in self.open_order_ids))
        for sym in symbols:
            series = self.data.series.get(sym)
            if series is None:
                continue
            i = series.index_exact(ts_ns)
            if i is None:
                continue  # market closed for this symbol at this time: orders keep waiting
            asset = series.asset
            for o in self._open_orders(sym):
                if o.time_in_force is TimeInForce.DAY and o.created_at is not None \
                        and _session_date(asset, bar_time) > _session_date(asset, o.created_at - timedelta(seconds=1)):
                    self._close_order(o, OrderStatus.EXPIRED, "DAY order expired")
            spread_bps = None
            if series.spread_bps is not None and math.isfinite(series.spread_bps[i]):
                spread_bps = float(series.spread_bps[i])
            ctx = MarketContext(
                asset=asset, timestamp=bar_time, open=float(series.open[i]), high=float(series.high[i]),
                low=float(series.low[i]), close=float(series.close[i]), volume=float(series.volume[i]),
                sigma=float(series.sigma[i]), sigma_typical=series.sigma_typical,
                typical_volume=float(series.typical_volume[i]), spread_mult=float(series.spread_mult[i]),
                liq_mult=float(series.liq_mult[i]), spread_bps_observed=spread_bps,
            )
            fills.extend(self._execute_symbol(asset, ctx))
        return fills

    def _mark_all(self, ts_ns: int) -> None:
        assert self.data is not None
        for pos in self.portfolio.open_positions():
            series = self.data.series.get(pos.symbol)
            if series is None:
                continue
            i = series.index_le(ts_ns)
            if i is not None:
                self.portfolio.mark(pos.symbol, series.close[i])

    # ------------------------------------------------------------ market view
    def _quote(self, asset: Asset) -> Quote:
        assert self.data is not None, "call reset() first"
        series = self.data.ensure(asset.symbol)
        ts_ns = int(self._calendar[self._k])
        i = series.index_le(ts_ns)
        if i is None:
            raise KeyError(f"No price for {asset.symbol} at {self.now.isoformat()} (before first bar)")
        last = float(series.close[i])
        nxt = min(i + 1, len(series.sigma) - 1)
        ctx = MarketContext(asset=asset, timestamp=self.now, open=last, high=last, low=last, close=last,
                            volume=float("nan"), sigma=float(series.sigma[nxt]), sigma_typical=series.sigma_typical,
                            typical_volume=float(series.typical_volume[nxt]), spread_mult=float(series.spread_mult[i]))
        bid, ask = self.engine.touch(ctx, last, at_entry=False)
        return Quote(symbol=asset.symbol, timestamp=ns_to_dt(int(series.ts[i])) + timedelta(seconds=self.timeframe.seconds),
                     bid=bid, ask=ask, last=last, modelled=True)

    def get_historical_data(self, symbol: str, timeframe: Timeframe | str | None = None, limit: int = 100,
                            indicators: bool = False) -> pd.DataFrame:
        """Completed bars up to the current time (never future data).

        ``timeframe`` may differ from the simulation timeframe: coarser bars are
        aggregated from the simulation bars, finer bars are fetched from the provider.
        """
        assert self.data is not None, "call reset() first"
        tf = Timeframe.parse(timeframe) if timeframe else self.timeframe
        now_ns = dt_to_ns(self.now)
        cur_ns = int(self._calendar[self._k])
        extra = 60 if indicators else 0
        if tf is self.timeframe:
            df = self.data.frame_upto(symbol, cur_ns, limit + extra)
        elif tf.seconds > self.timeframe.seconds:
            ratio = tf.seconds // self.timeframe.seconds
            base = self.data.frame_upto(symbol, cur_ns, (limit + extra + 2) * ratio * 2)
            df = resample_bars(base[["open", "high", "low", "close", "volume"]], tf)
            df = df[df.index.asi8 + tf.seconds * 1_000_000_000 <= now_ns]  # complete bars only
        else:
            key = (symbol, tf.value)
            df = self._aux_cache.get(key)
            if df is None:
                start = self.data.start - pd.Timedelta(days=10)
                df = self.provider.get_bars(self.universe.get(symbol), tf, start.to_pydatetime(),
                                            self.data.end.to_pydatetime())
                self._aux_cache[key] = df
            df = df[df.index.asi8 + tf.seconds * 1_000_000_000 <= now_ns]
        df = df[["open", "high", "low", "close", "volume"]]
        if indicators:
            df = add_indicators(df)
        return df.tail(limit)

    # ------------------------------------------------------------ persistence
    def save_state(self, path: str | Path) -> None:
        """Snapshot everything needed to resume this run exactly (data is re-fetched on load)."""
        assert self.data is not None
        write_json(path, {
            "version": 1,
            "kind": "historical",
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "symbols": self.symbols,
            "start": self.data.start.isoformat(),
            "end": self.data.end.isoformat(),
            "k": self._k,
            "start_k": self._start_k,
            "end_k": self._end_k,
            "portfolio": self.portfolio.state(),
            "orders": [o.to_dict() for o in self.orders.values()],
            "open_order_ids": self.open_order_ids,
            "order_id_counter": self.order_ids.counter,
            "fill_id_counter": self.fill_ids.counter,
            "rng_exec": self.rng_exec.bit_generator.state,
            "rng_episode": self.rng_episode.bit_generator.state,
            "bankrupt": self.bankrupt,
        })

    @classmethod
    def load_state(cls, path: str | Path, provider: DataProvider, universe: AssetUniverse | None = None,
                   config: SimulatorConfig | None = None) -> "TradingSimulator":
        """Restore a run saved with :meth:`save_state`. Pass ``config`` to supply custom objects
        (e.g. a custom ``FeeModel`` or ``Scenario``) that are stored by name only."""
        st = read_json(path)
        cfg = config or SimulatorConfig.from_dict(st["config"])
        sim = cls(universe, provider, cfg)
        sim.reset(symbols=st["symbols"], start=st["start"], end=st["end"])
        sim._k = int(st["k"])
        sim._start_k = int(st.get("start_k", 0))
        sim._end_k = int(st.get("end_k", sim.num_bars - 1))
        sim.run_id = st["run_id"]
        sim.portfolio = Portfolio.from_state(st["portfolio"])
        sim.orders = {d["id"]: Order.from_dict(d) for d in st["orders"]}
        sim.open_order_ids = list(st["open_order_ids"])
        sim.order_ids.counter = st["order_id_counter"]
        sim.fill_ids.counter = st["fill_id_counter"]
        sim.rng_exec.bit_generator.state = st["rng_exec"]
        sim.rng_episode.bit_generator.state = st["rng_episode"]
        sim.bankrupt = st["bankrupt"]
        return sim
