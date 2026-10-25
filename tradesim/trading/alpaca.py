"""Alpaca broker adapter: run an agent on a real (or Alpaca paper) account with no code changes.

``AlpacaTrader`` exposes the same interface agents already use (``observe``, ``apply_action``,
``place_order``, ``get_portfolio``...), so an agent developed and validated in the simulator and in
daily paper trading can be pointed at a brokerage account by changing one word in the config
(``"broker": "alpaca-paper"`` / ``"alpaca-live"``) and copying its learned state file.

How it works:

* Orders pass the same local pre-trade checks as in the simulator (buying power, minimum notional,
  no shorting, position cap) and are then sent to Alpaca. Nothing is filled locally.
* ``sync()`` pulls each working order from Alpaca and books the newly filled quantity at the
  broker's fill price into a local *shadow ledger* (the same ``Portfolio`` the simulators use), so
  metrics, trade history and the dashboard work identically for paper and real accounts.
* Every sync reconciles the ledger with the broker: cash is set to the broker's cash and position
  mismatches are reported in ``reconciliation`` (the broker is always the source of truth).

Safety: real money (``paper=False``) is refused unless ``allow_live=True`` or the environment
variable ``TRADESIM_ALLOW_LIVE=1`` is set. Credentials come from ``ALPACA_API_KEY`` /
``ALPACA_SECRET_KEY`` (or arguments); never commit them.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.base import DataProvider
from tradesim.live import LivePaperTrader
from tradesim.persistence import read_json
from tradesim.portfolio.portfolio import Portfolio
from tradesim.simulator import SimulatorConfig
from tradesim.types import (
    ZERO, AssetClass, D, Fill, Liquidity, Order, OrderStatus, OrderType, Side, TimeInForce,
)

log = logging.getLogger(__name__)

_STATUS = {
    "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELLED, "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED, "done_for_day": OrderStatus.EXPIRED, "stopped": OrderStatus.CANCELLED,
    "suspended": OrderStatus.CANCELLED,
}


def broker_symbol(asset: Asset) -> str:
    if asset.asset_class is AssetClass.CRYPTO:
        return asset.symbol.replace("-USD", "/USD")
    return asset.symbol.replace("-", ".")


def local_symbol(sym: str, universe: AssetUniverse) -> str:
    """Alpaca position symbols: ``BTCUSD`` / ``BTC/USD`` for crypto, ``BRK.B`` for share classes."""
    s = sym.replace("/", "")
    if s.endswith("USD") and f"{s[:-3]}-USD" in universe:
        return f"{s[:-3]}-USD"
    return sym.replace(".", "-")


def _val(x: Any) -> str:
    return x.value if hasattr(x, "value") else str(x)


class AlpacaTrader(LivePaperTrader):
    """A brokerage account behind the agent interface (see module docstring).

    Args:
        paper: ``True`` = Alpaca paper account (fake money, real market); ``False`` = real money.
        data_provider: market data for observations (default: Alpaca market data with the same keys).
        client: an ``alpaca.trading.client.TradingClient`` (built from the keys if omitted; injectable for tests).
        allow_live: required (or ``TRADESIM_ALLOW_LIVE=1``) when ``paper=False``.
    """

    def __init__(self, paper: bool = True, api_key: str | None = None, secret_key: str | None = None,
                 universe: AssetUniverse | None = None, config: SimulatorConfig | None = None,
                 data_provider: DataProvider | None = None, clock=None, symbols: list[str] | None = None,
                 client: Any = None, allow_live: bool | None = None):
        if not paper and not (allow_live or os.environ.get("TRADESIM_ALLOW_LIVE") == "1"):
            raise PermissionError("Real-money trading is disabled. Set TRADESIM_ALLOW_LIVE=1 (or allow_live=True) "
                                  "only after validating the agent in paper trading.")
        if client is None:
            from alpaca.trading.client import TradingClient

            api_key = api_key or os.environ.get("ALPACA_API_KEY")
            secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                raise PermissionError("ALPACA_API_KEY / ALPACA_SECRET_KEY are required for an Alpaca account")
            client = TradingClient(api_key, secret_key, paper=paper)
        if data_provider is None:
            from tradesim.data.alpaca_provider import AlpacaProvider

            data_provider = AlpacaProvider(api_key, secret_key)
        super().__init__(data_provider, universe, config, clock=clock, symbols=symbols, use_order_book=False)
        self.paper = paper
        self.client = client
        self.broker_ids: dict[str, str] = {}  # local order id -> broker order id
        self.booked: dict[str, list[str]] = {}  # broker order id -> [booked qty, booked notional]
        self.reconciliation: list[dict[str, Any]] = []
        self.broker_equity: float | None = None
        self._bootstrap_from_broker()

    # ------------------------------------------------------------- ledger
    def _bootstrap_from_broker(self) -> None:
        """New ledger: start from the broker's cash and import positions opened outside tradesim."""
        acct = self.client.get_account()
        cash = D(str(acct.cash))
        self.portfolio = Portfolio(cash + sum((D(str(p.qty)) * D(str(p.avg_entry_price))
                                               for p in self.client.get_all_positions()), ZERO), self.config.margin)
        self.portfolio.cash = self.portfolio.initial_cash
        for p in self.client.get_all_positions():
            sym = local_symbol(p.symbol, self.universe)
            if sym not in self.universe:
                continue
            fill = Fill(order_id="imported", symbol=sym, side=Side.BUY, quantity=D(str(p.qty)),
                        price=D(str(p.avg_entry_price)), fee=ZERO, liquidity=Liquidity.TAKER, timestamp=self.now,
                        id=self.fill_ids())
            self.portfolio.apply_fill(fill)
        self.portfolio.equity_curve.clear()
        self._record_equity(self.now)

    def state(self) -> dict[str, Any]:
        st = super().state()
        st.update(kind="alpaca", paper=self.paper, broker_ids=self.broker_ids, booked=self.booked,
                  reconciliation=self.reconciliation[-50:], broker_equity=self.broker_equity)
        return st

    def load_ledger(self, path: str | Path) -> None:
        st = read_json(path)
        self.restore(st)
        self.broker_ids = st.get("broker_ids", {})
        self.booked = st.get("booked", {})
        self.reconciliation = st.get("reconciliation", [])
        self.broker_equity = st.get("broker_equity")

    # ------------------------------------------------------------- orders
    def _request(self, order: Order, asset: Asset):
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.enums import TimeInForce as ATif
        from alpaca.trading.requests import (
            LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest,
        )

        crypto = asset.asset_class is AssetClass.CRYPTO
        fractional = order.quantity != order.quantity.to_integral_value()
        if crypto:
            tif = ATif.IOC if order.time_in_force is TimeInForce.IOC else ATif.GTC
        elif order.time_in_force is TimeInForce.GTC and not fractional and order.order_type is not OrderType.MARKET:
            tif = ATif.GTC
        else:
            tif = ATif.DAY  # Alpaca requires DAY for fractional and market equity orders
        common = dict(symbol=broker_symbol(asset), qty=float(order.quantity),
                      side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL, time_in_force=tif,
                      client_order_id=f"ts-{order.id}-{uuid.uuid4().hex[:8]}")
        if order.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order.order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=float(order.limit_price), **common)
        if order.order_type is OrderType.STOP:
            return StopOrderRequest(stop_price=float(order.stop_price), **common)
        return StopLimitOrderRequest(stop_price=float(order.stop_price), limit_price=float(order.limit_price), **common)

    def place_order(self, symbol: str, side: Any, quantity: Any = None, **kw: Any) -> dict[str, Any]:
        res = super().place_order(symbol, side, quantity, **kw)  # local pre-trade checks
        if res["status"] != "new":
            return res
        order = self.orders[res["id"]]
        try:
            bo = self.client.submit_order(self._request(order, self.universe.get(symbol)))
            self.broker_ids[order.id] = str(bo.id)
            self.events.emit("broker_submitted", self.now, order_id=order.id, broker_id=str(bo.id))
        except Exception as exc:  # broker refusal (market closed, not tradable, insufficient funds...)
            self._close_order(order, OrderStatus.REJECTED, f"broker rejected: {exc}")
        return order.to_dict()

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        bid = self.broker_ids.get(order_id)
        if bid and order_id in self.open_order_ids:
            try:
                self.client.cancel_order_by_id(bid)
            except Exception as exc:  # noqa: BLE001 - already filled / cancelled at the broker
                log.warning("cancel %s at broker failed: %s", order_id, exc)
        return super().cancel_order(order_id)

    def cancel_all_orders(self, symbol: str | None = None) -> int:
        ids = [i for i in self.open_order_ids if symbol is None or self.orders[i].symbol == symbol]
        for i in ids:
            self.cancel_order(i)
        return len(ids)

    # ------------------------------------------------------------- sync
    def sync(self) -> list[dict[str, Any]]:
        """Book new broker fills into the ledger, reconcile cash / positions, mark, record equity."""
        now = self.now
        fills: list[Fill] = []
        for oid in list(self.open_order_ids):
            bid = self.broker_ids.get(oid)
            if bid is None:
                continue
            bo = self.client.get_order_by_id(bid)
            order = self.orders[oid]
            filled = D(str(bo.filled_qty or 0))
            done_qty, done_notional = (D(x) for x in self.booked.get(bid, ["0", "0"]))
            if filled > done_qty and bo.filled_avg_price is not None:
                total_notional = filled * D(str(bo.filled_avg_price))
                qty = filled - done_qty
                price = (total_notional - done_notional) / qty  # price of the newly filled slice
                liquidity = Liquidity.MAKER if order.order_type is OrderType.LIMIT else Liquidity.TAKER
                sched = self.fee_model.schedule_for(self.universe.get(order.symbol).asset_class)
                fee = sched.compute(order.side, qty, price, liquidity, order.fees_paid, order.filled_quantity == 0)
                ts = bo.filled_at or now
                fill = Fill(id=self.fill_ids(), order_id=oid, symbol=order.symbol, side=order.side, quantity=qty,
                            price=price, fee=fee, liquidity=liquidity, timestamp=ts)
                booked, _ = self.portfolio.apply_fill(fill)
                fills.append(booked)
                order.avg_fill_price = (order.avg_fill_price * order.filled_quantity + price * qty) / filled
                order.filled_quantity = filled
                order.fees_paid += fee
                order.status = OrderStatus.PARTIALLY_FILLED
                self.booked[bid] = [str(filled), str(total_notional)]
                self.events.emit("fill", now, fill=booked.to_dict())
            status = _val(bo.status)
            if status in _STATUS:
                self._close_order(order, OrderStatus.FILLED if filled >= order.quantity else _STATUS[status],
                                  None if status == "filled" else f"broker status: {status}")
        self._reconcile(now)  # broker-side interest and real fees show up here, through cash
        self._last_sync = now
        self._record_equity(now)
        return [f.to_dict() for f in fills]

    def _reconcile(self, now: datetime) -> None:
        acct = self.client.get_account()
        self.broker_equity = float(acct.equity)
        broker_cash = D(str(acct.cash))
        diff = broker_cash - self.portfolio.cash
        if abs(diff) > Decimal("0.01"):
            # Real fees, rounding and interest differ from the model: the broker's cash wins.
            self.portfolio.cash = broker_cash
            self.portfolio.total_financing -= diff
            self.reconciliation.append({"time": now.isoformat(), "kind": "cash", "adjustment": str(diff)})
        broker_pos = {local_symbol(p.symbol, self.universe): p for p in self.client.get_all_positions()}
        for sym in set(broker_pos) | {p.symbol for p in self.portfolio.open_positions()}:
            bq = D(str(broker_pos[sym].qty)) if sym in broker_pos else ZERO
            lq = self.portfolio.quantity(sym)
            if abs(bq - lq) > Decimal("0.000001"):
                self.reconciliation.append({"time": now.isoformat(), "kind": "position", "symbol": sym,
                                            "broker_qty": str(bq), "ledger_qty": str(lq)})
                log.warning("position mismatch %s: broker %s vs ledger %s", sym, bq, lq)
            if sym in broker_pos and broker_pos[sym].current_price is not None:
                self.portfolio.mark(sym, D(str(broker_pos[sym].current_price)))
        del self.reconciliation[:-200]
