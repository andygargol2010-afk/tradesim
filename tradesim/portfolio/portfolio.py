"""Portfolio accounting in exact decimal arithmetic.

Conventions:

* ``cash`` is the account's cash balance in USD. Buying debits
  ``qty * price + fee``; selling (including short-selling) credits
  ``qty * price - fee``. Negative cash means borrowing on margin.
* ``Position.quantity`` is signed (negative = short). Cost basis uses the
  *average cost* method; realised PnL is booked whenever a fill reduces
  exposure. A fill that crosses through zero closes the old position and
  opens a new one at the fill price.
* Entry fees are attributed pro rata to the quantity later closed, so each
  :class:`ClosedTrade` carries its full round-trip cost.
* ``equity = cash + sum(quantity * mark_price)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from tradesim.types import ZERO, ClosedTrade, D, Fill, Side, fmt

SECONDS_PER_YEAR = Decimal(365 * 24 * 3600)


@dataclass
class Position:
    symbol: str
    quantity: Decimal = ZERO
    avg_price: Decimal = ZERO
    open_fees: Decimal = ZERO  # entry fees not yet attributed to a closed trade
    opened_at: datetime | None = None
    realized_pnl: Decimal = ZERO  # gross of fees, lifetime for this symbol
    fees_paid: Decimal = ZERO  # lifetime for this symbol
    mark_price: Decimal = ZERO

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def direction(self) -> str:
        return "long" if self.quantity > 0 else "short" if self.quantity < 0 else "flat"

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.mark_price

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.mark_price - self.avg_price) * self.quantity

    def to_dict(self, equity: Decimal | None = None) -> dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "quantity": fmt(self.quantity),
            "direction": self.direction,
            "avg_price": fmt(self.avg_price, 10),
            "mark_price": fmt(self.mark_price, 10),
            "market_value": str(self.market_value.quantize(Decimal("0.01"))),
            "cost_basis": str(self.cost_basis.quantize(Decimal("0.01"))),
            "unrealized_pnl": str(self.unrealized_pnl.quantize(Decimal("0.01"))),
            "unrealized_pnl_pct": float(self.unrealized_pnl / abs(self.cost_basis)) if self.cost_basis else 0.0,
            "realized_pnl": str(self.realized_pnl.quantize(Decimal("0.01"))),
            "fees_paid": str(self.fees_paid.quantize(Decimal("0.01"))),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }
        if equity:
            d["weight"] = float(self.market_value / equity)
        return d

    def state(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Decimal) else v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.__dict__.items()}

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Position":
        return cls(
            symbol=d["symbol"], quantity=D(d["quantity"]), avg_price=D(d["avg_price"]), open_fees=D(d["open_fees"]),
            opened_at=datetime.fromisoformat(d["opened_at"]) if d.get("opened_at") else None,
            realized_pnl=D(d["realized_pnl"]), fees_paid=D(d["fees_paid"]), mark_price=D(d["mark_price"]),
        )


@dataclass(frozen=True)
class MarginConfig:
    """Leverage & financing. Defaults = plain cash account (no leverage, no shorts)."""

    max_leverage: Decimal = Decimal(1)  # gross exposure / equity allowed when opening positions
    maintenance_margin: Decimal = Decimal("0.25")  # liquidate when equity < mm * gross exposure
    allow_short: bool = False
    margin_interest_annual: Decimal = Decimal("0.07")  # charged on negative cash
    short_borrow_fee_annual: Decimal = Decimal("0.005")  # charged on short market value (easy-to-borrow)
    cash_interest_annual: Decimal = ZERO  # paid on positive idle cash (many brokers pay ~0-4%)


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: Decimal
    cash: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal


class Portfolio:
    def __init__(self, initial_cash: Decimal | float | str = Decimal(10_000), margin: MarginConfig | None = None):
        self.initial_cash = D(initial_cash)
        self.cash = self.initial_cash
        self.margin = margin or MarginConfig()
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []
        self.total_fees = ZERO
        self.total_slippage = ZERO
        self.total_financing = ZERO  # interest + borrow fees paid (positive = cost)
        self.peak_equity = self.initial_cash
        self.max_drawdown = ZERO  # fraction, positive number

    # ------------------------------------------------------------------ queries
    def position(self, symbol: str) -> Position:
        pos = self.positions.get(symbol)
        if pos is None:
            pos = Position(symbol)
            self.positions[symbol] = pos
        return pos

    def quantity(self, symbol: str) -> Decimal:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else ZERO

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    @property
    def market_value(self) -> Decimal:
        return sum((p.market_value for p in self.positions.values()), ZERO)

    @property
    def equity(self) -> Decimal:
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> Decimal:
        return sum((abs(p.market_value) for p in self.positions.values()), ZERO)

    @property
    def net_exposure(self) -> Decimal:
        return self.market_value

    @property
    def leverage(self) -> Decimal:
        eq = self.equity
        return self.gross_exposure / eq if eq > 0 else Decimal("Infinity")

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self.positions.values()), ZERO)

    @property
    def realized_pnl(self) -> Decimal:
        return sum((p.realized_pnl for p in self.positions.values()), ZERO)

    def buying_power(self, reserved: Decimal = ZERO) -> Decimal:
        """Notional that can still be added to gross exposure.

        Cash account (max_leverage == 1): available cash. Margin account:
        ``max_leverage * equity - gross_exposure``. ``reserved`` = notional
        already committed to open orders.
        """
        if self.margin.max_leverage <= 1:
            bp = self.cash
        else:
            bp = self.margin.max_leverage * self.equity - self.gross_exposure
        return max(bp - reserved, ZERO)

    def is_margin_call(self) -> bool:
        gross = self.gross_exposure
        if gross == 0:
            return False
        if self.margin.max_leverage <= 1 and not self.margin.allow_short:
            return self.equity <= 0
        return self.equity < self.margin.maintenance_margin * gross

    # ----------------------------------------------------------------- updates
    def mark(self, symbol: str, price: Decimal | float) -> None:
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.mark_price = D(price)

    def apply_fill(self, fill: Fill) -> tuple[Fill, list[ClosedTrade]]:
        """Book a fill. Returns the fill (with ``realized_pnl`` set) and any closed trades."""
        pos = self.position(fill.symbol)
        signed = fill.quantity * fill.side.sign
        qty_before = pos.quantity
        closed: list[ClosedTrade] = []
        realized = ZERO

        if fill.side is Side.BUY:
            self.cash -= fill.quantity * fill.price + fill.fee
        else:
            self.cash += fill.quantity * fill.price - fill.fee
        self.total_fees += fill.fee
        self.total_slippage += fill.slippage_cost
        pos.fees_paid += fill.fee

        reducing = qty_before != 0 and (qty_before > 0) != (signed > 0)
        if reducing:
            close_qty = min(abs(signed), abs(qty_before))
            direction = 1 if qty_before > 0 else -1
            realized = (fill.price - pos.avg_price) * close_qty * direction
            entry_fee_part = pos.open_fees * close_qty / abs(qty_before)
            exit_fee_part = fill.fee * close_qty / fill.quantity
            pos.open_fees -= entry_fee_part
            pos.realized_pnl += realized
            closed.append(ClosedTrade(
                symbol=fill.symbol, direction="long" if direction > 0 else "short", quantity=close_qty,
                entry_price=pos.avg_price, exit_price=fill.price,
                pnl=realized - entry_fee_part - exit_fee_part, fees=entry_fee_part + exit_fee_part,
                opened_at=pos.opened_at, closed_at=fill.timestamp,
            ))
            new_qty = qty_before + signed
            if new_qty == 0:
                pos.quantity, pos.avg_price, pos.open_fees, pos.opened_at = ZERO, ZERO, ZERO, None
            elif (new_qty > 0) == (qty_before > 0):
                pos.quantity = new_qty  # partial reduction: avg price unchanged
            else:  # flipped through zero: remainder opens a new position at the fill price
                pos.quantity = new_qty
                pos.avg_price = fill.price
                pos.open_fees = fill.fee - exit_fee_part
                pos.opened_at = fill.timestamp
        else:  # opening or adding
            new_qty = qty_before + signed
            pos.avg_price = (pos.avg_price * abs(qty_before) + fill.price * fill.quantity) / abs(new_qty)
            pos.quantity = new_qty
            pos.open_fees += fill.fee
            if qty_before == 0:
                pos.opened_at = fill.timestamp
        if pos.mark_price == 0:
            pos.mark_price = fill.price

        booked = replace(fill, realized_pnl=realized)
        self.fills.append(booked)
        self.closed_trades.extend(closed)
        return booked, closed

    def accrue_financing(self, seconds: float) -> Decimal:
        """Charge margin interest / borrow fees (and pay cash interest) for ``seconds`` elapsed."""
        if seconds <= 0:
            return ZERO
        frac = D(seconds) / SECONDS_PER_YEAR
        cost = ZERO
        if self.cash < 0:
            cost += -self.cash * self.margin.margin_interest_annual * frac
        elif self.margin.cash_interest_annual > 0:
            cost -= self.cash * self.margin.cash_interest_annual * frac
        short_value = sum((-p.market_value for p in self.positions.values() if p.quantity < 0), ZERO)
        cost += short_value * self.margin.short_borrow_fee_annual * frac
        self.cash -= cost
        self.total_financing += cost
        return cost

    def record_equity(self, timestamp: datetime) -> EquityPoint:
        eq = self.equity
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity > 0:
            self.max_drawdown = max(self.max_drawdown, (self.peak_equity - eq) / self.peak_equity)
        point = EquityPoint(timestamp, eq, self.cash, self.gross_exposure, self.net_exposure)
        self.equity_curve.append(point)
        return point

    # ---------------------------------------------------------------- reporting
    def summary(self) -> dict[str, Any]:
        eq = self.equity
        q = Decimal("0.01")
        dd = (self.peak_equity - eq) / self.peak_equity if self.peak_equity > 0 else ZERO
        return {
            "equity": str(eq.quantize(q)),
            "cash": str(self.cash.quantize(q)),
            "initial_cash": str(self.initial_cash.quantize(q)),
            "market_value": str(self.market_value.quantize(q)),
            "total_return_pct": float((eq / self.initial_cash - 1) * 100) if self.initial_cash else 0.0,
            "realized_pnl": str(self.realized_pnl.quantize(q)),
            "unrealized_pnl": str(self.unrealized_pnl.quantize(q)),
            "total_fees": str(self.total_fees.quantize(q)),
            "total_slippage_cost": str(self.total_slippage.quantize(q)),
            "total_financing": str(self.total_financing.quantize(q)),
            "gross_exposure": str(self.gross_exposure.quantize(q)),
            "net_exposure": str(self.net_exposure.quantize(q)),
            "leverage": float(self.leverage) if eq > 0 else None,
            "buying_power": str(self.buying_power().quantize(q)),
            "current_drawdown_pct": float(dd * 100),
            "max_drawdown_pct": float(self.max_drawdown * 100),
            "num_positions": len(self.open_positions()),
            "num_fills": len(self.fills),
            "num_closed_trades": len(self.closed_trades),
            "positions": [p.to_dict(eq if eq > 0 else None) for p in self.open_positions()],
        }

    # ----------------------------------------------------------- persistence
    def state(self) -> dict[str, Any]:
        return {
            "initial_cash": str(self.initial_cash),
            "cash": str(self.cash),
            "margin": {k: (str(v) if isinstance(v, Decimal) else v) for k, v in self.margin.__dict__.items()},
            "positions": [p.state() for p in self.positions.values()],
            "fills": [f.to_dict() for f in self.fills],
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity_curve": [[p.timestamp.isoformat(), str(p.equity), str(p.cash), str(p.gross_exposure),
                              str(p.net_exposure)] for p in self.equity_curve],
            "total_fees": str(self.total_fees),
            "total_slippage": str(self.total_slippage),
            "total_financing": str(self.total_financing),
            "peak_equity": str(self.peak_equity),
            "max_drawdown": str(self.max_drawdown),
        }

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Portfolio":
        m = d["margin"]
        margin = MarginConfig(**{k: (v if isinstance(v, bool) else D(v)) for k, v in m.items()})
        p = cls(D(d["initial_cash"]), margin)
        p.cash = D(d["cash"])
        p.positions = {x["symbol"]: Position.from_state(x) for x in d["positions"]}
        p.fills = [Fill.from_dict(x) for x in d["fills"]]
        p.closed_trades = [ClosedTrade.from_dict(x) for x in d["closed_trades"]]
        p.equity_curve = [EquityPoint(datetime.fromisoformat(t), D(e), D(c), D(g), D(n))
                          for t, e, c, g, n in d["equity_curve"]]
        p.total_fees = D(d["total_fees"])
        p.total_slippage = D(d["total_slippage"])
        p.total_financing = D(d["total_financing"])
        p.peak_equity = D(d["peak_equity"])
        p.max_drawdown = D(d["max_drawdown"])
        return p
