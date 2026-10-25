"""Core domain types shared across the simulator.

All monetary amounts and quantities that touch the books (cash, positions,
fills, fees) are :class:`decimal.Decimal`. Market data (OHLCV series) stays in
``float64`` numpy/pandas structures for speed and is converted to ``Decimal``
at the execution boundary, where rounding to tick size / lot step happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from enum import Enum
from typing import Any

getcontext().prec = 34  # IEEE 754 decimal128 precision; plenty for 1e-12 lot steps.

ZERO = Decimal(0)
ONE = Decimal(1)


def D(value: Any) -> Decimal:
    """Convert ``value`` to Decimal without binary-float artefacts.

    Floats go through ``repr`` (shortest round-tripping representation), so
    ``D(0.1) == Decimal("0.1")`` rather than ``0.1000000000000000055...``.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a number")
    if isinstance(value, int):
        return Decimal(int(value))
    if isinstance(value, str):
        return Decimal(value.strip())
    value = float(value)  # floats, numpy scalars (np.float64 repr is "np.float64(x)" in numpy 2)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"Cannot convert non-finite number {value!r} to Decimal")
    return Decimal(repr(value))


def fmt(value: Decimal, places: int = 12) -> str:
    """Plain (non-scientific) string for a Decimal, rounded to ``places`` decimals, trailing zeros dropped."""
    q = value.quantize(Decimal(1).scaleb(-places)) if value.as_tuple().exponent < -places else value
    text = format(q, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` toward zero to a multiple of ``step``."""
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def quantize_up(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` away from zero to a multiple of ``step``."""
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GTC = "gtc"  # good till cancelled
    DAY = "day"  # expires at the end of the (UTC) trading day it was placed in
    IOC = "ioc"  # immediate or cancel: whatever fills on the next execution pass, rest cancelled
    FOK = "fok"  # fill or kill: all on the next execution pass or nothing


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


class Liquidity(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


class IdGenerator:
    """Per-simulator sequential ids, so runs are reproducible and independent."""

    def __init__(self, prefix: str, start: int = 1) -> None:
        self.prefix = prefix
        self.counter = start

    def __call__(self) -> str:
        value = f"{self.prefix}{self.counter:08d}"
        self.counter += 1
        return value


@dataclass
class Order:
    """An order as tracked by the execution engine."""

    symbol: str
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: datetime | None = None
    id: str = ""  # assigned by the execution engine on submission
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: Decimal = ZERO
    avg_fill_price: Decimal = ZERO
    fees_paid: Decimal = ZERO
    triggered: bool = False  # stop / stop-limit orders: has the stop been hit?
    updated_at: datetime | None = None
    reject_reason: str | None = None
    tag: str | None = None  # free-form label from the agent (strategy id, rationale, ...)

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.filled_quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": fmt(self.quantity),
            "order_type": self.order_type.value,
            "limit_price": None if self.limit_price is None else fmt(self.limit_price),
            "stop_price": None if self.stop_price is None else fmt(self.stop_price),
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": fmt(self.filled_quantity),
            "avg_fill_price": fmt(self.avg_fill_price),
            "fees_paid": fmt(self.fees_paid),
            "triggered": self.triggered,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "reject_reason": self.reject_reason,
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Order":
        return cls(
            id=d["id"],
            symbol=d["symbol"],
            side=Side(d["side"]),
            quantity=Decimal(d["quantity"]),
            order_type=OrderType(d["order_type"]),
            limit_price=None if d.get("limit_price") is None else Decimal(d["limit_price"]),
            stop_price=None if d.get("stop_price") is None else Decimal(d["stop_price"]),
            time_in_force=TimeInForce(d["time_in_force"]),
            status=OrderStatus(d["status"]),
            filled_quantity=Decimal(d["filled_quantity"]),
            avg_fill_price=Decimal(d["avg_fill_price"]),
            fees_paid=Decimal(d["fees_paid"]),
            triggered=bool(d.get("triggered", False)),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else None,
            updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else None,
            reject_reason=d.get("reject_reason"),
            tag=d.get("tag"),
        )


@dataclass(frozen=True)
class Fill:
    """A single execution (an order can have many fills)."""

    order_id: str
    symbol: str
    side: Side
    quantity: Decimal
    price: Decimal
    fee: Decimal
    liquidity: Liquidity
    timestamp: datetime
    slippage_cost: Decimal = ZERO  # cost vs. the mid/reference price at decision time, in quote ccy
    realized_pnl: Decimal = ZERO  # gross realised PnL this fill produced (filled in by the portfolio)
    id: str = ""

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": fmt(self.quantity),
            "price": fmt(self.price),
            "notional": fmt(self.notional, 8),
            "fee": fmt(self.fee),
            "liquidity": self.liquidity.value,
            "slippage_cost": fmt(self.slippage_cost),
            "realized_pnl": fmt(self.realized_pnl),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fill":
        return cls(
            id=d["id"],
            order_id=d["order_id"],
            symbol=d["symbol"],
            side=Side(d["side"]),
            quantity=Decimal(d["quantity"]),
            price=Decimal(d["price"]),
            fee=Decimal(d["fee"]),
            liquidity=Liquidity(d["liquidity"]),
            slippage_cost=Decimal(d.get("slippage_cost", "0")),
            realized_pnl=Decimal(d.get("realized_pnl", "0")),
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )


@dataclass(frozen=True)
class ClosedTrade:
    """A (partial) round trip: produced whenever a fill reduces an existing position.

    Used for win-rate / profit-factor style statistics. ``pnl`` is net of the
    entry fees attributable to the closed quantity and the exit fee.
    """

    symbol: str
    direction: str  # "long" or "short"
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    fees: Decimal
    opened_at: datetime | None
    closed_at: datetime
    entry_order_id: str | None = None  # order that opened the position (its tag = entry rationale)
    exit_order_id: str | None = None  # order that closed this quantity

    @property
    def allocated(self) -> Decimal:
        """Capital committed to the closed quantity at entry (quantity x entry price)."""
        return self.quantity * self.entry_price

    @property
    def return_pct(self) -> float:
        basis = self.entry_price * self.quantity
        return float(self.pnl / basis) if basis else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": fmt(self.quantity),
            "entry_price": fmt(self.entry_price),
            "exit_price": fmt(self.exit_price),
            "pnl": fmt(self.pnl),
            "fees": fmt(self.fees),
            "return_pct": self.return_pct,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat(),
            "entry_order_id": self.entry_order_id,
            "exit_order_id": self.exit_order_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClosedTrade":
        return cls(
            symbol=d["symbol"],
            direction=d["direction"],
            quantity=Decimal(d["quantity"]),
            entry_price=Decimal(d["entry_price"]),
            exit_price=Decimal(d["exit_price"]),
            pnl=Decimal(d["pnl"]),
            fees=Decimal(d["fees"]),
            opened_at=datetime.fromisoformat(d["opened_at"]) if d.get("opened_at") else None,
            closed_at=datetime.fromisoformat(d["closed_at"]),
            entry_order_id=d.get("entry_order_id"),
            exit_order_id=d.get("exit_order_id"),
        )


@dataclass
class Quote:
    """Top-of-book snapshot (real or modelled)."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_size: float | None = None
    ask_size: float | None = None
    modelled: bool = True  # True when bid/ask were estimated from OHLCV rather than observed

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4 if self.mid > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "spread_bps": round(self.spread_bps, 3),
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "modelled": self.modelled,
        }
