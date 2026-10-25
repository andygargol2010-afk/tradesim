from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from tradesim.portfolio.metrics import compute_metrics
from tradesim.portfolio.portfolio import MarginConfig, Portfolio
from tradesim.types import Fill, Liquidity, Side

T = datetime(2024, 1, 2, tzinfo=timezone.utc)


def fill(side, qty, price, fee="0", sym="X"):
    return Fill(order_id="o", symbol=sym, side=side, quantity=Decimal(qty), price=Decimal(price), fee=Decimal(fee),
                liquidity=Liquidity.TAKER, timestamp=T, id="f")


def test_average_cost_and_realized_pnl():
    p = Portfolio(10_000)
    p.apply_fill(fill(Side.BUY, "1.5", "100", "1"))
    p.apply_fill(fill(Side.BUY, "0.5", "120", "1"))
    pos = p.positions["X"]
    assert pos.quantity == Decimal("2.0")
    assert pos.avg_price == Decimal("105")
    _, closed = p.apply_fill(fill(Side.SELL, "1", "110", "1"))
    assert pos.realized_pnl == Decimal("5")
    # Net PnL of the round trip: 5 gross - half the 2 entry fees - the 1 exit fee
    assert closed[0].pnl == Decimal("3")
    assert p.cash == Decimal(10_000) - Decimal("150") - Decimal("60") - 2 + Decimal("110") - 1
    assert pos.avg_price == Decimal("105")  # unchanged on partial reduction


def test_fractional_precision_is_exact():
    p = Portfolio(1_000)
    for _ in range(10):
        p.apply_fill(fill(Side.BUY, "0.0003", "65000.12"))
    assert p.positions["X"].quantity == Decimal("0.0030")
    p.apply_fill(fill(Side.SELL, "0.003", "65000.12"))
    assert p.positions["X"].quantity == 0
    assert p.cash == Decimal(1_000)


def test_flip_long_to_short():
    p = Portfolio(10_000, MarginConfig(max_leverage=Decimal(2), allow_short=True))
    p.apply_fill(fill(Side.BUY, "10", "50"))
    _, closed = p.apply_fill(fill(Side.SELL, "15", "60"))
    pos = p.positions["X"]
    assert pos.quantity == Decimal("-5")
    assert pos.avg_price == Decimal("60")
    assert closed[0].pnl == Decimal("100")
    p.mark("X", 55)
    assert pos.unrealized_pnl == Decimal("25")
    assert p.equity == p.cash + Decimal("-5") * Decimal("55")


def test_equity_and_drawdown_tracking():
    p = Portfolio(1_000)
    p.apply_fill(fill(Side.BUY, "10", "100"))
    p.mark("X", 100)
    p.record_equity(T)
    p.mark("X", 80)
    p.record_equity(T)
    assert p.equity == Decimal(800)
    assert p.max_drawdown == Decimal("0.2")


def test_financing_accrues_on_margin_debt():
    p = Portfolio(1_000, MarginConfig(max_leverage=Decimal(2), margin_interest_annual=Decimal("0.10")))
    p.apply_fill(fill(Side.BUY, "20", "100"))  # cash -1000
    cost = p.accrue_financing(365 * 24 * 3600)
    assert cost == Decimal("100")


def test_state_roundtrip():
    p = Portfolio(5_000)
    p.apply_fill(fill(Side.BUY, "1.25", "100", "0.5"))
    p.record_equity(T)
    q = Portfolio.from_state(p.state())
    assert q.state() == p.state()


def test_metrics_basic():
    eq = pd.Series([100, 110, 99, 120], index=pd.date_range("2024-01-01", periods=4))
    m = compute_metrics(eq, periods_per_year=252)
    assert m["total_return_pct"] == pytest.approx(20.0)
    assert m["max_drawdown_pct"] == pytest.approx(10.0)
    assert m["sharpe"] > 0
