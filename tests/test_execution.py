from decimal import Decimal

import pytest

from conftest import make_ctx, make_engine
from tradesim.data.base import OrderBookSnapshot
from tradesim.execution.fees import FeeModel, FeeSchedule, SCHEDULES
from tradesim.portfolio.portfolio import Portfolio
from tradesim.types import Liquidity, Order, OrderStatus, OrderType, Side, TimeInForce


def order(side=Side.BUY, qty="10", typ=OrderType.MARKET, limit=None, stop=None, tif=TimeInForce.GTC):
    return Order(symbol="TEST", side=side, quantity=Decimal(qty), order_type=typ,
                 limit_price=None if limit is None else Decimal(str(limit)),
                 stop_price=None if stop is None else Decimal(str(stop)), time_in_force=tif, id="O1")


def test_market_buy_pays_ask_at_next_open(stock, portfolio):
    eng = make_engine()
    o = order()
    fills = eng.execute([o], make_ctx(stock, 100, 101, 99, 100.5), portfolio)
    hs = eng.half_spread_frac(make_ctx(stock, 100, 101, 99, 100.5))
    assert o.status is OrderStatus.FILLED
    assert fills[0].liquidity is Liquidity.TAKER
    # ask = open + half spread (>= half a tick), rounded up to the next cent
    assert fills[0].price == Decimal("100.01") and 100 * hs <= 0.01


def test_market_sell_receives_bid_rounded_down(stock, portfolio):
    eng = make_engine()
    o = order(side=Side.SELL, qty="5")
    fills = eng.execute([o], make_ctx(stock, 50.003, 51, 49, 50), Portfolio(10_000))
    assert fills[0].price <= Decimal("50.00")
    assert fills[0].price == fills[0].price.quantize(Decimal("0.01"))


def test_limit_buy_not_filled_if_price_never_trades_through(stock, portfolio):
    eng = make_engine()
    o = order(typ=OrderType.LIMIT, limit=95)
    assert eng.execute([o], make_ctx(stock, 100, 101, 95.0, 100), portfolio) == []  # touch only, p=0
    assert o.status is OrderStatus.NEW


def test_limit_buy_fills_at_limit_as_maker(stock, portfolio):
    eng = make_engine()
    o = order(typ=OrderType.LIMIT, limit=95)
    fills = eng.execute([o], make_ctx(stock, 100, 101, 94, 100), portfolio)
    assert fills[0].price == Decimal("95")
    assert fills[0].liquidity is Liquidity.MAKER


def test_marketable_limit_gets_price_improvement(stock, portfolio):
    eng = make_engine()
    o = order(typ=OrderType.LIMIT, limit=105)
    fills = eng.execute([o], make_ctx(stock, 100, 101, 99, 100), portfolio)
    assert fills[0].price < Decimal("100.10")
    assert fills[0].liquidity is Liquidity.TAKER


def test_sell_stop_gap_fills_at_open_not_stop(stock):
    eng = make_engine()
    pf = Portfolio(10_000)
    o = order(side=Side.SELL, typ=OrderType.STOP, stop=95)
    fills = eng.execute([o], make_ctx(stock, 90, 91, 88, 89), pf)  # gapped below the stop
    assert o.triggered
    assert fills[0].price <= Decimal("90")  # filled through the gap, ~open minus half spread


def test_buy_stop_triggers_intrabar(stock, portfolio):
    eng = make_engine()
    o = order(typ=OrderType.STOP, stop=102)
    fills = eng.execute([o], make_ctx(stock, 100, 103, 99, 102.5), portfolio)
    assert fills and fills[0].price >= Decimal("102")


def test_stop_not_triggered(stock, portfolio):
    eng = make_engine()
    o = order(typ=OrderType.STOP, stop=110)
    assert eng.execute([o], make_ctx(stock, 100, 103, 99, 102.5), portfolio) == []
    assert not o.triggered


def test_stop_limit_triggered_but_limit_not_reached(stock, portfolio):
    eng = make_engine()
    o = order(side=Side.SELL, typ=OrderType.STOP_LIMIT, stop=95, limit=94)
    pf = Portfolio(10_000)
    fills = eng.execute([o], make_ctx(stock, 90, 91, 88, 89), pf)  # gap below the limit
    assert o.triggered and fills == []
    assert o.status is OrderStatus.NEW  # rests as a limit at 94


def test_participation_cap_partial_fill(stock, portfolio):
    eng = make_engine(participation_rate=0.1)
    o = order(qty="1000")
    fills = eng.execute([o], make_ctx(stock, 100, 101, 99, 100, volume=5000, typical_volume=5000), portfolio)
    assert fills[0].quantity == Decimal("500")
    assert o.status is OrderStatus.PARTIALLY_FILLED
    o2 = order(qty="1000", tif=TimeInForce.IOC)
    o2.id = "O2"
    eng.execute([o2], make_ctx(stock, 100, 101, 99, 100, volume=5000, typical_volume=5000), portfolio)
    assert o2.status is OrderStatus.CANCELLED and o2.filled_quantity == Decimal("500")


def test_fok_all_or_nothing(stock, portfolio):
    eng = make_engine(participation_rate=0.1)
    o = order(qty="1000", tif=TimeInForce.FOK)
    assert eng.execute([o], make_ctx(stock, 100, 101, 99, 100, volume=5000, typical_volume=5000), portfolio) == []
    assert o.status is OrderStatus.CANCELLED and o.filled_quantity == 0


def test_square_root_impact_grows_with_size(stock):
    eng = make_engine(slippage_model="sqrt_impact")
    prices = []
    for qty in ("10", "1000", "10000"):
        pf = Portfolio(10_000_000)
        o = order(qty=qty)
        f = eng.execute([o], make_ctx(stock, 100, 100, 100, 100, volume=1e6, typical_volume=1e5, sigma=0.02), pf)
        prices.append(f[0].price)
    assert prices[0] < prices[1] < prices[2]
    # 10k units vs 100k typical: ~ 2/3 * 2% * sqrt(0.1) ~ 42 bps of impact
    assert Decimal("100.3") < prices[2] < Decimal("100.6")


def test_liquidity_consumed_within_bar_is_shared(stock):
    eng = make_engine(slippage_model="sqrt_impact")
    pf = Portfolio(10_000_000)
    a, b = order(qty="5000"), order(qty="5000")
    b.id = "O2"
    fa, fb = eng.execute([a, b], make_ctx(stock, 100, 100, 100, 100, volume=1e6, typical_volume=1e5, sigma=0.02), pf)
    assert fb.price > fa.price  # the second slice walks deeper into the book


def test_real_l2_book_is_swept(coin):
    eng = make_engine(slippage_model="sqrt_impact")
    book = OrderBookSnapshot("COIN-USD", None, bids=[(99.0, 1.0)], asks=[(100.0, 1.0), (101.0, 1.0), (105.0, 10.0)])
    ctx = make_ctx(coin, 100, 100, 100, 100, volume=float("nan"), bid=99.0, ask=100.0, book=book, live=True)
    o = Order(symbol="COIN-USD", side=Side.BUY, quantity=Decimal("2.5"), id="O1")
    f = eng.execute([o], ctx, Portfolio(10_000))
    assert f[0].price == Decimal("101.4")  # (100 + 101 + 0.5*105) / 2.5


def test_buying_power_caps_fill():
    from tradesim.assets.universe import Asset
    from tradesim.types import AssetClass

    asset = Asset("TEST", AssetClass.EQUITY, liquidity_tier=1)
    eng = make_engine()
    pf = Portfolio(1_000)
    o = order(qty="100")
    f = eng.execute([o], make_ctx(asset, 100, 100, 100, 100), pf,
                    max_qty=lambda order, a, price: pf.buying_power() / price)
    assert f[0].quantity < Decimal("10.0")
    assert pf.cash >= 0


@pytest.mark.parametrize("shares,price,expected", [("100", "50", "1.00"), ("1000", "50", "5.00"), ("10", "0.5", "0.05")])
def test_ibkr_fixed_min_and_cap(shares, price, expected):
    s = SCHEDULES["ibkr_fixed"]
    fee = s.compute(Side.BUY, Decimal(shares), Decimal(price), Liquidity.TAKER)
    assert fee == Decimal(expected)


def test_regulatory_fees_on_sells_only():
    s = SCHEDULES["alpaca_equity"]
    assert s.compute(Side.BUY, Decimal(100), Decimal(100), Liquidity.TAKER) == 0
    sell = s.compute(Side.SELL, Decimal(100), Decimal(100), Liquidity.TAKER)
    assert sell == (Decimal(10_000) * Decimal("0.0000278") + Decimal(100) * Decimal("0.000166")).quantize(Decimal("1e-8"))


def test_maker_taker_and_fixed_fee():
    s = FeeSchedule("x", maker_pct=Decimal("0.001"), taker_pct=Decimal("0.002"), fixed_per_order=Decimal("1"))
    assert s.compute(Side.BUY, Decimal(10), Decimal(100), Liquidity.MAKER) == Decimal("2")
    assert s.compute(Side.BUY, Decimal(10), Decimal(100), Liquidity.TAKER, first_fill=False) == Decimal("2")
    assert FeeModel.preset("binance").by_class


def test_tick_rounding_sub_dollar(stock, portfolio):
    eng = make_engine()
    o = order(qty="1000")
    f = eng.execute([o], make_ctx(stock, 0.5, 0.5, 0.5, 0.5), portfolio)
    assert f[0].price == f[0].price.quantize(Decimal("0.0001"))
