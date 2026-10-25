from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tradesim import Scenario, SimulatorConfig, SyntheticProvider, TradingSimulator
from tradesim.data.cache import CachedProvider
from tradesim.data.timeframes import Timeframe
from tradesim.market.conditions import FlashCrash, GapInjection, VolatilityScale
from tradesim.portfolio.portfolio import MarginConfig


def test_default_universe_has_1000_plus_assets(universe):
    assert len(universe) >= 1000
    counts = universe.counts()
    assert counts["equity"] > 500 and counts["crypto"] >= 200 and counts["etf"] > 100


def test_orders_execute_on_next_bar_not_current(sim):
    s = sim.data.series["AAPL"]
    while s.index_exact(int(sim._calendar[sim.step_index + 1])) is None:  # land on a stock trading day
        sim.step()
    nxt_i = s.index_exact(int(sim._calendar[sim.step_index + 1]))
    o = sim.place_order("AAPL", "buy", 1)
    assert o["status"] == "new"
    fills = sim.step()
    assert len(fills) == 1
    price = float(fills[0]["price"])
    assert s.low[nxt_i] * 0.99 <= price <= s.high[nxt_i] * 1.01
    assert abs(price - s.open[nxt_i]) / s.open[nxt_i] < 0.01  # filled near the *next* open


def test_no_lookahead_in_history(sim):
    sim.step(10)
    df = sim.get_historical_data("AAPL", limit=500)
    assert (df.index + pd.Timedelta(days=1) <= pd.Timestamp(sim.now)).all()
    weekly = sim.get_historical_data("AAPL", timeframe="1w", limit=50)
    assert (weekly.index + pd.Timedelta(days=7) <= pd.Timestamp(sim.now)).all()


def test_fractional_orders_and_min_notional(sim):
    assert sim.place_order("BTC-USD", "buy", 0.000001)["status"] == "rejected"  # below $1
    r = sim.place_order("AAPL", "buy", "1.234567891234")
    assert r["status"] == "new" and r["quantity"] == "1.234567891"  # quantized to the 1e-9 lot step


def test_rejections(sim):
    assert "buying power" in sim.place_order("SPY", "buy", notional=50_000)["reject_reason"]
    assert "short" in sim.place_order("SPY", "sell", 1)["reject_reason"]
    assert "limit_price" in sim.place_order("SPY", "buy", 1, order_type="limit")["reject_reason"]
    assert "exactly one" in sim.place_order("SPY", "buy", 1, notional=10)["reject_reason"]
    assert sim.place_order("^GSPC", "buy", 1)["status"] == "rejected"  # index: data only


def test_reserved_cash_prevents_double_spending(sim):
    assert sim.place_order("SPY", "buy", notional=6_000)["status"] == "new"
    assert sim.place_order("MSFT", "buy", notional=6_000)["status"] == "rejected"


def test_round_trip_and_metrics(sim):
    sim.place_order("BTC-USD", "buy", notional=3_000)
    sim.step(5)
    qty = sim.portfolio.quantity("BTC-USD")
    assert qty > 0
    sim.place_order("BTC-USD", "sell", qty)
    sim.step(2)
    assert sim.portfolio.quantity("BTC-USD") == 0
    trades = sim.get_trades()
    assert len(trades) == 1
    m = sim.get_metrics()
    assert m["num_trades"] == 1 and "sharpe" in m and m["total_fees"] > 0  # alpaca crypto fees


def test_day_order_expires(sim):
    while sim.data.series["AAPL"].index_exact(int(sim._calendar[sim.step_index + 1])) is None:
        sim.step()
    price = sim.get_price("AAPL")["bid"]
    o = sim.place_order("AAPL", "buy", 1, order_type="limit", limit_price=price * 0.5, time_in_force="day")
    sim.step(4)
    assert sim.get_order(o["id"])["status"] == "expired"


def test_reproducibility(universe):
    def run(seed):
        s = TradingSimulator(universe, SyntheticProvider(seed=3),
                             SimulatorConfig(seed=seed, execution=__import__("tradesim").ExecutionConfig(slippage_noise_bps=5)))
        s.reset(["ETH-USD", "SPY"], "2023-01-01", "2023-02-01")
        for _ in range(20):
            s.place_order("ETH-USD", "buy", notional=100)
            s.step()
        return [f["price"] for f in s.get_fills(100)], str(s.portfolio.equity)
    assert run(1) == run(1)
    assert run(1) != run(2)  # execution noise depends on the seed


def test_save_and_load_state(sim, tmp_path):
    sim.place_order("BTC-USD", "buy", notional=1_000)
    sim.place_order("AAPL", "buy", 2, order_type="limit", limit_price=sim.get_price("AAPL")["bid"] * 0.8)
    sim.step(3)
    path = tmp_path / "state.json"
    sim.save_state(path)
    restored = TradingSimulator.load_state(path, SyntheticProvider(seed=7), sim.universe)
    assert restored.portfolio.state() == sim.portfolio.state()
    assert restored.get_open_orders() == sim.get_open_orders()
    sim.place_order("BTC-USD", "buy", notional=500)
    restored.place_order("BTC-USD", "buy", notional=500)
    sim.step(5)
    restored.step(5)
    assert str(sim.portfolio.equity) == str(restored.portfolio.equity)


def test_margin_and_shorting(universe):
    cfg = SimulatorConfig(initial_cash=10_000, margin=MarginConfig(max_leverage=Decimal(2), allow_short=True))
    s = TradingSimulator(universe, SyntheticProvider(seed=7), cfg)
    s.reset(["SPY", "MSFT"], "2023-01-01", "2023-03-01")
    assert s.place_order("SPY", "buy", notional=15_000)["status"] == "new"
    assert s.place_order("MSFT", "sell", notional=4_000)["status"] == "new"
    s.step(3)
    assert s.portfolio.quantity("MSFT") < 0
    assert s.portfolio.leverage > 1


def test_scenarios_keep_ohlc_consistent(universe):
    df = SyntheticProvider(seed=1).get_bars(universe.get("SPY"), Timeframe.D1, pd.Timestamp("2023-01-01", tz="UTC"),
                                            pd.Timestamp("2024-01-01", tz="UTC"))
    rng = np.random.default_rng(0)
    for t in (VolatilityScale(2.5), GapInjection(0.1, 0.05), FlashCrash(0.3)):
        out = t(df, rng)
        assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (out["low"] > 0).all()
    hv = VolatilityScale(2.0)(df, rng)
    assert np.log(hv.close).diff().std() == pytest.approx(2 * np.log(df.close).diff().std(), rel=0.01)


def test_high_volatility_scenario_runs(universe):
    s = TradingSimulator(universe, SyntheticProvider(seed=7), SimulatorConfig(scenario="high_volatility"))
    s.reset(["SPY"], "2023-01-01", "2023-03-01")
    assert "spread_mult" in s.data.series["SPY"].frame.columns


def test_cached_provider_roundtrip(universe, tmp_path):
    calls = []

    class Counting(SyntheticProvider):
        def get_bars(self, *a, **k):
            calls.append(1)
            return super().get_bars(*a, **k)

    p = CachedProvider(Counting(seed=1), tmp_path)
    a = universe.get("AAPL")
    start, end = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC")
    first = p.get_bars(a, Timeframe.D1, start, end)
    second = p.get_bars(a, Timeframe.D1, start + pd.Timedelta(days=10), end)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first.loc[second.index], second, check_freq=False)


def test_synthetic_timeframes_agree_on_daily_open_close(universe):
    p = SyntheticProvider(seed=2)
    a = universe.get("BTC-USD")
    s, e = pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-03-03", tz="UTC")
    d = p.get_bars(a, "1d", s, e)
    h = p.get_bars(a, "1h", s, e)
    assert h["open"].iloc[0] == pytest.approx(d["open"].iloc[0])
    assert h["close"].iloc[23] == pytest.approx(d["close"].iloc[0])
