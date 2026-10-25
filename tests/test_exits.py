"""Stop-loss / take-profit: reduce-only one-cancels-other exits, in backtests and in live accounts."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator
from tradesim.agents import OrderIntent, SetExits
from tradesim.data.memory import InMemoryProvider
from tradesim.data.replay import ReplayLiveProvider
from tradesim.live import LivePaperTrader

T0 = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)


def five_min_frame(path_points, start=T0 - timedelta(days=40), seed=0):
    """BTC-USD 5-minute bars: a calm random walk around 100, then ``path_points`` [(time, price)] after T0."""
    idx = pd.date_range(start, T0 + timedelta(hours=12), freq="5min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, len(idx))))
    close = close / close[idx < T0][-1] * 100.0  # 100 at T0
    after = idx >= T0
    if path_points:
        xs = [(t - T0).total_seconds() for t, _ in path_points]
        secs = (idx[after] - T0).total_seconds()
        close[after] = np.interp(secs, xs, [p for _, p in path_points])
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                       "low": np.minimum(open_, close) * 0.9998, "close": close, "volume": 1e7}, index=idx)
    return df


def live_trader(df, clock):
    provider = ReplayLiveProvider(InMemoryProvider({"BTC-USD": df}, timeframe="5m"), clock, timeframe="5m")
    return LivePaperTrader(provider, config=SimulatorConfig(initial_cash=10_000, fees="zero"), clock=clock,
                           quote_ttl=0, use_order_book=False, symbols=["BTC-USD"])


def test_backtest_take_profit_cancels_stop():
    sim = TradingSimulator(provider=SyntheticProvider(seed=3), config=SimulatorConfig(initial_cash=10_000),
                           symbols=["BTC-USD"], start="2023-01-01", end="2023-12-31")
    sim.reset(start_offset=50)
    px = sim.get_price("BTC-USD")["mid"]
    sim.apply_action(OrderIntent.buy("BTC-USD", notional=2000, stop_loss=px * 0.97, take_profit=px * 1.04, tag="t"))
    sim.step()  # entry fills; exits attached
    ex = sim.get_exits("BTC-USD")
    assert ex["stop_loss"] == pytest.approx(px * 0.97, rel=1e-3) and ex["take_profit"] == pytest.approx(px * 1.04, rel=1e-3)
    for _ in range(30):
        sim.step()
        if not sim.portfolio.open_positions():
            break
    statuses = {o.tag: o.status.value for o in sim.orders.values()}
    assert sorted(statuses.values()).count("filled") == 2 and "cancelled" in statuses.values()
    assert not sim.open_order_ids  # nothing orphaned


def test_reduce_only_and_replacement():
    sim = TradingSimulator(provider=SyntheticProvider(seed=1), config=SimulatorConfig(initial_cash=10_000),
                           symbols=["SPY"], start="2023-01-01", end="2023-12-31")
    sim.reset(start_offset=60)
    assert sim.set_exits("SPY", 1.0, None) == []  # no position: nothing to protect
    sim.buy("SPY", notional=3000)
    sim.step()
    px = sim.get_price("SPY")["mid"]
    res = sim.apply_action(SetExits("SPY", stop_loss=px * 0.95, take_profit=px * 1.10))
    assert [r["status"] for r in res] == ["new", "new"]
    # protective orders do not block a normal sale of the same shares (reduce-only, outside the projection)
    assert sim.sell("SPY", quantity=float(sim.portfolio.quantity("SPY")) / 2)["status"] == "new"
    # replacing exits (e.g. a higher, trailing stop) cancels the previous pair
    res2 = sim.set_exits("SPY", stop_loss=px * 0.97, take_profit=None)
    assert len(res2) == 1 and len([o for o in sim._open_orders("SPY") if o.reduce_only]) == 1
    bad = sim.set_exits("SPY", stop_loss=px * 1.05)
    assert bad[0]["status"] == "rejected" and "below the market" in bad[0]["reject_reason"]


def test_live_stop_fills_at_the_touch_between_syncs():
    # price drifts to 101, dips to 94 at 03:00 (stop 96), recovers to 103 by 06:00
    df = five_min_frame([(T0, 100), (T0 + timedelta(hours=2), 101), (T0 + timedelta(hours=3), 94),
                         (T0 + timedelta(hours=6), 103)])
    now = {"t": T0 + timedelta(minutes=5)}
    trader = live_trader(df, lambda: now["t"])
    trader.apply_action(OrderIntent.buy("BTC-USD", notional=2000, stop_loss=96, take_profit=110, tag="thesis"))
    trader.sync()  # entry executes now; exits attached
    assert trader.get_exits("BTC-USD") == {"stop_loss": 96.0, "take_profit": 110.0}
    now["t"] = T0 + timedelta(hours=8)  # next sync hours later: the price is back above the stop
    trader.sync()
    assert trader.portfolio.quantity("BTC-USD") == 0
    trade = trader.portfolio.closed_trades[-1]
    assert float(trade.exit_price) == pytest.approx(96, rel=0.004)  # filled at the stop, not at today's 103
    assert T0 + timedelta(hours=2) < trade.closed_at < T0 + timedelta(hours=3, minutes=10)
    tp = next(o for o in trader.orders.values() if o.tag.startswith("take-profit"))
    assert tp.status.value == "cancelled" and "OCO" in tp.reject_reason
    # the replay position is saved with the account
    assert "BTC-USD" in trader.state()["path_checked"]


def test_live_stop_gap_fills_at_the_open_and_target_works():
    # a gap: one bar opens far below the stop -> the stop fills at that open (like a real stop)
    df = five_min_frame([(T0, 100), (T0 + timedelta(hours=2), 100.5)])
    gap_at = T0 + timedelta(hours=2)
    df.loc[df.index >= gap_at, ["open", "high", "low", "close"]] *= 0.90
    now = {"t": T0 + timedelta(minutes=5)}
    trader = live_trader(df, lambda: now["t"])
    trader.apply_action(OrderIntent.buy("BTC-USD", notional=2000, stop_loss=97, take_profit=120))
    trader.sync()
    now["t"] = T0 + timedelta(hours=4)
    trader.sync()
    assert float(trader.portfolio.closed_trades[-1].exit_price) < 92  # the gap, not 97
    # take-profit path
    df2 = five_min_frame([(T0, 100), (T0 + timedelta(hours=3), 108), (T0 + timedelta(hours=5), 101)], seed=1)
    now2 = {"t": T0 + timedelta(minutes=5)}
    t2 = live_trader(df2, lambda: now2["t"])
    t2.apply_action(OrderIntent.buy("BTC-USD", notional=2000, stop_loss=95, take_profit=105))
    t2.sync()
    now2["t"] = T0 + timedelta(hours=6)
    t2.sync()
    tr = t2.portfolio.closed_trades[-1]
    assert float(tr.exit_price) == pytest.approx(105, rel=0.002) and tr.pnl > 0
    # resting orders created after a bar never see that bar (no look-back): a fresh stop below today's
    # earlier low is not triggered retroactively
    now3 = {"t": T0 + timedelta(hours=4)}
    t3 = live_trader(df2, lambda: now3["t"])
    t3.buy("BTC-USD", notional=1000)
    t3.sync()
    t3.set_exits("BTC-USD", stop_loss=102)  # price ~104.5 now (~102.8 in 30 min); it was below 102 before 01:10
    now3["t"] = T0 + timedelta(hours=4, minutes=30)
    t3.sync()
    assert t3.portfolio.quantity("BTC-USD") > Decimal(0)


def test_alpaca_manages_exits_locally_and_sends_a_market_order():
    from test_live_trading import FakeAlpaca

    from tradesim.trading.alpaca import AlpacaTrader

    df = five_min_frame([(T0, 100), (T0 + timedelta(hours=3), 94), (T0 + timedelta(hours=6), 103)])
    now = {"t": T0 + timedelta(minutes=5)}
    clock = lambda: now["t"]  # noqa: E731
    data = ReplayLiveProvider(InMemoryProvider({"BTC-USD": df}, timeframe="5m"), clock, timeframe="5m")
    fake = FakeAlpaca()
    t = AlpacaTrader(paper=True, client=fake, data_provider=data, clock=clock, symbols=["BTC-USD"],
                     config=SimulatorConfig(initial_cash=10_000, fees="zero"))
    t.place_order("BTC-USD", "buy", 10)
    fake.fill("b1", price=100.0)
    t.sync()
    res = t.set_exits("BTC-USD", stop_loss=96, take_profit=110, tag="tesis")
    assert [r["status"] for r in res] == ["new", "new"] and len(fake.orders) == 1  # nothing sent to the broker
    now["t"] = T0 + timedelta(hours=1)
    t.sync()
    assert len(fake.orders) == 1  # not reached yet
    now["t"] = T0 + timedelta(hours=7)
    t.sync()
    assert len(fake.orders) == 2 and fake.orders["b2"].side == "sell" and float(fake.orders["b2"].qty) == 10
    stop = next(o for o in t.orders.values() if o.tag == "stop-loss: tesis")
    assert stop.status.value == "cancelled" and "market order sent" in stop.reject_reason
    assert not [o for o in t._open_orders("BTC-USD") if o.reduce_only]


def test_discretionary_agent_protects_every_position_and_trails(monkeypatch):
    from tradesim.agents.discretionary import DiscretionaryAgent
    from tradesim.practice import run_episode

    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = TradingSimulator(provider=SyntheticProvider(seed=4), config=SimulatorConfig(initial_cash=10_000),
                           symbols=["SPY", "AAPL", "GLD", "BTC-USD"], start="2022-01-01", end="2023-06-30")
    agent = DiscretionaryAgent(min_score=-1.0, max_weight=0.2, every=3)
    seen: dict[str, list[float]] = {}
    orig = env.set_exits

    def spy(symbol, stop_loss=None, take_profit=None, tag=None):
        fresh = env.get_exits(symbol)["stop_loss"] is None  # a new position (or one whose exits fired)
        out = orig(symbol, stop_loss, take_profit, tag)
        if any(o.get("status") == "new" and o.get("order_type") == "stop" for o in out):
            seen[symbol] = [stop_loss] if fresh else seen.get(symbol, []) + [stop_loss]
        return out

    env.set_exits = spy
    run_episode(agent, env, seed=0, episode_length=14)
    held = [p.symbol for p in env.portfolio.open_positions()]
    assert held and seen
    for sym in held:
        ex = env.get_exits(sym)
        price = env.get_price(sym)["mid"]
        assert ex["stop_loss"] is not None and ex["take_profit"] is not None
        assert ex["stop_loss"] < price < ex["take_profit"] or ex["stop_loss"] < price
    for stops in seen.values():
        assert all(b >= a - 1e-9 for a, b in zip(stops, stops[1:]))  # never lowered
    th = next(iter(agent.theses.values()))
    assert th["stop_loss"] and th["take_profit"]
    env.close()


def test_auto_trailing_stop_for_unprotected_positions():
    sim = TradingSimulator(provider=SyntheticProvider(seed=2), config=SimulatorConfig(initial_cash=10_000, auto_stop_atr=2.5),
                           symbols=["SPY", "AAPL"], start="2023-01-01", end="2023-12-31")
    sim.reset(start_offset=60)
    sim.buy("SPY", notional=3000)
    sim.buy("AAPL", notional=3000)
    sim.step()  # fills; SPY gets an automatic stop
    ex = sim.get_exits("SPY")
    assert ex["stop_loss"] is not None and ex["stop_loss"] < sim.get_price("SPY")["mid"] and ex["take_profit"] is None
    sim.set_exits("AAPL", stop_loss=sim.get_price("AAPL")["mid"] * 0.8)  # the agent's own stop is respected
    own = sim.get_exits("AAPL")["stop_loss"]
    stops = [ex["stop_loss"]]
    for _ in range(40):
        sim.step()
        if sim.portfolio.quantity("SPY") == 0:
            break
        stops.append(sim.get_exits("SPY")["stop_loss"])
        assert sim.get_exits("AAPL")["stop_loss"] == own
    assert all(b >= a for a, b in zip(stops, stops[1:]))  # trailing: never lowered
    if sim.portfolio.quantity("SPY") == 0:
        last = sim.portfolio.closed_trades[-1]
        assert last.symbol == "SPY" and sim.orders[last.exit_order_id].tag.endswith("automático")


def test_risk_sizing_groups_and_drawdown_exposure():
    import numpy as np
    import pandas as pd

    from tradesim.agents.risk import correlation_groups, drawdown_exposure, size_positions

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=61, freq="D")
    common = rng.normal(0, 0.04, 61)
    rets = pd.DataFrame({"BTC-USD": common + rng.normal(0, 0.01, 61), "ETH-USD": common + rng.normal(0, 0.01, 61),
                         "DOGE-USD": common * 1.5 + rng.normal(0, 0.02, 61), "KO": rng.normal(0, 0.008, 61),
                         "SPY": rng.normal(0, 0.01, 61)}, index=idx)
    groups = correlation_groups(rets, 0.7)
    assert groups["BTC-USD"] == groups["ETH-USD"] == groups["DOGE-USD"] != groups["KO"]
    w = size_positions(list(rets), rets, target_vol=0.15, max_weight=0.5)
    assert w["KO"] > 4 * w["DOGE-USD"]  # equal risk, not equal money
    assert sum(w[s] for s in ("BTC-USD", "ETH-USD", "DOGE-USD")) <= 0.30 + 1e-9  # crypto / correlated group cap
    assert drawdown_exposure(0, 35) == 1 and drawdown_exposure(17.5, 35) == 0.75 and drawdown_exposure(40, 35) == 0
