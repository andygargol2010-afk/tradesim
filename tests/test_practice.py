from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tradesim import (
    BuyAndHoldAgent, CancelOrders, ClosePosition, FunctionAgent, MovingAverageCrossAgent, OrderIntent, PolicyAgent,
    RandomAgent, SimpleRSIAgent, SimulatorConfig, SyntheticProvider, TargetWeights, TradingSimulator,
    compare_agents, evaluate_agent, run_episode,
)
from tradesim.indicators import rsi, rsi_last

SYMBOLS = ["SPY", "AAPL", "MSFT", "BTC-USD", "ETH-USD"]
TRADE_KEYS = ["final_equity", "total_return_pct", "max_drawdown_pct", "sharpe", "sortino", "win_rate_pct",
              "profit_factor", "num_trades", "avg_trade_pnl", "best_trade", "worst_trade"]


@pytest.fixture
def env(universe) -> TradingSimulator:
    return TradingSimulator(universe, SyntheticProvider(seed=3), SimulatorConfig(initial_cash=10_000),
                            symbols=SYMBOLS, start="2022-01-01", end="2024-06-30")


def test_metrics_have_all_keys_even_without_trades(env):
    env.reset()
    env.step(5)
    m = env.get_metrics()
    for k in TRADE_KEYS:
        assert k in m
    assert m["num_trades"] == 0 and m["win_rate_pct"] is None and m["best_trade"] is None


def test_metrics_from_trades(env):
    r = run_episode(SimpleRSIAgent(), env, seed=1, episode_length=200)
    m = r.metrics
    pnls = r.trades["pnl"].to_numpy()
    assert m["num_trades"] == len(pnls) > 0
    assert m["avg_trade_pnl"] == pytest.approx(pnls.mean())
    assert m["best_trade"] == pytest.approx(pnls.max()) and m["worst_trade"] == pytest.approx(pnls.min())
    assert m["win_rate_pct"] == pytest.approx((pnls > 0).mean() * 100)
    assert m["final_equity"] == pytest.approx(float(env.portfolio.equity))
    assert m["total_return_pct"] == pytest.approx((m["final_equity"] / 10_000 - 1) * 100)


def test_trade_history(env):
    run_episode(RandomAgent(trade_prob=0.5, seed=2), env, seed=0, episode_length=60)
    fills = env.get_trade_history()
    df = env.get_trade_history("fills", as_dataframe=True)
    assert len(fills) == len(df) == len(env.portfolio.fills) > 0
    assert df["quantity"].dtype == float and str(df["timestamp"].dtype).startswith("datetime64")
    closed = env.get_trade_history("closed", symbol=fills[0]["symbol"])
    assert all(t["symbol"] == fills[0]["symbol"] for t in closed)
    with pytest.raises(ValueError):
        env.get_trade_history("nope")


def test_portfolio_shows_fractional_positions(env):
    env.reset()
    env.apply_action(OrderIntent.buy("BTC-USD", quantity="0.12345678"))
    env.step()
    pf = env.get_portfolio()
    pos = {p["symbol"]: p for p in pf["positions"]}
    assert pos["BTC-USD"]["quantity"] == "0.12345678"
    for k in ("avg_price", "unrealized_pnl", "market_value"):
        assert k in pos["BTC-USD"]
    assert "cash" in pf and "equity" in pf


def test_apply_action_forms(env):
    env.reset()
    assert env.apply_action(None) == [] and env.apply_action([]) == [] and env.apply_action("hold") == []
    res = env.apply_action([OrderIntent.buy("SPY", notional=1000), {"symbol": "AAPL", "side": "buy", "quantity": 1.5}])
    assert [r["status"] for r in res] == ["new", "new"]
    env.apply_action(CancelOrders("AAPL"))
    assert len(env.get_open_orders()) == 1
    env.step(3)  # the window starts on a weekend: stock orders wait for Monday's open
    assert env.portfolio.quantity("SPY") > 0 and env.portfolio.quantity("AAPL") == 0
    env.apply_action(ClosePosition("SPY"))
    env.step(3)
    assert env.portfolio.quantity("SPY") == 0
    env.apply_action(TargetWeights({"BTC-USD": 0.3, "ETH-USD": 0.2}))
    env.step()
    obs = env.observe()
    assert obs.weight("BTC-USD") == pytest.approx(0.3, abs=0.02)
    assert env.apply_action({"symbol": "SPY", "side": "sell", "quantity": 5})[0]["status"] == "rejected"
    with pytest.raises(TypeError):
        env.apply_action(3.14)


def test_observation_is_point_in_time(env):
    env.reset()
    env.step(30)
    obs = env.observe()
    s = env.data.series["SPY"]
    i = s.index_le(int(env._calendar[env.step_index]))
    assert obs.prices["SPY"] == s.close[i]
    assert obs.closes("SPY", 5)[-1] == s.close[i]
    hist = obs.history("SPY", 10)
    assert hist["close"].iloc[-1] == s.close[i]
    assert (hist.index + pd.Timedelta(days=1) <= pd.Timestamp(obs.timestamp)).all()
    assert obs.step == 30 and isinstance(obs.positions, dict)


def test_episode_length_and_random_start(env):
    env.reset(seed=5, episode_length=40, random_start=True)
    start_a = env.now
    steps = 0
    while not env.done:
        env.step()
        steps += 1
    assert steps == 40
    env.reset(seed=5, episode_length=40, random_start=True)
    assert env.now == start_a  # same seed -> same start
    env.reset(seed=6, episode_length=40, random_start=True)
    assert env.now != start_a


def test_evaluate_agent_reproducible_and_complete(env):
    rep1 = evaluate_agent(SimpleRSIAgent(), env, n_episodes=4, episode_length=60, seed=10, verbose=False)
    rep2 = evaluate_agent(SimpleRSIAgent(), env, n_episodes=4, episode_length=60, seed=10, verbose=False)
    pd.testing.assert_frame_equal(rep1.episodes, rep2.episodes)
    s = rep1.summary
    for k in ("final_equity", "total_return_pct", "sharpe", "max_drawdown_pct"):
        assert set(s[k]) >= {"mean", "median", "std", "min", "max"}
    assert s["best_return_pct"] == rep1.episodes["total_return_pct"].max()
    assert s["worst_return_pct"] == rep1.episodes["total_return_pct"].min()
    assert len(set(rep1.episodes["start"])) > 1  # different market periods
    assert "Evaluation of" in str(rep1)


def test_buy_and_hold_agent_tracks_benchmark(env):
    r = run_episode(BuyAndHoldAgent(invested_fraction=0.99), env, seed=0, episode_length=100)
    # The benchmark is cost-free and fully invested: only costs, 1 bar of latency and 1% cash differ.
    assert abs(r.metrics["excess_return_pct"]) < 1.0


def test_compare_agents(env):
    table = compare_agents([SimpleRSIAgent(), MovingAverageCrossAgent(), FunctionAgent(lambda o: [], "idle")],
                           env, n_episodes=2, episode_length=60)
    assert list(table.index)[-1] == "idle" and table.loc["idle", "mean_return_%"] == pytest.approx(0.0)


def test_rl_env_with_agents(universe):
    pytest.importorskip("gymnasium")
    from tradesim.env import PaperTradingEnv

    gym_env = PaperTradingEnv(["SPY", "BTC-USD"], "2022-01-01", "2023-06-01", provider=SyntheticProvider(seed=4),
                              universe=universe, episode_length=30)

    class Model:
        def predict(self, obs, deterministic=True):
            return np.array([0.4, 0.4], dtype=np.float32), None

    r = run_episode(PolicyAgent(Model()), gym_env, seed=1)
    assert r.metrics["steps"] == 30 and r.metrics["total_reward"] != 0 and r.metrics["num_fills"] > 0
    rep = evaluate_agent(RandomAgent(action_space=gym_env.action_space), gym_env, n_episodes=3, verbose=False)
    assert len(rep.episodes) == 3
    gym_env.close()


def test_rsi_last_matches_pandas():
    x = np.cumsum(np.random.default_rng(1).normal(size=120)) + 50
    assert rsi_last(x, 14) == pytest.approx(float(rsi(pd.Series(x), 14).iloc[-1]))
    assert np.isnan(rsi_last(x[:10], 14))


def test_summary_and_plots(env, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    run_episode(SimpleRSIAgent(), env, seed=2, episode_length=80)
    env.print_summary()
    env.print_portfolio()
    out = capsys.readouterr().out
    for label in ("Final equity", "Total return", "Max drawdown", "Sharpe ratio", "Sortino ratio", "Win rate",
                  "Profit factor", "Average trade PnL", "Best trade", "Worst trade", "Portfolio @"):
        assert label in out
    env.plot_results(path=tmp_path / "ep.png", show=False)
    rep = evaluate_agent(SimpleRSIAgent(), env, n_episodes=3, episode_length=50, verbose=False)
    rep.plot(tmp_path / "eval.png", show=False)
    assert (tmp_path / "ep.png").stat().st_size > 10_000 and (tmp_path / "eval.png").stat().st_size > 10_000


def test_state_roundtrip_keeps_episode_bounds(env, tmp_path):
    env.reset(seed=3, episode_length=30, random_start=True)
    env.step(10)
    env.save_state(tmp_path / "s.json")
    restored = TradingSimulator.load_state(tmp_path / "s.json", SyntheticProvider(seed=3), env.universe)
    assert restored.steps_remaining == env.steps_remaining == 20
    assert restored.episode_step == env.episode_step == 10
    assert isinstance(restored.observe().cash, float) and restored.portfolio.cash == Decimal(10_000)
