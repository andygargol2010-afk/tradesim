import json

import numpy as np
import pandas as pd
import pytest

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator, run_episode
from tradesim.agents.learning import FEATURES, OnlineLinearAgent, RecursiveLeastSquares, WalkForwardRSIAgent, price_features
from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.memory import InMemoryProvider
from tradesim.types import AssetClass


def mean_reverting_frames(n_symbols=4, n=900, phi=-0.3, seed=0):
    """Daily closes whose returns follow r_t = phi * r_{t-1} + noise: a learnable pattern."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    frames = {}
    for k in range(n_symbols):
        r = np.zeros(n)
        eps = rng.normal(0, 0.02, n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + eps[t]
        close = 100 * np.exp(np.cumsum(r))
        frames[f"MR{k}-USD"] = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                                             "close": close, "volume": 1e9}, index=idx)
    return frames


def test_rls_recovers_linear_relation():
    rng = np.random.default_rng(0)
    m = RecursiveLeastSquares(3, forgetting=1.0, ridge=1.0)
    w_true = np.array([0.5, -2.0, 1.0])
    for _ in range(2000):
        x = np.r_[1.0, rng.normal(size=2)]
        m.update(x, float(w_true @ x + rng.normal(0, 0.1)))
    assert np.allclose(m.w, w_true, atol=0.05)
    m2 = RecursiveLeastSquares.from_state(json.loads(json.dumps(m.state_dict())))
    assert np.allclose(m2.w, m.w) and m2.n == m.n


def test_price_features_are_scale_free():
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 100)))
    x1, s1 = price_features(c)
    x2, s2 = price_features(c * 1234.5)
    assert np.allclose(x1, x2) and s1 == pytest.approx(s2)
    assert price_features(c[:30]) is None
    assert len(x1) == len(FEATURES)


def test_online_agent_learns_mean_reversion():
    frames = mean_reverting_frames()
    agent = OnlineLinearAgent(pretrain_bars=0)
    agent.pretrain({s: f["close"] for s, f in frames.items()})
    w = dict(zip(FEATURES, agent.model.w))
    assert w["ret_1"] < -0.15  # learned that yesterday's move tends to reverse
    summary = agent.learning_summary()
    assert summary["rank_ic_mean"] > 0.1 and summary["rank_ic_t"] > 2 and summary["direction_hit_rate_pct"] > 55
    assert agent.has_edge()


def test_online_agent_trades_and_keeps_learning_in_simulator():
    frames = mean_reverting_frames(n=700, seed=3)
    universe = AssetUniverse()
    for s in frames:
        universe.add(Asset(s, AssetClass.CRYPTO, liquidity_tier=1))
    env = TradingSimulator(universe, InMemoryProvider(frames), SimulatorConfig(initial_cash=10_000, fees="zero"),
                           symbols=list(frames), start="2021-01-01", end="2021-11-30")
    agent = OnlineLinearAgent(pretrain_bars=350, min_samples=200)
    res = run_episode(agent, env, seed=0, episode_length=150)
    assert res.metrics["num_fills"] > 0
    n_after = agent.model.n
    assert n_after > agent.pretrained_on  # it kept learning online during the episode
    # state round trip: a restored agent predicts exactly like the original
    clone = OnlineLinearAgent()
    clone.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    x = np.ones(len(FEATURES))
    assert clone.model.predict(x) == pytest.approx(agent.model.predict(x))
    assert clone.pending.keys() == agent.pending.keys()


def test_walk_forward_rsi_retunes_and_persists():
    env = TradingSimulator(provider=SyntheticProvider(seed=2), config=SimulatorConfig(initial_cash=10_000),
                           symbols=["SPY", "AAPL", "BTC-USD"], start="2021-01-01", end="2023-12-31")
    agent = WalkForwardRSIAgent(grid=[(7, 30.0, 70.0), (14, 30.0, 70.0)], retune_days=10, lookback_bars=150)
    run_episode(agent, env, seed=0, episode_length=25)
    assert len(agent.tunes) >= 2 and agent.last_tune is not None
    rec = agent.tunes[0]
    assert {"previous", "best", "best_sharpe", "switched", "top"} <= set(rec)
    restored = WalkForwardRSIAgent()
    restored.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    assert restored.params == agent.params and restored._inner.period == agent.params[0]
    assert "params" in agent.explain()


def test_no_edge_gate_and_horizon_without_lookahead():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=700, freq="D", tz="UTC")
    frames = {f"RW{k}-USD": pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 700))), index=idx) for k in range(6)}
    agent = OnlineLinearAgent(pretrain_bars=0, horizon=5, ic_gate=True)
    agent.pretrain(frames)
    st = agent.ic_stats()
    assert abs(st["rank_ic_mean"]) < 0.1 and not agent.has_edge()  # random walks: no edge, gate stays shut
    mr = OnlineLinearAgent(pretrain_bars=0, horizon=1, ic_gate=True)
    mr.pretrain({s: f["close"] for s, f in mean_reverting_frames(n_symbols=6).items()})
    assert mr.has_edge()
