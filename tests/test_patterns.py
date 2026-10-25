import numpy as np
import pandas as pd
import pytest

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator
from tradesim.agents.llm_tools import dispatch
from tradesim.patterns import (PATTERN_FEATURES, candlestick_patterns, detect_patterns, find_pivots,
                               pattern_features)


def path_frame(points, seed=0, noise=0.002, wick=0.004):
    """OHLC bars following a piecewise-linear path through ``points`` = [(bar, price), ...]."""
    xs, ys = zip(*points)
    close = np.interp(np.arange(xs[-1] + 1), xs, ys)
    rng = np.random.default_rng(seed)
    close = close * (1 + rng.normal(0, noise, len(close)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def names(report):
    return {p.name: p for p in report.patterns}


def test_double_bottom_confirmed_with_target():
    df = path_frame([(0, 120), (40, 100), (55, 112), (70, 100.5), (88, 115)])
    rep = detect_patterns(df, "X")
    p = names(rep)["double_bottom"]
    assert p.direction == "bullish" and p.status == "confirmed"
    assert p.levels["target"] > p.levels["neckline"] > p.levels["invalidation"]
    assert rep.score() > 0


def test_head_and_shoulders_confirmed():
    df = path_frame([(0, 90), (30, 110), (45, 100), (65, 122), (80, 100), (100, 110), (115, 96)])
    p = names(detect_patterns(df, "X"))["head_and_shoulders"]
    assert p.direction == "bearish" and p.status == "confirmed"
    assert p.levels["target"] < p.levels["neckline"]


def test_ascending_triangle_and_breakout():
    pts, t = [(0, 92)], 0
    lows = [92, 95, 97.5, 99.5]
    for lo in lows:
        pts += [(t + 8, 105), (t + 16, lo)]
        t += 16
    forming = path_frame(pts + [(t + 6, 103)], noise=0.0005)
    assert names(detect_patterns(forming, "X"))["ascending_triangle"].status == "forming"
    broke = path_frame(pts + [(t + 8, 109)], noise=0.0005)
    rep = detect_patterns(broke, "X")
    assert any(p.direction == "bullish" and p.status == "confirmed" for p in rep.patterns)


def test_candlesticks():
    n = 30
    o = np.linspace(110, 100, n)  # downtrend
    c = o - 0.5
    h, l = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    # bullish engulfing on the last bar
    o[-1], c[-1] = c[-2] - 0.1, o[-2] + 0.4
    h[-1], l[-1] = c[-1] + 0.05, o[-1] - 0.05
    got = {p.name for p in candlestick_patterns(o, h, l, c)}
    assert "bullish_engulfing" in got
    # hammer: small body at the top, long lower wick, after a decline
    o2, c2 = o.copy(), c.copy()
    o2[-1], c2[-1] = 100.0, 100.2
    h2, l2 = h.copy(), l.copy()
    h2[-1], l2[-1] = 100.25, 98.5
    assert "hammer" in {p.name for p in candlestick_patterns(o2, h2, l2, c2)}


def test_support_resistance_levels_and_pivots():
    df = path_frame([(0, 100), (10, 110), (20, 100), (30, 110), (40, 100), (50, 110), (60, 104)], noise=0.0005)
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    ph, pl = find_pivots(h, l, 4)
    assert len(ph) >= 2 and len(pl) >= 2
    rep = detect_patterns(df, "X")
    assert rep.resistance and abs(rep.resistance[0]["price"] - 110) < 1.5 and rep.resistance[0]["touches"] >= 2
    assert rep.support and abs(rep.support[0]["price"] - 100) < 1.5


def test_causal_and_serialisable():
    df = path_frame([(0, 120), (40, 100), (55, 112), (70, 100.5), (88, 115), (120, 90)])
    a = detect_patterns(df.iloc[:89], "X").to_dict()
    b = detect_patterns(df, "X")
    assert a["as_of"] < b.as_of  # earlier report only used earlier bars
    assert detect_patterns(df.iloc[:89], "X").to_dict() == a  # deterministic
    assert isinstance(b.summary(), str) and "X" in b.summary()


def test_pattern_features_bounded():
    df = path_frame([(0, 100), (60, 130)], noise=0.001)
    f = pattern_features(df["close"].to_numpy(), df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy())
    assert len(f) == len(PATTERN_FEATURES) and np.all(np.abs(f) <= 1)
    assert f[0] == 1.0 and f[1] > 0.8  # breaking out, clean uptrend


def test_simulator_observation_and_llm_tool():
    sim = TradingSimulator(provider=SyntheticProvider(seed=1), config=SimulatorConfig(initial_cash=10_000),
                           symbols=["SPY", "BTC-USD"], start="2022-01-01", end="2023-12-31")
    sim.reset(seed=0, start_offset=300)
    obs = sim.observe()
    rep = obs.patterns("SPY")
    assert rep.symbol == "SPY" and pd.Timestamp(rep.as_of) < pd.Timestamp(obs.timestamp) + pd.Timedelta(days=1)
    assert -1 <= rep.score() <= 1
    out = dispatch(sim, "detect_patterns", {"symbol": "BTC-USD"})
    assert '"patterns"' in out and '"support"' in out
    sim.close()


@pytest.mark.parametrize("seed", range(5))
def test_no_crash_on_random_walks(seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c,
                       "volume": rng.uniform(1e5, 1e6, 400)},
                      index=pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC"))
    rep = detect_patterns(df, "R")
    assert all(0 <= p.strength <= 1 for p in rep.patterns)
    assert len(rep.support) <= 3 and len(rep.resistance) <= 3
