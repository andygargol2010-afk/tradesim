import json

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from tradesim import SyntheticProvider  # noqa: E402
from tradesim.agents.llm_tools import TOOLS, dispatch  # noqa: E402
from tradesim.env import PaperTradingEnv  # noqa: E402


@pytest.fixture(scope="module")
def env(universe):
    e = PaperTradingEnv(["AAPL", "BTC-USD", "SPY"], "2022-01-01", "2023-06-01", provider=SyntheticProvider(seed=4),
                        universe=universe, episode_length=60)
    yield e
    e.close()


def test_check_env(env):
    check_env(env.unwrapped, skip_render_check=True)


def test_episode_runs_and_is_reproducible(env):
    def rollout(seed):
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(60):
            obs, r, term, trunc, info = env.step(np.array([0.3, 0.3, 0.3], dtype=np.float32))
            rewards.append(r)
            if term or trunc:
                break
        return obs, rewards, info
    o1, r1, i1 = rollout(11)
    o2, r2, _ = rollout(11)
    assert np.allclose(o1, o2) and np.allclose(r1, r2)
    assert trunc_ok(i1)


def trunc_ok(info):
    return info["step"] == 60 and info["equity"] > 0


def test_order_dict_actions(env):
    env.reset(seed=1)
    _, _, _, _, info = env.step({"orders": [{"symbol": "SPY", "side": "buy", "notional": 1000},
                                            {"symbol": "SPY", "side": "sell", "quantity": 1e6}]})
    assert len(info["rejected_orders"]) == 1


def test_tool_schemas_and_dispatch(sim):
    names = {t["name"] for t in TOOLS}
    assert {"get_price", "get_portfolio", "place_order", "get_historical_data", "get_available_assets"} <= names
    for t in TOOLS:
        assert t["input_schema"]["type"] == "object"
    out = json.loads(dispatch(sim, "place_order", {"symbol": "AAPL", "side": "buy", "quantity": 1.75}))
    assert out["status"] == "new"
    bars = json.loads(dispatch(sim, "get_historical_data", {"symbol": "AAPL", "limit": 5, "indicators": True}))
    assert len(bars["bars"]) == 5 and "rsi_14" in bars["bars"][0]
    assert "error" in json.loads(dispatch(sim, "get_price", {"symbol": "NOPE"}))
