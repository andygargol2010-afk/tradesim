from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from tradesim import AssetUniverse, SimulatorConfig, SyntheticProvider, TradingSimulator
from tradesim.assets.universe import Asset
from tradesim.execution.engine import ExecutionConfig, ExecutionEngine, MarketContext
from tradesim.execution.fees import FeeModel
from tradesim.portfolio.portfolio import Portfolio
from tradesim.types import AssetClass


@pytest.fixture(scope="session")
def universe() -> AssetUniverse:
    return AssetUniverse.default()


@pytest.fixture
def stock() -> Asset:
    return Asset("TEST", AssetClass.EQUITY, liquidity_tier=1)


@pytest.fixture
def coin() -> Asset:
    return Asset("COIN-USD", AssetClass.CRYPTO, liquidity_tier=1, shortable=False)


def make_ctx(asset: Asset, o: float, h: float, l: float, c: float, volume: float = 1e9,
             typical_volume: float = 1e9, sigma: float = 0.0, **kw) -> MarketContext:
    return MarketContext(asset=asset, timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc), open=o, high=h, low=l,
                         close=c, volume=volume, sigma=sigma, sigma_typical=0.01, typical_volume=typical_volume, **kw)


def make_engine(**cfg) -> ExecutionEngine:
    defaults = dict(slippage_model="fixed_bps", limit_touch_fill_prob=0.0)
    defaults.update(cfg)
    return ExecutionEngine(ExecutionConfig(**defaults), FeeModel.preset("zero"), np.random.default_rng(0))


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(100_000)


@pytest.fixture
def sim(universe) -> TradingSimulator:
    s = TradingSimulator(universe, SyntheticProvider(seed=7), SimulatorConfig(initial_cash=10_000, seed=1))
    s.reset(["AAPL", "MSFT", "BTC-USD", "SPY"], "2023-01-01", "2023-04-01")
    return s
