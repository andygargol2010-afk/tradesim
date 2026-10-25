"""tradesim: a high-fidelity trading simulator for training AI trading agents."""

from tradesim.assets.universe import Asset, AssetUniverse
from tradesim.data.base import DataProvider, DataUnavailableError, RoutingProvider
from tradesim.data.cache import CachedProvider
from tradesim.data.synthetic import SyntheticProvider
from tradesim.data.timeframes import Timeframe
from tradesim.execution.engine import ExecutionConfig
from tradesim.execution.fees import FeeModel, FeeSchedule
from tradesim.market.conditions import Scenario
from tradesim.market.liquidity import SpreadModel
from tradesim.portfolio.portfolio import MarginConfig, Portfolio
from tradesim.simulator import SimulatorConfig, TradingSimulator
from tradesim.types import AssetClass, OrderStatus, OrderType, Side, TimeInForce

__version__ = "0.1.0"

__all__ = [
    "Asset", "AssetUniverse", "AssetClass", "CachedProvider", "DataProvider", "DataUnavailableError",
    "ExecutionConfig", "FeeModel", "FeeSchedule", "MarginConfig", "OrderStatus", "OrderType", "Portfolio",
    "RoutingProvider", "Scenario", "Side", "SimulatorConfig", "SpreadModel", "SyntheticProvider", "TimeInForce",
    "Timeframe", "TradingSimulator",
]
