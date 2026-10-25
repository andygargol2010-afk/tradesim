"""The agent contract: implement ``act(observation) -> action``; the simulator does the rest.

Two kinds of environment are supported by :func:`tradesim.practice.run_episode`:

* :class:`~tradesim.simulator.TradingSimulator` (rule-based and LLM agents): the
  agent receives an :class:`Observation` and returns an action (see below).
* :class:`~tradesim.env.PaperTradingEnv` (reinforcement learning): the agent
  receives the Gymnasium observation (``np.ndarray``) and returns an element of
  ``env.action_space``.

Actions accepted by ``TradingSimulator.apply_action``:

* ``None`` / ``[]`` / :data:`HOLD`: do nothing;
* :class:`OrderIntent` or a dict with ``place_order`` arguments
  (``{"symbol": "BTC-USD", "side": "buy", "quantity": 0.003}``);
* :class:`TargetWeights` or ``{"weights": {"SPY": 0.5, "BTC-USD": 0.1}}``: rebalance;
* :class:`ClosePosition`, :class:`CancelOrders`;
* a list mixing any of the above.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from tradesim.simulator import BaseSimulator


# --------------------------------------------------------------------------- actions
@dataclass(frozen=True)
class OrderIntent:
    """One order. Size with exactly one of ``quantity`` (fractions OK), ``notional`` (USD)
    or ``percent_of_equity`` (0-100). Other fields as in ``place_order``."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float | str | Decimal | None = None
    notional: float | None = None
    percent_of_equity: float | None = None
    order_type: str = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "gtc"
    tag: str | None = None

    @classmethod
    def buy(cls, symbol: str, **kw: Any) -> "OrderIntent":
        return cls(symbol, "buy", **kw)

    @classmethod
    def sell(cls, symbol: str, **kw: Any) -> "OrderIntent":
        return cls(symbol, "sell", **kw)

    def kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class TargetWeights:
    """Rebalance to these portfolio weights (fractions of equity; symbols not listed go to 0)."""

    weights: dict[str, float]


@dataclass(frozen=True)
class ClosePosition:
    """Flatten ``symbol`` with a market order (cancels its open orders first)."""

    symbol: str


@dataclass(frozen=True)
class CancelOrders:
    """Cancel open orders (all, or only ``symbol``'s)."""

    symbol: str | None = None


HOLD: list = []

Action = Union[None, OrderIntent, TargetWeights, ClosePosition, CancelOrders, dict, list]


# ----------------------------------------------------------------------- observation
@dataclass
class Observation:
    """What a non-RL agent sees at a decision point (the close of the current bar).

    Everything here is known at decision time; ``history()`` never returns future bars.

    Attributes:
        timestamp: decision time (UTC).
        step: bars elapsed in this episode; ``steps_remaining`` bars are left.
        symbols: the tradable symbols of the episode.
        prices: last close per symbol (``nan`` before a symbol's first bar).
        cash / equity: account values in USD.
        positions: signed quantity per symbol with an open position (exact ``Decimal``).
        open_orders: number of working orders.
    """

    timestamp: datetime
    step: int
    steps_remaining: int
    symbols: list[str]
    prices: dict[str, float]
    cash: float
    equity: float
    positions: dict[str, Decimal]
    open_orders: int
    _sim: "BaseSimulator | None" = field(default=None, repr=False, compare=False)

    def position(self, symbol: str) -> Decimal:
        return self.positions.get(symbol, Decimal(0))

    def weight(self, symbol: str) -> float:
        price = self.prices.get(symbol, float("nan"))
        if not self.equity or price != price:
            return 0.0
        return float(self.position(symbol)) * price / self.equity

    def history(self, symbol: str, limit: int = 100, timeframe: str | None = None,
                indicators: bool = False) -> pd.DataFrame:
        """Completed OHLCV bars up to now (optionally with SMA/EMA/RSI/ATR/MACD/Bollinger)."""
        return self._require_sim().get_historical_data(symbol, timeframe, limit, indicators)

    def closes(self, symbol: str, limit: int = 100) -> np.ndarray:
        """Last ``limit`` closes as a numpy array (fast path for indicator math; no future bars)."""
        return self._require_sim().recent_closes(symbol, limit)

    def quote(self, symbol: str) -> dict[str, Any]:
        """Bid/ask/mid/spread. Buys pay the ask, sells receive the bid."""
        return self._require_sim().get_price(symbol)

    def portfolio(self) -> dict[str, Any]:
        """Full account state (positions with average price and unrealized PnL)."""
        return self._require_sim().get_portfolio()

    @property
    def sim(self) -> "BaseSimulator":
        """The simulator itself, e.g. for LLM agents that act through ``llm_tools.dispatch``."""
        return self._require_sim()

    def to_text(self) -> str:
        """Compact description, convenient as an LLM prompt."""
        pos = ", ".join(f"{s} {q}" for s, q in self.positions.items()) or "none"
        px = ", ".join(f"{s} {p:.6g}" for s, p in list(self.prices.items())[:30])
        return (f"{self.timestamp:%Y-%m-%d %H:%M} UTC | step {self.step} ({self.steps_remaining} left) | "
                f"equity {self.equity:,.2f} | cash {self.cash:,.2f} | positions: {pos} | "
                f"open orders {self.open_orders} | prices: {px}")

    def _require_sim(self) -> "BaseSimulator":
        if self._sim is None:
            raise RuntimeError("Observation is detached from its simulator")
        return self._sim


# ---------------------------------------------------------------------------- agents
class Agent(abc.ABC):
    """Base class for agents. Only :meth:`act` is required.

    ``act`` receives an :class:`Observation` (``TradingSimulator``) or a numpy array
    (``PaperTradingEnv``) and returns an action for that environment.
    """

    name: str = ""

    @abc.abstractmethod
    def act(self, observation: Any) -> Any:
        """Decide what to do at this step."""

    def reset(self) -> None:
        """Called at the start of each episode (clear any per-episode state)."""

    def __str__(self) -> str:
        return self.name or type(self).__name__


class FunctionAgent(Agent):
    """Wrap a plain function ``f(observation) -> action`` as an agent."""

    def __init__(self, fn, name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "function_agent")

    def act(self, observation: Any) -> Any:
        return self.fn(observation)


class PolicyAgent(Agent):
    """Adapter for RL policies with a ``predict(obs, deterministic=...)`` method
    (Stable-Baselines3 and similar), for use with ``PaperTradingEnv``."""

    def __init__(self, model: Any, deterministic: bool = True, name: str | None = None):
        self.model = model
        self.deterministic = deterministic
        self.name = name or type(model).__name__

    def act(self, observation: Any) -> Any:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action
