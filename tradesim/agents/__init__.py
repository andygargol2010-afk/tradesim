"""Agent interface, example agents and LLM tools."""

from tradesim.agents.base import (
    HOLD, Action, Agent, CancelOrders, ClosePosition, FunctionAgent, Observation, OrderIntent, PolicyAgent,
    SetExits, TargetWeights,
)
from tradesim.agents.discretionary import ClaudePortfolioManager, DiscretionaryAgent
from tradesim.agents.examples import BuyAndHoldAgent, MovingAverageCrossAgent, RandomAgent, SimpleRSIAgent
from tradesim.agents.investor import TraditionalInvestorAgent
from tradesim.agents.learning import OnlineLinearAgent, WalkForwardRSIAgent
from tradesim.agents.meta import HedgeMetaAgent

__all__ = [
    "HOLD", "Action", "Agent", "BuyAndHoldAgent", "CancelOrders", "ClaudePortfolioManager", "DiscretionaryAgent", "ClosePosition", "FunctionAgent", "HedgeMetaAgent",
    "MovingAverageCrossAgent", "Observation", "OnlineLinearAgent", "OrderIntent", "PolicyAgent", "RandomAgent", "SetExits", "SimpleRSIAgent",
    "TargetWeights", "TraditionalInvestorAgent", "WalkForwardRSIAgent",
]
