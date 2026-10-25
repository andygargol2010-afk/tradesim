"""Live-operation configuration (``live/config.json``)."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tradesim.agents.base import Agent
from tradesim.persistence import read_json, write_json


@dataclass
class RiskLimits:
    """Guard rails applied by the runner on top of the agent (they matter even more with real money).

    Attributes:
        max_position_pct: pre-trade cap on any single position, % of equity.
        halt_drawdown_pct: stop opening positions once the account is this far below its peak.
        max_orders_per_run: refuse to send more orders than this in one decision (runaway agent guard).
    """

    max_position_pct: float = 30.0
    halt_drawdown_pct: float = 35.0
    max_orders_per_run: int = 20


@dataclass
class AccountSpec:
    """One agent with its own account.

    ``agent`` is a class name exported by ``tradesim.agents`` (``"OnlineLinearAgent"``) or a dotted
    path to your own class (``"my_pkg.my_module:MyAgent"``); ``params`` are its constructor arguments.
    ``broker`` is ``"paper"`` (simulated execution on live quotes), ``"alpaca-paper"`` or ``"alpaca-live"``.
    """

    id: str
    name: str
    agent: str
    params: dict[str, Any] = field(default_factory=dict)
    initial_cash: float = 10_000.0
    fees: str = "alpaca"
    broker: str = "paper"
    description: str = ""
    enabled: bool = True

    def build_agent(self) -> Agent:
        if ":" in self.agent:
            module, cls_name = self.agent.split(":", 1)
            cls = getattr(importlib.import_module(module), cls_name)
        else:
            import tradesim.agents as registry

            cls = getattr(registry, self.agent)
        return cls(**self.params)


@dataclass
class LiveConfig:
    """Everything a live run needs. See ``live/config.json`` for the deployed instance."""

    symbols: list[str]
    accounts: list[AccountSpec]
    timeframe: str = "1d"
    crypto_exchange: str | None = "coinbase"
    stocks_data: str = "auto"  # "auto" (Alpaca if keys are set, else Yahoo) | "alpaca" | "yfinance"
    state_dir: str = "live/accounts"
    runs_log: str = "live/runs.jsonl"
    dashboard_file: str = "dashboard/live.json"
    risk: RiskLimits = field(default_factory=RiskLimits)
    schedule: list[dict[str, str]] = field(default_factory=list)  # informative, shown on the dashboard
    research: dict[str, Any] = field(default_factory=dict)  # see tradesim.research.refresh.ResearchConfig

    def research_config(self):
        """Research settings; research is off unless the config has a ``research`` section."""
        from tradesim.research.refresh import ResearchConfig

        if not self.research:
            return ResearchConfig(enabled=False)
        return ResearchConfig.from_dict(self.research)

    @classmethod
    def load(cls, path: str | Path) -> "LiveConfig":
        d = read_json(path)
        d["accounts"] = [AccountSpec(**a) for a in d["accounts"]]
        d["risk"] = RiskLimits(**d.get("risk", {}))
        return cls(**d)

    def save(self, path: str | Path) -> None:
        write_json(path, asdict(self))

    def resolve(self, base: str | Path) -> "LiveConfig":
        """Make relative paths relative to ``base`` (the repository root)."""
        base = Path(base)
        for attr in ("state_dir", "runs_log", "dashboard_file"):
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                setattr(self, attr, str(base / p))
        if self.research:
            rdir = Path(self.research.get("dir", "live/research"))
            self.research = {**self.research, "dir": str(rdir if rdir.is_absolute() else base / rdir)}
        return self
