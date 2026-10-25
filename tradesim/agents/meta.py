"""Meta-agent: allocates its capital among the other agents with the Hedge algorithm (multiplicative weights).

Every decision it observes each agent's return since the last one and updates ``w_i <- w_i * exp(eta * r_i)``, then
normalises. Hedge's regret against the best agent in hindsight grows at most like ``sqrt(T log N)``: its average
return converges to the best agent's, the only kind of learning here with a formal guarantee. Its portfolio is the
weighted sum of the agents' portfolios (cash included), traded only where the difference is worth it.

Live, the runner hands it the other accounts (:meth:`attach_peers`); offline, :func:`hedge_weights` replays it on
equity curves.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tradesim.agents.base import Agent, Observation, OrderIntent


def hedge_weights(equity: pd.DataFrame, eta: float = 2.5) -> pd.DataFrame:
    """Hedge weights over time from the agents' equity curves (one column per agent, one row per decision);
    row ``t`` holds the weights used from ``t`` on (computed with returns up to ``t``)."""
    r = np.log(equity / equity.shift(1)).fillna(0.0).to_numpy()
    logw = np.zeros(equity.shape[1])
    out = []
    for row in r:
        logw += eta * row
        w = np.exp(logw - logw.max())
        out.append(w / w.sum())
    return pd.DataFrame(out, index=equity.index, columns=equity.columns)


class HedgeMetaAgent(Agent):
    """Hedge over the other live agents.

    Args:
        peers: account ids to allocate among (default: every other account the runner provides).
        eta: learning rate. ``sqrt(8 ln N / T) / range`` with N = 6 agents, T = 252 days and daily returns within
            about +/-5 % gives ~2.5: a 1-point daily edge multiplies an agent's relative weight by ~1.025 per day.
        min_trade_pct: per-asset differences smaller than this fraction of equity are not traded.
    """

    name = "HedgeMetaAgent"

    def __init__(self, peers: Sequence[str] | None = None, eta: float = 2.5, min_trade_pct: float = 0.02):
        self.peers = list(peers) if peers else None
        self.eta, self.min_trade_pct = eta, min_trade_pct
        self.log_w: dict[str, float] = {}
        self.last_equity: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self._peers: dict[str, dict[str, Any]] = {}
        self._last: dict[str, Any] = {}

    def attach_peers(self, peers: dict[str, dict[str, Any]]) -> None:
        """``{account_id: {"name", "equity", "weights": {symbol: fraction of its equity}}}``."""
        self._peers = {k: v for k, v in peers.items() if self.peers is None or k in self.peers}

    def weights(self) -> dict[str, float]:
        if not self.log_w:
            return {}
        top = max(self.log_w.values())
        w = {k: math.exp(v - top) for k, v in self.log_w.items()}
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()}

    def update(self, equity: dict[str, float]) -> dict[str, float]:
        """One Hedge step with each agent's log return since the previous call; new agents start at the mean."""
        returns = {}
        mean = float(np.mean(list(self.log_w.values()))) if self.log_w else 0.0
        for k, eq in equity.items():
            if k not in self.log_w:
                self.log_w[k] = mean
            prev = self.last_equity.get(k)
            if prev and prev > 0 and eq > 0:
                returns[k] = math.log(eq / prev)
                self.log_w[k] += self.eta * returns[k]
            self.last_equity[k] = eq
        for k in [k for k in self.log_w if k not in equity]:  # agent removed
            del self.log_w[k]
            self.last_equity.pop(k, None)
        return returns

    def act(self, obs: Observation) -> list[Any]:
        if not self._peers:
            self._last = {"signals": ["sin agentes que combinar (solo funciona en la operativa en vivo)"]}
            return []
        returns = self.update({k: float(v["equity"]) for k, v in self._peers.items() if v.get("equity")})
        w = self.weights()
        target: dict[str, float] = {}
        for k, wk in w.items():
            for sym, x in (self._peers[k].get("weights") or {}).items():
                target[sym] = target.get(sym, 0.0) + wk * float(x)
        self.history.append({"date": obs.timestamp.isoformat(), "weights": {k: round(v, 4) for k, v in w.items()},
                             "returns": {k: round(v, 5) for k, v in returns.items()}})
        del self.history[:-60]
        self._last = {"weights": {k: round(v, 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1])},
                      "returns": {k: round(v * 100, 3) for k, v in returns.items()},
                      "target": {s: round(v, 4) for s, v in sorted(target.items(), key=lambda kv: -kv[1]) if v > 0.001}}
        if obs.open_orders:
            return []
        eq = float(obs.equity)
        sells, buys = [], []
        for sym in sorted(set(target) | {s for s, q in obs.positions.items() if q > 0}):
            px = obs.prices.get(sym, float("nan"))
            if not px > 0:
                continue
            diff = target.get(sym, 0.0) - obs.weight(sym)
            if abs(diff) < self.min_trade_pct and target.get(sym, 0.0) > 0:
                continue
            if diff < 0:
                qty = float(obs.position(sym)) if target.get(sym, 0.0) <= 0.001 else min(float(obs.position(sym)),
                                                                                         -diff * eq / px)
                if qty > 0:
                    sells.append(OrderIntent.sell(sym, quantity=qty, tag="Hedge: menos peso"))
            elif diff * eq >= 1.0:
                buys.append(OrderIntent.buy(sym, notional=round(diff * eq, 2), tag="Hedge: más peso"))
        return sells + buys

    def explain(self) -> dict[str, Any] | None:
        return self._last or None

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        w = self._last.get("weights") or {}
        if not w:
            return d.entry(d.did(orders), "No tenía agentes que combinar.", "")
        names = {k: v.get("name", k) for k, v in self._peers.items()}
        top = ", ".join(f"{names.get(k, k)} {v:.0%}" for k, v in list(w.items())[:4])
        best = max(self._last.get("returns") or {"": 0}, key=lambda k: (self._last.get("returns") or {}).get(k, 0))
        por = (f"Reparto mi capital entre los agentes con pesos multiplicativos (Hedge): cada día gana peso quien "
               f"más rinde. Pesos hoy: {top}." + (f" Ayer rindió más {names.get(best, best)}." if best else ""))
        segun = (f"Con el mercado {d.tag(situation)} no elijo activos: copio la cartera combinada de los agentes, "
                 "así que me adapto a la situación a través de los que mejor la están leyendo.")
        return d.entry(d.did(orders), por, segun)

    def learning_summary(self) -> dict[str, Any] | None:
        return {"method": "Hedge (multiplicative weights) over the other agents; regret O(sqrt(T log N))",
                "eta": self.eta, "weights": self.weights(), "recent": self.history[-5:]}

    def state_dict(self) -> dict[str, Any]:
        return {"log_w": self.log_w, "last_equity": self.last_equity, "history": self.history}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            self.log_w = dict(state.get("log_w", {}))
            self.last_equity = dict(state.get("last_equity", {}))
            self.history = list(state.get("history", []))
