"""Second-level model: combines the *signals* of the other agents (unlike :class:`HedgeMetaAgent`, which splits
capital among them).

Per asset and day the features are the learner's prediction, the analyst's technical / news / trend scores, the
RSI's distance from 50 and the moving-average trend flag, each also multiplied by a "stress regime" flag (the
learner's most volatile market regime), so the model can learn that a signal works in calm markets and fails in
stressed ones. Target: the asset's return over ``horizon_days`` relative to the others, in volatility units. Online
regression (RLS); it only trades while its own out-of-sample rank IC is significant.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from tradesim.agents.base import Agent, ClosePosition, Observation, OrderIntent
from tradesim.agents.learning import RecursiveLeastSquares, _ic_summary, rank_ic_by_day
from tradesim.data.base import DataUnavailableError

BASE = ["learner_pred", "analyst_tech", "analyst_news", "analyst_trend", "rsi_dev", "ma_up"]
STACK_FEATURES = ["bias"] + BASE + [f"{f}_x_stress" for f in BASE] + ["stress"]
DEFAULT_SOURCES = {"learner": "aprendiz", "analyst": "analista", "rsi": "rsi-adaptativo", "ma": "cruce-medias"}


class SignalStackingAgent(Agent):
    name = "SignalStackingAgent"

    def __init__(self, sources: dict[str, str] | None = None, horizon_days: int = 7, top_k: int = 5,
                 min_samples: int = 300, ic_min_days: int = 20, ic_min_t: float = 2.0, target_vol: float = 0.15,
                 max_weight: float = 0.2, rebalance_band: float = 0.03, forgetting: float = 0.999):
        self.sources = {**DEFAULT_SOURCES, **(sources or {})}
        self.horizon_days, self.top_k, self.min_samples = horizon_days, top_k, min_samples
        self.ic_min_days, self.ic_min_t = ic_min_days, ic_min_t
        self.target_vol, self.max_weight, self.rebalance_band = target_vol, max_weight, rebalance_band
        self.model = RecursiveLeastSquares(len(STACK_FEATURES), forgetting, ridge=20.0)
        self.pending: list[dict[str, Any]] = []
        self.history: list[list[Any]] = []  # [date, prediction at the time, relative outcome]
        self._peers: dict[str, dict[str, Any]] = {}
        self._last: dict[str, Any] = {}

    def attach_peers(self, peers: dict[str, dict[str, Any]]) -> None:
        self._peers = peers

    def _explain(self, role: str) -> dict[str, Any]:
        return (self._peers.get(self.sources.get(role, "")) or {}).get("explain") or {}

    def features(self, symbols: list[str]) -> dict[str, np.ndarray]:
        learner, analyst = self._explain("learner"), self._explain("analyst")
        rsi, ma = self._explain("rsi").get("rsi") or {}, set(self._explain("ma").get("uptrend") or [])
        preds, scores = learner.get("predictions") or {}, analyst.get("scores") or {}
        stress = 1.0 if learner.get("regime") == 2 else 0.0
        out = {}
        for s in symbols:
            sc = scores.get(s) or [0.0, 0.0, 0.0]
            base = [float(preds.get(s, 0.0)), *map(float, sc[:3]),
                    (50.0 - float(rsi[s])) / 50.0 if s in rsi else 0.0, 1.0 if s in ma else 0.0]
            out[s] = np.clip(np.r_[1.0, base, np.array(base) * stress, stress], -5, 5)
        return out

    def _sigma(self, obs: Observation, sym: str) -> float | None:
        try:
            c = obs.closes(sym, 21)
        except (DataUnavailableError, KeyError, ValueError):
            return None
        if len(c) < 10 or not (c > 0).all():
            return None
        sd = float(np.std(np.diff(np.log(c)), ddof=1))
        return sd if sd > 0 else None

    def _learn(self, now: datetime, prices: dict[str, float]) -> None:
        due: dict[str, list] = {}
        waiting = []
        for p in self.pending:
            age = (now - datetime.fromisoformat(p["date"])).total_seconds() / 86400
            px = prices.get(p["sym"])
            if age >= self.horizon_days and px and px > 0:
                y = math.log(px / p["price"]) / (p["sigma"] * math.sqrt(self.horizon_days * 5 / 7))
                due.setdefault(p["date"][:10], []).append((p, float(np.clip(y, -6, 6))))
            elif age < 3 * self.horizon_days:
                waiting.append(p)
        self.pending = waiting
        for day, group in sorted(due.items()):
            mu = float(np.mean([y for _, y in group])) if len(group) >= 5 else 0.0
            for p, y in group:
                self.model.update(np.array(p["x"]), y - mu)
                self.history.append([day, round(p["pred"], 4), round(y - mu, 4)])
        del self.history[:-8000]

    def ic_stats(self) -> dict[str, Any]:
        return _ic_summary(rank_ic_by_day([tuple(r) for r in self.history], every=5, window=60))

    def has_edge(self) -> bool:
        st = self.ic_stats()
        return (self.model.n >= self.min_samples and st["days"] >= self.ic_min_days
                and (st["rank_ic_mean"] or 0) > 0 and (st["rank_ic_t"] or 0) >= self.ic_min_t)

    def act(self, obs: Observation) -> list[Any]:
        symbols = [s for s in obs.symbols if obs.prices.get(s, 0) > 0]
        prices = {s: float(obs.prices[s]) for s in symbols}
        self._learn(obs.timestamp, prices)
        if not self._peers:
            self._last = {"signals": ["sin señales de otros agentes (solo funciona en la operativa en vivo)"]}
            return []
        feats = self.features(symbols)
        preds = {s: self.model.predict(x) for s, x in feats.items()}
        if not any(p["date"][:10] == obs.timestamp.date().isoformat() for p in self.pending):
            for s, x in feats.items():
                sg = self._sigma(obs, s)
                if sg:
                    self.pending.append({"date": obs.timestamp.isoformat(), "sym": s, "x": [round(v, 5) for v in x],
                                         "price": prices[s], "sigma": sg, "pred": preds[s]})
        w = dict(zip(STACK_FEATURES, (round(float(v), 4) for v in self.model.w)))
        self._last = {"predictions": {s: round(v, 4) for s, v in sorted(preds.items(), key=lambda kv: -kv[1])[:15]},
                      "weights": w, "samples": self.model.n, "ic": self.ic_stats()}
        if obs.open_orders or obs.tradable is not None:
            return []
        if not self.has_edge():
            self._last["signals"] = [f"sin ventaja demostrada todavía (IC {self.ic_stats()['rank_ic_mean'] or 0:+.3f}, "
                                     f"{self.model.n} ejemplos): efectivo"]
            return [ClosePosition(s) for s, q in obs.positions.items() if q > 0]
        from tradesim.agents.risk import returns_frame, size_positions

        chosen = [s for s in sorted(preds, key=lambda s: -preds[s]) if preds[s] > 0][: self.top_k]
        targets = size_positions(chosen, returns_frame(obs, chosen), getattr(obs._sim, "universe", None),
                                 target_vol=self.target_vol, max_weight=self.max_weight,
                                 exposure=getattr(obs, "exposure_cap", 1.0))
        actions: list[Any] = [ClosePosition(s) for s, q in obs.positions.items() if q > 0 and s not in targets]
        for s, t in targets.items():
            diff = t - obs.weight(s)
            if abs(diff) < self.rebalance_band:
                continue
            if diff > 0:
                actions.append(OrderIntent.buy(s, notional=round(diff * obs.equity, 2), tag=f"stacking {preds[s]:+.3f}σ"))
            else:
                actions.append(OrderIntent.sell(s, quantity=min(float(obs.position(s)), -diff * obs.equity / prices[s]),
                                                tag=f"stacking {preds[s]:+.3f}σ"))
        self._last["signals"] = [f"{s}: {preds[s]:+.3f}σ → {targets[s]:.0%}" for s in targets] or ["mantener"]
        return actions

    def explain(self) -> dict[str, Any] | None:
        return self._last or None

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        w = self._last.get("weights") or {}
        top = sorted(((k, v) for k, v in w.items() if k != "bias"), key=lambda kv: -abs(kv[1]))[:3]
        por = ("Combino las señales de los demás agentes (predicción del aprendiz, técnico/noticias/tendencia del "
               "analista, RSI y medias) con pesos que aprendo de los resultados. Las que más pesan ahora: "
               + ", ".join(f"{k} {v:+.2f}" for k, v in top) + ". "
               + ("; ".join(self._last.get("signals") or [])))
        segun = (f"Con el mercado {d.tag(situation)}: cada señal tiene además un peso propio para los regímenes de "
                 "tensión, así aprendo cuándo conviene cada una.")
        return d.entry(d.did(orders), por, segun)

    def learning_summary(self) -> dict[str, Any] | None:
        st = self.ic_stats()
        return {"method": "Stacking: online regression over the other agents' signals (with stress interactions)",
                "samples_learned": self.model.n, "trading_enabled": self.has_edge(),
                "rank_ic_mean": st["rank_ic_mean"], "rank_ic_t": st["rank_ic_t"], "rank_ic_days": st["days"],
                "weights": dict(zip(STACK_FEATURES, (round(float(v), 4) for v in self.model.w)))}

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "pending": self.pending, "history": self.history}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state and len(state.get("model", {}).get("w", [])) == len(STACK_FEATURES):
            self.model = RecursiveLeastSquares.from_state(state["model"])
            self.pending, self.history = state.get("pending", []), state.get("history", [])
