"""Simple example agents to exercise the whole system (not meant to be profitable).

All of them implement only ``act(observation) -> action`` and work with any symbol set
(fractional sizing, 1000+ symbols). They read prices from the observation and history via
``observation.history()``, which never returns future bars.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from tradesim.agents.base import Agent, ClosePosition, Observation, OrderIntent, TargetWeights
from tradesim.indicators import rsi_last


class SimpleRSIAgent(Agent):
    """Mean reversion on RSI: buy when RSI < ``oversold``, sell when RSI > ``overbought``.

    Each entry invests ``position_pct`` % of equity (fractional quantities); at most
    ``max_positions`` open positions at once.

    Args:
        symbols: symbols to trade (default: every symbol of the episode).
        period: RSI look-back in bars.
        oversold / overbought: entry / exit thresholds.
        position_pct: % of equity per position.
        max_positions: cap on simultaneous positions.
    """

    def __init__(self, symbols: Sequence[str] | None = None, period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, position_pct: float = 10.0, max_positions: int = 8):
        self.symbols = list(symbols) if symbols else None
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.position_pct = position_pct
        self.max_positions = max_positions
        self.name = f"SimpleRSIAgent({period}, {oversold:g}/{overbought:g})"

    def act(self, obs: Observation) -> list[Any]:
        self._last: dict[str, Any] = {"rsi": {}, "signals": []}
        if obs.open_orders:  # previous orders still working (e.g. market closed): wait
            self._last["signals"].append("open orders pending: waiting")
            return []
        actions: list[Any] = []
        n_positions = len(obs.positions)
        for sym in self.symbols or obs.symbols:
            value = rsi_last(obs.closes(sym, self.period * 3), self.period)
            if value != value:  # not enough history yet
                continue
            self._last["rsi"][sym] = round(value, 1)
            held = obs.position(sym)
            if held > 0 and value > self.overbought:
                actions.append(ClosePosition(sym))
                self._last["signals"].append(f"sell {sym}: RSI {value:.1f} > {self.overbought:g}")
                n_positions -= 1
            elif held == 0 and value < self.oversold and n_positions < self.max_positions:
                actions.append(OrderIntent.buy(sym, percent_of_equity=self.position_pct, tag=f"RSI {value:.1f}"))
                self._last["signals"].append(f"buy {sym}: RSI {value:.1f} < {self.oversold:g}")
                n_positions += 1
        return actions

    def explain(self) -> dict[str, Any] | None:
        return getattr(self, "_last", None)

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        last = getattr(self, "_last", None) or {}
        rsi = last.get("rsi") or {}
        rule = (f"Compro cuando el RSI({self.period}) baja de {self.oversold:g} (sobreventa) y vendo cuando supera "
                f"{self.overbought:g} (sobrecompra), con un {self.position_pct:g} % del capital por posición.")
        if "open orders pending: waiting" in last.get("signals", []):
            return d.entry(d.did(orders), rule + " Hoy tenía órdenes pendientes y esperé.", "")
        done = {(o["side"], o["symbol"]) for o in orders if o.get("status") in d.DONE}
        acts = [f"{'compré' if side == 'buy' else 'vendí'} {s} (RSI {rsi[s]:.0f})" for side, s in sorted(done) if s in rsi]
        low = sum(v < self.oversold for v in rsi.values())
        high = sum(v > self.overbought for v in rsi.values())
        mean = sum(rsi.values()) / len(rsi) if rsi else float("nan")
        segun = (f"Con el mercado {d.tag(situation)} había {low} activos en sobreventa y {high} en sobrecompra "
                 f"(RSI medio {mean:.0f}). ")
        segun += {"bajista": "En las caídas aparecen más sobreventas, así que compro más.",
                  "alcista": "En las subidas casi no hay sobreventas: compro poco y vendo lo sobrecomprado."}.get(
            d.trend(situation) or "", "En mercados laterales opero poco: solo en los extremos.")
        return d.entry(d.did(orders), rule + (" Hoy: " + "; ".join(acts) + "." if acts else ""), segun)


class MovingAverageCrossAgent(Agent):
    """Trend following: hold equal weights in symbols whose fast SMA is above the slow SMA.

    Rebalances only when the set of "uptrend" symbols differs from the symbols actually held, to
    limit trading costs. Comparing with the real positions (not with what it last asked for) keeps
    the agent in sync with its account even when an order is rejected or only partly filled.

    Args:
        max_weight: cap per symbol (fraction of equity), e.g. to respect a broker / risk limit.
    """

    def __init__(self, symbols: Sequence[str] | None = None, fast: int = 20, slow: int = 50,
                 invested_fraction: float = 0.95, max_weight: float = 1.0):
        self.symbols = list(symbols) if symbols else None
        self.fast, self.slow = fast, slow
        self.invested_fraction = invested_fraction
        self.max_weight = max_weight
        self.name = f"MovingAverageCrossAgent({fast}/{slow})"

    def act(self, obs: Observation) -> Any:
        if obs.open_orders:
            self._last = {"waiting": True}
            return []
        up = []
        for sym in self.symbols or obs.symbols:
            closes = obs.closes(sym, self.slow)
            if len(closes) == self.slow and closes[-self.fast:].mean() > closes.mean():
                up.append(sym)
        chosen = frozenset(up)
        held = frozenset(s for s, q in obs.positions.items() if q > 0)
        self._last = {"uptrend": sorted(chosen), "held": sorted(held), "checked": len(self.symbols or obs.symbols)}
        if chosen == held:
            return []
        w = min(self.max_weight, self.invested_fraction / len(chosen)) if chosen else 0.0
        return TargetWeights({s: w for s in chosen})

    def explain(self) -> dict[str, Any] | None:
        return getattr(self, "_last", None)

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        last = getattr(self, "_last", None) or {}
        if last.get("waiting"):
            return d.entry(d.did(orders), "Tenía órdenes pendientes de ejecutar y esperé.", "")
        up, held, n = last.get("uptrend", []), last.get("held", []), last.get("checked") or 1
        new, gone = sorted(set(up) - set(held)), sorted(set(held) - set(up))
        por = (f"Mantengo a partes iguales los activos cuya media de {self.fast} sesiones está por encima de la de "
               f"{self.slow}: hoy {len(up)} de {n}.")
        if new:
            por += " Entran en tendencia: " + ", ".join(new) + "."
        if gone:
            por += " Pierden la tendencia: " + ", ".join(gone) + "."
        if not new and not gone:
            por += " Las tendencias no cambiaron, así que no toqué la cartera."
        share = len(up) / n
        segun = (f"Con el mercado {d.tag(situation)}, el {share:.0%} del universo está en tendencia alcista: "
                 + ("estoy muy invertido, sigo la subida." if share > 0.6 else
                    "me quedo con poca exposición hasta que vuelvan las tendencias." if share < 0.3 else
                    "tengo una exposición intermedia."))
        return d.entry(d.did(orders), por, segun)


class RandomAgent(Agent):
    """Controlled random agent: each step, with probability ``trade_prob``, buys or sells a random
    symbol for ``position_pct`` % of equity. Seeded, so runs are reproducible. A useful baseline:
    a real strategy should beat it after costs.

    For ``PaperTradingEnv`` pass ``action_space=env.action_space`` and it samples from it instead.
    """

    def __init__(self, trade_prob: float = 0.2, position_pct: float = 10.0, seed: int = 0,
                 action_space: Any = None):
        self.trade_prob = trade_prob
        self.position_pct = position_pct
        self.seed = seed
        self.action_space = action_space
        self.name = f"RandomAgent(p={trade_prob:g})"
        self.rng = np.random.default_rng(seed)
        self._episodes = 0

    def reset(self) -> None:
        self.rng = np.random.default_rng([self.seed, self._episodes])
        self._episodes += 1
        if self.action_space is not None:
            self.action_space.seed(int(self.rng.integers(2**31 - 1)))

    def act(self, obs: Any) -> Any:
        if self.action_space is not None:
            return self.action_space.sample()
        if self.rng.random() >= self.trade_prob:
            return []
        sym = obs.symbols[int(self.rng.integers(len(obs.symbols)))]
        if obs.position(sym) > 0 and self.rng.random() < 0.5:
            return ClosePosition(sym)
        return OrderIntent.buy(sym, percent_of_equity=self.position_pct, tag="random")


class BuyAndHoldAgent(Agent):
    """Benchmark: invest equally in all symbols on the first step, then hold."""

    name = "BuyAndHoldAgent"

    def __init__(self, symbols: Sequence[str] | None = None, invested_fraction: float = 0.95):
        self.symbols = list(symbols) if symbols else None
        self.invested_fraction = invested_fraction
        self._done = False

    def reset(self) -> None:
        self._done = False

    def act(self, obs: Observation) -> Any:
        if self._done:
            return []
        syms = self.symbols or obs.symbols
        if obs.open_orders or not syms:
            return []
        self._done = obs.tradable is None  # partial session (weekend crypto): finish the rest on the next full one
        return TargetWeights({s: self.invested_fraction / len(syms) for s in syms})

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        por = ("Hoy hice mi única compra: todos los activos a partes iguales." if orders else
               "Compré todos los activos a partes iguales el primer día y los mantengo; nunca vendo.")
        return d.entry(d.did(orders) if orders else "Mantuve la cartera sin tocarla.", por,
                       f"La situación {d.tag(situation)} no cambia nada: soy la referencia pasiva con la que "
                       "se compara a los demás.")

    def state_dict(self) -> dict[str, Any]:
        return {"done": self._done}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._done = bool(state.get("done", False))
