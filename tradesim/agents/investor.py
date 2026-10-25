"""A traditional long-term investor: a policy portfolio, held and rebalanced, never traded.

No signals, no stops, no market timing. The agent owns a strategic allocation (broad index funds, bonds, gold, a
basket of quality blue chips and a small crypto slice), reviews the stock basket once a year and brings the portfolio
back to its targets once a quarter, or earlier only if an asset class drifts outside its tolerance band (the classic
"5/25" rule: 5 points or 25 % of its target, whichever is smaller). Small deviations are left alone to keep turnover
and costs low.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import numpy as np

from tradesim.agents.base import Agent, Observation, OrderIntent

# Strategic allocation by asset class (fractions of equity); "stocks" is the weight of the blue-chip basket.
DEFAULT_POLICY: dict[str, Any] = {
    "core": {"SPY": 0.25, "QQQ": 0.12, "EEM": 0.06},
    "bonds": {"TLT": 0.15},
    "gold": {"GLD": 0.07},
    "stocks": 0.28,
    "crypto": {"BTC-USD": 0.035, "ETH-USD": 0.015},
}  # 2 % stays in cash


def _quarter(t: datetime) -> tuple[int, int]:
    return t.year, (t.month - 1) // 3


class TraditionalInvestorAgent(Agent):
    """Buy, hold and rebalance a diversified policy portfolio.

    Args:
        policy: asset class -> {symbol: weight}, plus ``"stocks"`` -> weight of the stock basket.
        n_stocks: blue chips in the basket, at most ``per_sector`` per sector, equal weighted.
        abs_band, rel_band: drift tolerance of an asset class (5/25 rule) before an early rebalance.
        min_trade_pct: per-asset deviations below this (fraction of equity) are not traded at a rebalance.
        lookback: days of history used to pick the blue chips (steady 1-year compounders: return / volatility).
        max_weight: cap per asset (keep it within the account's ``max_position_pct``); the excess stays in cash.
    """

    name = "TraditionalInvestorAgent"

    def __init__(self, policy: dict[str, Any] | None = None, n_stocks: int = 8, per_sector: int = 1,
                 abs_band: float = 0.05, rel_band: float = 0.25, min_trade_pct: float = 0.01,
                 lookback: int = 252, max_weight: float = 0.25, symbols: Sequence[str] | None = None):
        self.policy = policy or DEFAULT_POLICY
        self.n_stocks, self.per_sector = n_stocks, per_sector
        self.abs_band, self.rel_band, self.min_trade_pct = abs_band, rel_band, min_trade_pct
        self.lookback, self.max_weight = lookback, max_weight
        self.symbols = list(symbols) if symbols else None
        self.reset()

    def reset(self) -> None:
        self.basket: list[str] = []
        self.reviewed_year: int | None = None
        self.rebalanced_q: tuple[int, int] | None = None
        self.log: list[dict[str, Any]] = []
        self._last: dict[str, Any] = {}

    # ------------------------------------------------------------------ policy
    def _review_basket(self, obs: Observation, universe: list[str]) -> list[str]:
        """Annual review: steady compounders, one per sector; holdings stay unless they fall out of the top 2n."""
        fixed = {s for k, v in self.policy.items() if isinstance(v, dict) for s in v}
        sim = obs._sim
        scored = []
        for sym in universe:
            asset = sim.universe.get(sym) if sim is not None else None
            if sym in fixed or asset is None or asset.asset_class.value != "equity":
                continue
            c = obs.closes(sym, self.lookback + 1)
            if len(c) < self.lookback * 0.9 or not (c > 0).all():
                continue
            r = np.diff(np.log(c))
            ret, vol = float(np.log(c[-1] / c[0])), float(r.std() * np.sqrt(252))
            if ret > 0 and c[-1] > c[-200:].mean():  # growing and above its 200-day average
                scored.append((ret / max(vol, 1e-6), sym, getattr(asset, "sector", "") or sym))
        scored.sort(reverse=True)
        keep_zone = {s for _, s, _ in scored[: 2 * self.n_stocks]}
        out, sectors = [], {}
        for _, sym, sector in [x for x in scored if x[1] in self.basket and x[1] in keep_zone] + scored:
            if sym in out or sectors.get(sector, 0) >= self.per_sector or len(out) >= self.n_stocks:
                continue
            out.append(sym)
            sectors[sector] = sectors.get(sector, 0) + 1
        return out

    def targets(self, available: set[str]) -> dict[str, float]:
        """Per-symbol targets; a missing fund's weight is shared within its asset class (or by the rest)."""
        out: dict[str, float] = {}
        missing = 0.0
        for cls, v in self.policy.items():
            if cls == "stocks":
                group = {s: float(v) / len(self.basket) for s in self.basket} if self.basket else {}
                if not group:
                    missing += float(v)
            else:
                group = {s: float(w) for s, w in v.items()}
            ok = {s: w for s, w in group.items() if s in available}
            lost = sum(group.values()) - sum(ok.values())
            if ok:
                k = sum(group.values()) / sum(ok.values())
                out.update({s: w * k for s, w in ok.items()})
            else:
                missing += lost
        if missing and out:  # e.g. no stock passes the review: hold more of the index funds instead
            total = sum(out.values())
            out = {s: w * (1 + missing / total) for s, w in out.items()}
        return {s: min(w, self.max_weight) for s, w in out.items()}

    def _classes(self) -> dict[str, list[str]]:
        cls = {k: list(v) for k, v in self.policy.items() if isinstance(v, dict)}
        cls["stocks"] = list(self.basket)
        return cls

    # ------------------------------------------------------------------ act
    def act(self, obs: Observation) -> Any:
        if obs.tradable is not None or obs.open_orders or obs.equity <= 0:
            return []  # an investor does not use weekend sessions; wait for pending orders
        universe = [s for s in (self.symbols or obs.symbols) if obs.prices.get(s, 0) > 0]
        now = obs.timestamp
        reasons = []
        if self.reviewed_year != now.year or not self.basket:
            new = self._review_basket(obs, universe)
            if new != self.basket:
                reasons.append(f"revisión anual de la cartera de acciones: {', '.join(new) or 'ninguna'}")
            self.basket, self.reviewed_year = new, now.year
        tgt = self.targets(set(universe))
        weights = {s: obs.weight(s) for s in set(tgt) | {s for s, q in obs.positions.items() if q > 0}}
        drift = {}
        for cls, syms in self._classes().items():
            t = sum(tgt.get(s, 0.0) for s in syms)
            w = sum(weights.get(s, 0.0) for s in syms)
            drift[cls] = round(w - t, 4)
            if t > 0 and abs(w - t) > min(self.abs_band, self.rel_band * t):
                reasons.append(f"{cls} fuera de banda ({w:.1%} vs objetivo {t:.1%})")
        invested = sum(weights.values())
        if invested < 0.5 * sum(tgt.values()):
            reasons.insert(0, "inversión inicial")
        elif self.rebalanced_q != _quarter(now):
            reasons.append("rebalanceo trimestral")
        if self.rebalanced_q is None:
            self.rebalanced_q = _quarter(now)
        self._last = {"basket": self.basket, "targets": {s: round(w, 4) for s, w in sorted(tgt.items())},
                      "drift_by_class": drift, "reasons": reasons,
                      "signals": reasons + ([f"cartera de acciones: {', '.join(self.basket)}"] if reasons else [])}
        if not reasons:
            return []
        self.rebalanced_q = _quarter(now)
        eq = float(obs.equity)
        floor = 0.0 if "inversión inicial" in reasons else self.min_trade_pct
        sells, buys = [], []
        for sym in sorted(weights.keys() | tgt.keys()):
            delta = tgt.get(sym, 0.0) - weights.get(sym, 0.0)
            if sym not in tgt and obs.position(sym) > 0:
                sells.append(OrderIntent.sell(sym, quantity=float(obs.position(sym)), tag="fuera de la política"))
            elif abs(delta) >= max(floor, 1.0 / eq) and obs.prices.get(sym, 0) > 0:
                (buys if delta > 0 else sells).append((sym, delta))
        orders = [o for o in sells if isinstance(o, OrderIntent)]
        cash = float(obs.cash) + sum(weights.get(o.symbol, 0.0) * eq for o in orders)
        for sym, d in [x for x in sells if not isinstance(x, OrderIntent)]:
            qty = min(float(obs.position(sym)), -d * eq / obs.prices[sym])
            orders.append(OrderIntent.sell(sym, quantity=qty, tag=f"rebalanceo: {reasons[0]}"))
            cash += qty * obs.prices[sym]
        need = sum(d * eq for _, d in buys)
        k = min(1.0, cash * 0.995 / need) if need > 0 else 0.0  # buys never exceed the cash available
        for sym, d in buys:
            if d * eq * k >= 1.0:
                orders.append(OrderIntent.buy(sym, notional=round(d * eq * k, 2), tag=f"rebalanceo: {reasons[0]}"))
        self.log.append({"date": now.isoformat(), "reasons": reasons, "orders": len(orders)})
        del self.log[:-20]
        return orders

    # ------------------------------------------------------------------ state
    def explain(self) -> dict[str, Any] | None:
        return self._last or None

    def state_dict(self) -> dict[str, Any]:
        return {"basket": self.basket, "reviewed_year": self.reviewed_year,
                "rebalanced_q": list(self.rebalanced_q) if self.rebalanced_q else None, "log": self.log}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.basket = list(state.get("basket", []))
        self.reviewed_year = state.get("reviewed_year")
        q = state.get("rebalanced_q")
        self.rebalanced_q = tuple(q) if q else None
        self.log = list(state.get("log", []))
