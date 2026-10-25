"""A discretionary portfolio manager driven by Claude: no fixed rule, it reads the market and decides.

At each decision :class:`ClaudePortfolioManager` assembles a dossier and asks Claude for the whole target
portfolio, with a thesis, catalysts and an invalidation level for every position:

* **technical picture** of every asset (trend, support/resistance, candlestick and chart patterns,
  divergences) from :mod:`tradesim.patterns`;
* **macro data relevant to each asset type** (rates and inflation for bonds, real yields and the dollar
  for gold and crypto, oil for energy, the curve and credit spreads for banks...);
* **news**: the detailed analyses of the important items and the triage of everything else;
* **the event calendar of the next 30 days** (central banks, data releases, earnings, expiries...);
* **its own memory**: open theses, how past decisions turned out, lessons it wrote for itself.

Claude may also search the web (and open pages) before deciding. The research comes from
:mod:`tradesim.research` (attach a :class:`~tradesim.research.refresh.ResearchStore` with
:meth:`attach_research`; the live runner does it). Without research it still works on technicals only;
without an Anthropic API key it holds and says why.

Learning: after every decision it reviews the outcome of earlier theses (P&L since entry, closed trades)
and records lessons that are fed back into later prompts, so its judgement accumulates across runs. The
memory is part of ``state_dict`` and moves with the agent to a real account.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from tradesim.agents.base import Agent, ClosePosition, Observation, OrderIntent
from tradesim.data.base import DataUnavailableError

log = logging.getLogger(__name__)

DECISION_TOOL = {
    "name": "submit_decision",
    "description": "Submit the complete target portfolio and your reasoning. Assets not listed are sold.",
    "input_schema": {
        "type": "object",
        "properties": {
            "market_view": {"type": "string", "description": "Your reading of the market today (<= 200 words)"},
            "regime": {"type": "string", "enum": ["risk_on", "neutral", "risk_off"]},
            "key_risks": {"type": "array", "items": {"type": "string"}},
            "positions": {
                "type": "array",
                "description": "The full target portfolio. Repeat unchanged positions with their current weight.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "target_weight": {"type": "number", "description": "fraction of equity, e.g. 0.08"},
                        "conviction": {"type": "string", "enum": ["low", "medium", "high"]},
                        "thesis": {"type": "string"},
                        "catalysts": {"type": "array", "items": {"type": "string"}},
                        "invalidation": {"type": "string", "description": "price level or condition that proves you wrong"},
                        "horizon_days": {"type": "integer"},
                    },
                    "required": ["symbol", "target_weight", "conviction", "thesis", "catalysts", "invalidation",
                                 "horizon_days"],
                },
            },
            "watchlist": {"type": "array", "items": {"type": "string"},
                          "description": "Assets you are watching and what would make you act"},
            "lessons": {"type": "array", "items": {"type": "string"},
                        "description": "New lessons from reviewing past decisions (may be empty)"},
            "notes_for_next_time": {"type": "string"},
        },
        "required": ["market_view", "regime", "key_risks", "positions", "watchlist", "lessons", "notes_for_next_time"],
    },
}


class ClaudePortfolioManager(Agent):
    """Discretionary LLM portfolio manager (see module docstring).

    Args:
        symbols: universe (default: the observation's symbols).
        model: Claude model for the decision.
        effort: reasoning effort (``low`` ... ``max``).
        max_weight: cap per position (fraction of equity); ``max_positions`` caps the count.
        invested_fraction: maximum gross exposure (the rest stays in cash).
        min_change: weight changes smaller than this are not traded (avoids churning on noise).
        web_searches / web_fetches: per-decision caps on Claude's own web research (0 disables).
        every: decide every ``every`` observations (1 in live trading = once per daily run).
        language: language of the texts it writes (theses, views, lessons).
        client: an ``anthropic.Anthropic``; by default built from ``ANTHROPIC_API_KEY`` on first use.
    """

    def __init__(self, symbols: Sequence[str] | None = None, model: str = "claude-opus-5-5", effort: str = "high",
                 max_weight: float = 0.15, max_positions: int = 12, invested_fraction: float = 0.95,
                 min_change: float = 0.02, web_searches: int = 5, web_fetches: int = 3, every: int = 1,
                 language: str = "es", max_news: int = 40, client: Any = None):
        self.symbols = list(symbols) if symbols else None
        self.model, self.effort = model, effort
        self.max_weight, self.max_positions, self.invested_fraction = max_weight, max_positions, invested_fraction
        self.min_change = min_change
        self.web_searches, self.web_fetches = web_searches, web_fetches
        self.every, self.language, self.max_news = every, language, max_news
        self.client = client
        self.research: Any = None
        self.name = f"ClaudePortfolioManager({model})"
        # memory (persisted)
        self.theses: dict[str, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.lessons: list[str] = []
        self.reviews: list[dict[str, Any]] = []
        self.notes = ""
        self.usage_total: dict[str, Any] = {}
        self.steps = 0
        self._last: dict[str, Any] | None = None

    # ------------------------------------------------------------------ wiring
    def attach_research(self, store: Any) -> None:
        """Give the agent access to news, macro and calendar (:class:`tradesim.research.ResearchStore`)."""
        self.research = store

    def _client(self) -> Any:
        if self.client is None:
            from tradesim.research.llm import make_client

            self.client = make_client()
        return self.client

    # ------------------------------------------------------------------ acting
    def act(self, obs: Observation) -> list[Any]:
        self.steps += 1
        if (self.steps - 1) % self.every:
            return []
        client = self._client()
        if client is None:
            self._last = {"status": "inactive", "message": "sin ANTHROPIC_API_KEY: el analista no opera (mantiene)"}
            return []
        from tradesim.research.llm import WEB_FETCH, WEB_SEARCH, LLMError, Usage, submit_call

        symbols = [s for s in (self.symbols or obs.symbols) if s in obs.prices and obs.prices[s] == obs.prices[s]]
        review = self._review(obs)
        dossier = self._dossier(obs, symbols, review)
        tools = []
        if self.web_searches:
            tools.append({**WEB_SEARCH, "max_uses": self.web_searches})
        if self.web_fetches:
            tools.append({**WEB_FETCH, "max_uses": self.web_fetches})
        usage = Usage()
        try:
            decision, sources = submit_call(client, self.model, self._system(), dossier, DECISION_TOOL, usage,
                                            server_tools=tools, max_tokens=48000, effort=self.effort)
        except LLMError as exc:
            self._account(usage)
            self._last = {"status": "error", "message": f"sin decisión: {exc}", "usage": usage.to_dict()}
            return []
        self._account(usage)
        targets, notes = self._targets(decision, obs, symbols)
        actions, changes = self._orders(obs, targets, decision)
        self._remember(obs, decision, targets, review)
        self._last = {"status": "ok", "market_view": decision.get("market_view"), "regime": decision.get("regime"),
                      "key_risks": decision.get("key_risks"), "targets": {s: round(w, 4) for s, w in targets.items()},
                      "positions": [p for p in decision.get("positions", []) if p.get("symbol") in targets],
                      "changes": changes, "watchlist": decision.get("watchlist"), "lessons_new": decision.get("lessons"),
                      "adjustments": notes, "sources": sources[:15], "usage": usage.to_dict(),
                      "dossier_chars": len(dossier)}
        return actions

    # ------------------------------------------------------------------ prompt
    def _system(self) -> str:
        lang = {"es": "Spanish", "en": "English"}.get(self.language, self.language)
        return (
            "You are the portfolio manager of a long-only account (no shorting, no leverage) that trades US stocks, "
            "ETFs and crypto with real prices, real spreads and fees. You are discretionary: there is no fixed rule. "
            "Each day you study the technical picture, the macro data relevant to each asset type, the news and the "
            "calendar of upcoming events, and you decide the whole target portfolio.\n\n"
            "How to work:\n"
            "- Start from your open theses: is each one still valid, invalidated, or played out? Check the news, "
            "price action and upcoming catalysts for every holding.\n"
            "- Look for new opportunities where several independent signals agree (e.g. a constructive chart pattern "
            "+ supportive macro + a positive catalyst). One signal alone is weak evidence.\n"
            "- Mind the calendar: size down or wait before binary events (FOMC, CPI, payrolls, earnings) unless the "
            "position is the bet on that event and the risk is acceptable.\n"
            "- Costs are real: do not trade for small changes; holding is a valid decision. Cash is a position.\n"
            f"- Limits: at most {self.max_positions} positions, at most {self.max_weight:.0%} of equity each, total "
            f"invested at most {self.invested_fraction:.0%}.\n"
            "- Be honest about uncertainty. Write lessons only when a past decision teaches something specific.\n"
            "- Use web search only to check something specific that matters for a decision (a fresh development, a "
            "figure you lack); the dossier already has news, macro and calendar.\n\n"
            f"Write every text field (market view, theses, catalysts, invalidation, risks, watchlist, lessons, notes) "
            f"in {lang}. Finish by calling submit_decision exactly once with the complete target portfolio."
        )

    def _dossier(self, obs: Observation, symbols: list[str], review: list[dict[str, Any]]) -> str:
        now = obs.timestamp
        pf = obs.portfolio()
        parts = [f"DATE: {now:%Y-%m-%d %H:%M} UTC ({now:%A})",
                 f"ACCOUNT: equity {pf['equity']}, cash {pf['cash']}, drawdown {pf.get('current_drawdown_pct', 0):.2f}% "
                 f"(max {pf.get('max_drawdown_pct', 0):.2f}%), fees paid {pf.get('total_fees')}"]
        rows = []
        for p in pf.get("positions", []):
            sym = p["symbol"]
            th = self.theses.get(sym, {})
            rows.append(f"- {sym}: weight {p['weight']:.1%}, qty {p['quantity']}, avg {p.get('avg_price')}, "
                        f"uPnL {p.get('unrealized_pnl')}"
                        + (f", held since {th['opened_at'][:10]}, thesis: {th.get('thesis')}; invalidation: "
                           f"{th.get('invalidation')}; catalysts: {', '.join(th.get('catalysts') or [])}" if th else ""))
        parts.append("OPEN POSITIONS:\n" + ("\n".join(rows) or "(none, all cash)"))
        if review:
            parts.append("REVIEW OF PAST DECISIONS (outcomes since your last decision):\n" + "\n".join(
                f"- {r['symbol']}: {r['outcome']}" for r in review))
        if self.decisions:
            parts.append("YOUR RECENT DECISIONS:\n" + "\n".join(
                f"- {d['date'][:10]} ({d.get('regime')}, equity {d.get('equity'):.0f}): {d.get('summary', '')[:300]}"
                for d in self.decisions[-5:]))
        if self.lessons:
            parts.append("YOUR LESSONS SO FAR:\n" + "\n".join(f"- {x}" for x in self.lessons[-12:]))
        if self.notes:
            parts.append(f"YOUR NOTES FROM LAST TIME: {self.notes}")
        parts += self._research_sections(now, symbols)
        parts.append("TECHNICALS (daily bars; score = net pattern bullishness -1..1):\n" + self._technicals(obs, symbols))
        return "\n\n".join(parts)

    def _research_sections(self, now: datetime, symbols: list[str]) -> list[str]:
        if self.research is None:
            return ["RESEARCH: not available (technicals only)."]
        from tradesim.research.calendar import calendar_brief, upcoming
        from tradesim.research.macro import RELEVANCE, macro_brief

        out = []
        macro = self.research.load("macro")
        if macro.get("indicators"):
            out.append("MACRO DATA:\n" + macro_brief(macro))
            out.append("WHICH MACRO DATA MATTERS FOR EACH ASSET TYPE:\n" + "\n".join(
                f"- {k}: {', '.join(v)}" for k, v in RELEVANCE.items()))
        cal = self.research.load("calendar")
        if cal.get("events"):
            soon = upcoming(cal, now, 7, "low")
            later = [e for e in upcoming(cal, now, 31, "medium") if e not in soon]
            text = "NEXT 7 DAYS:\n" + calendar_brief(soon) + "\n\nREST OF THE MONTH (medium/high):\n" + calendar_brief(later)
            past = [e for e in cal["events"] if e["date"] < now.date().isoformat() and e.get("actual")]
            if past:
                text += "\n\nRECENT RESULTS:\n" + "\n".join(f"- {e['date']} {e['title']}: {e['actual']}" for e in past[-15:])
            if cal.get("research_summary"):
                text += f"\n\nCALENDAR RESEARCHER'S NOTE: {cal['research_summary']}"
            out.append("EVENT CALENDAR (ids in brackets; ET = New York time):\n" + text)
        news = self.research.news()
        recent = news.recent(72, now)
        deep = [i for i in recent if i.analysis][: self.max_news]
        if deep:
            out.append("NEWS ANALYSES (most recent first):\n" + "\n".join(_fmt_analysis(i) for i in deep))
        lines = []
        for sym in symbols:
            items = [i for i in news.for_symbol(sym) if i.triage and i.triage["relevance"] >= 0.45
                     and i.published >= (now - timedelta(hours=48)).isoformat() and not i.analysis][:4]
            for i in items:
                t = i.triage
                lines.append(f"- {sym} {i.published[5:16]} [{t['impact']}, sent {t['sentiment']:+.1f}, rel "
                             f"{t['relevance']:.1f}] {t['summary']}")
        macro_items = [i for i in recent if i.triage and not i.triage.get("symbols") and i.triage["relevance"] >= 0.5
                       and not i.analysis][:15]
        lines += [f"- MACRO {i.published[5:16]} [{i.triage['impact']}] {i.triage['summary']}" for i in macro_items]
        if lines:
            out.append("OTHER NEWS (triage, last 48h):\n" + "\n".join(lines))
        return out

    def _technicals(self, obs: Observation, symbols: list[str]) -> str:
        from tradesim.research.macro import asset_profile

        sim = obs.sim
        lines = []
        for sym in symbols:
            try:
                rep = obs.patterns(sym)
            except (DataUnavailableError, KeyError, ValueError) as exc:
                lines.append(f"{sym}: no data ({exc})")
                continue
            try:
                profile = asset_profile(sim.universe.get(sym))
            except (AttributeError, KeyError):
                profile = "?"
            lines.append(f"[{profile}] {rep.summary(max_patterns=5)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ decision -> orders
    def _targets(self, decision: dict[str, Any], obs: Observation, symbols: list[str]) -> tuple[dict[str, float], list[str]]:
        notes: list[str] = []
        targets: dict[str, float] = {}
        for p in decision.get("positions", []):
            sym = p.get("symbol")
            if sym not in symbols:
                notes.append(f"ignored {sym}: not in the tradable universe")
                continue
            try:
                w = float(p.get("target_weight", 0))
            except (TypeError, ValueError):
                continue
            if w > 1.0:  # a percentage instead of a fraction
                w /= 100.0
            if not math.isfinite(w) or w <= 0:
                continue
            if w > self.max_weight:
                notes.append(f"{sym}: {w:.1%} capped to {self.max_weight:.0%}")
                w = self.max_weight
            targets[sym] = w
        if len(targets) > self.max_positions:
            keep = sorted(targets, key=lambda s: -targets[s])[: self.max_positions]
            notes.append(f"kept the {self.max_positions} largest of {len(targets)} positions")
            targets = {s: targets[s] for s in keep}
        total = sum(targets.values())
        if total > self.invested_fraction:
            k = self.invested_fraction / total
            targets = {s: w * k for s, w in targets.items()}
            notes.append(f"scaled weights by {k:.2f} to stay within {self.invested_fraction:.0%} invested")
        return targets, notes

    def _orders(self, obs: Observation, targets: dict[str, float], decision: dict[str, Any]) -> tuple[list[Any], list[str]]:
        by_sym = {p.get("symbol"): p for p in decision.get("positions", [])}
        actions: list[Any] = []
        changes: list[str] = []
        held = {s for s, q in obs.positions.items() if q > 0}
        for sym in sorted(held - set(targets)):
            actions.append(ClosePosition(sym))
            changes.append(f"vender {sym} (fuera de la cartera objetivo)")
        for sym in sorted(set(targets) & held):
            cur, tgt = obs.weight(sym), targets[sym]
            price = obs.prices.get(sym, float("nan"))
            if tgt < cur - self.min_change and price > 0:
                qty = min(float(obs.position(sym)), (cur - tgt) * obs.equity / price)
                actions.append(OrderIntent.sell(sym, quantity=qty, tag=_tag("reducir", by_sym.get(sym))))
                changes.append(f"reducir {sym} {cur:.1%} -> {tgt:.1%}")
        for sym in sorted(targets, key=lambda s: -targets[s]):
            cur, tgt = obs.weight(sym), targets[sym]
            if tgt > cur + self.min_change:
                actions.append(OrderIntent.buy(sym, notional=round((tgt - cur) * obs.equity, 2),
                                               tag=_tag("comprar" if sym not in held else "ampliar", by_sym.get(sym))))
                changes.append(f"{'comprar' if sym not in held else 'ampliar'} {sym} {cur:.1%} -> {tgt:.1%}")
        return actions, changes

    # ------------------------------------------------------------------ memory
    def _review(self, obs: Observation) -> list[dict[str, Any]]:
        """Outcome of each thesis since the last decision (open: P&L so far; closed: realised)."""
        out = []
        for sym, th in list(self.theses.items()):
            price = obs.prices.get(sym, float("nan"))
            entry = th.get("entry_price")
            if obs.position(sym) > 0 and entry and price == price:
                ret = price / entry - 1
                out.append({"symbol": sym, "outcome": f"open since {th['opened_at'][:10]}, {ret:+.1%} vs entry "
                                                      f"{entry:g}; thesis was: {th.get('thesis', '')[:200]}"})
            elif obs.position(sym) <= 0:
                closed = [t for t in obs.sim.get_trade_history("closed", sym)
                          if str(t.get("closed_at", "")) >= th["opened_at"][:10]]
                pnl = sum(float(t.get("pnl", 0) or 0) for t in closed)
                out.append({"symbol": sym, "outcome": f"closed; realised P&L {pnl:+.2f} USD; thesis was: "
                                                      f"{th.get('thesis', '')[:200]}"})
                self.reviews.append({"date": obs.timestamp.isoformat(), "symbol": sym, "pnl": round(pnl, 2),
                                     "thesis": th.get("thesis", "")[:300]})
                del self.reviews[:-30]
                del self.theses[sym]
        return out

    def _remember(self, obs: Observation, decision: dict[str, Any], targets: dict[str, float],
                  review: list[dict[str, Any]]) -> None:
        now = obs.timestamp.isoformat()
        for p in decision.get("positions", []):
            sym = p.get("symbol")
            if sym not in targets:
                continue
            old = self.theses.get(sym, {})
            self.theses[sym] = {
                "opened_at": old.get("opened_at", now), "entry_price": old.get("entry_price") or obs.prices.get(sym),
                "target_weight": round(targets[sym], 4), "conviction": p.get("conviction"),
                "thesis": p.get("thesis", ""), "catalysts": p.get("catalysts") or [],
                "invalidation": p.get("invalidation", ""), "horizon_days": p.get("horizon_days"), "updated_at": now}
        for sym in [s for s in self.theses if s not in targets and obs.position(s) <= 0]:
            del self.theses[sym]  # never bought (or sold now): nothing to track
        new_lessons = [x.strip() for x in decision.get("lessons", []) if isinstance(x, str) and x.strip()]
        self.lessons = (self.lessons + new_lessons)[-15:]
        self.notes = str(decision.get("notes_for_next_time", ""))[:1500]
        self.decisions.append({"date": now, "regime": decision.get("regime"), "equity": obs.equity,
                               "summary": str(decision.get("market_view", ""))[:600],
                               "targets": {s: round(w, 4) for s, w in targets.items()}})
        del self.decisions[:-20]

    def _account(self, usage: Any) -> None:
        from tradesim.research.llm import Usage

        total = Usage()
        total.by_model = self.usage_total.get("by_model", {})
        total.merge(usage)
        self.usage_total = total.to_dict()

    # ------------------------------------------------------------------ Agent hooks
    def explain(self) -> dict[str, Any] | None:
        return self._last

    def learning_summary(self) -> dict[str, Any] | None:
        last = self.decisions[-1] if self.decisions else {}
        return {"method": "Claude discretionary analysis (technicals + macro + news + calendar) with memory",
                "model": self.model, "decisions": len(self.decisions), "open_theses": len(self.theses),
                "lessons": self.lessons[-8:], "last_regime": last.get("regime"), "last_view": last.get("summary"),
                "recent_reviews": self.reviews[-5:], "llm_cost_usd_total": self.usage_total.get("cost_usd_estimate"),
                "theses": self.theses}

    def state_dict(self) -> dict[str, Any]:
        return {"theses": self.theses, "decisions": self.decisions, "lessons": self.lessons, "reviews": self.reviews,
                "notes": self.notes, "usage_total": self.usage_total, "steps": self.steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.theses = state.get("theses", {})
        self.decisions = state.get("decisions", [])
        self.lessons = state.get("lessons", [])
        self.reviews = state.get("reviews", [])
        self.notes = state.get("notes", "")
        self.usage_total = state.get("usage_total", {})
        self.steps = state.get("steps", 0)

    def reset(self) -> None:
        self.steps = 0


def _tag(verb: str, p: dict[str, Any] | None) -> str:
    thesis = (p or {}).get("thesis", "")
    return f"{verb}: {thesis}"[:240]


def _fmt_analysis(i: Any) -> str:
    a = i.analysis
    aff = "; ".join(f"{x['symbol']} {x['direction']}/{x['magnitude']}/{x['horizon']}" for x in a.get("affected", []))
    return (f"- [{i.published[5:16]}] {a.get('headline') or i.title} | {a.get('what_happened', '')[:400]} | "
            f"why: {a.get('why_it_matters', '')[:300]} | affects: {aff or 'n/a'} | priced in: {a.get('priced_in')} | "
            f"confidence {a.get('confidence')} | watch: {a.get('what_to_watch', '')[:200]}")
