"""A discretionary portfolio manager: no fixed rule, it reads the market and decides.

At each decision :class:`DiscretionaryAgent` assembles a dossier and asks an AI engine (free Gemini / Groq
tiers, or Claude) for the whole target portfolio, with a thesis, catalysts and an invalidation level for
every position. It reads:

* **technical picture** of every asset (trend, support/resistance, candlestick and chart patterns,
  divergences) from :mod:`tradesim.patterns`;
* **macro data relevant to each asset type** (rates and inflation for bonds, real yields and the dollar
  for gold and crypto, oil for energy, the curve and credit spreads for banks...);
* **news**: the detailed analyses of the important items and the triage of everything else;
* **the event calendar of the next 30 days** (central banks, data releases, earnings, expiries...);
* **its own memory**: open theses, how past decisions turned out, lessons it wrote for itself.

Claude may also search the web (and open pages) before deciding. Without any AI engine (no keys, or every
free quota spent) the free local engine decides from the same inputs. The research comes from
:mod:`tradesim.research` (attach a :class:`~tradesim.research.refresh.ResearchStore` with
:meth:`attach_research`; the live runner does it). Without research it still works on technicals only.

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

from tradesim.agents.base import Agent, ClosePosition, Observation, OrderIntent, SetExits
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
                        "stop_loss": {"type": "number",
                                      "description": "price of the protective stop (below the current price); "
                                                     "0 = let the system place it from support / volatility"},
                        "take_profit": {"type": "number",
                                        "description": "price at which to take profits (above the current price); "
                                                       "0 = let the system place it"},
                    },
                    "required": ["symbol", "target_weight", "conviction", "thesis", "catalysts", "invalidation",
                                 "horizon_days", "stop_loss", "take_profit"],
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


class DiscretionaryAgent(Agent):
    """Discretionary portfolio manager (see module docstring).

    Engines, tried in order (``engines``): free tiers of Google Gemini / Groq (free keys ``GEMINI_API_KEY`` /
    ``GROQ_API_KEY``), Claude (``ANTHROPIC_API_KEY``, paid; it can also search the web). If none is configured
    or all fail (quota, outage), the **local engine** decides for free: it weighs the technical patterns, the
    news sentiment, the trend, the macro regime and the upcoming events of every asset, and writes the same
    kind of theses and invalidation levels (``local_fallback=False`` holds instead).

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
        engines / free_models / daily_requests: engine order, model per free engine, requests per engine per day.
        local_fallback: decide with the local engine when no AI engine answers.
    """

    def __init__(self, symbols: Sequence[str] | None = None, model: str = "claude-opus-5-5", effort: str = "high",
                 max_weight: float = 0.15, max_positions: int = 12, invested_fraction: float = 0.95,
                 min_change: float = 0.02, web_searches: int = 5, web_fetches: int = 3, every: int = 1,
                 language: str = "es", max_news: int = 40, client: Any = None,
                 engines: Sequence[str] = ("gemini", "groq", "anthropic"), free_models: dict[str, str] | None = None,
                 daily_requests: dict[str, int] | None = None, local_fallback: bool = True,
                 min_score: float = 0.25):
        self.symbols = list(symbols) if symbols else None
        self.model, self.effort = model, effort
        self.max_weight, self.max_positions, self.invested_fraction = max_weight, max_positions, invested_fraction
        self.min_change = min_change
        self.web_searches, self.web_fetches = web_searches, web_fetches
        self.every, self.language, self.max_news = every, language, max_news
        self.client = client
        self.engines, self.free_models = list(engines), dict(free_models or {})
        self.daily_requests = daily_requests or {"gemini": 150, "groq": 400, "anthropic": 2000}
        self.local_fallback, self.min_score = local_fallback, min_score
        self.research: Any = None
        self.name = f"{type(self).__name__}({'/'.join(self.engines)})"
        self.last_engine: str | None = None
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
    def _backends(self) -> list[Any]:
        from tradesim.research.llm import AnthropicBackend, Budget, make_backends

        path = getattr(self.research, "dir", None)
        budget = Budget(path / "llm_budget.json" if path else None, self.daily_requests)
        free = make_backends([e for e in self.engines if e != "anthropic"], self.free_models, budget)
        client = self._client() if "anthropic" in self.engines else None
        out: list[Any] = []
        for name in self.engines:
            if name == "anthropic":
                if client is not None:
                    out.append(AnthropicBackend(client, self.model, budget))
            else:
                out += [b for b in free if b.name == name]
        return out

    def act(self, obs: Observation) -> list[Any]:
        self.steps += 1
        if (self.steps - 1) % self.every:
            return []
        from tradesim.research.llm import WEB_FETCH, WEB_SEARCH, LLMError, Usage, submit_call

        symbols = [s for s in (self.symbols or obs.symbols) if s in obs.prices and obs.prices[s] == obs.prices[s]]
        review = self._review(obs)
        reports = self._reports(obs, symbols)
        usage, errors, sources = Usage(), [], []
        decision, engine, dossier = None, None, ""
        for b in self._backends():
            try:
                if b.name == "anthropic":
                    b.budget.take(b.name)
                    dossier = self._dossier(obs, symbols, review, reports)
                    tools = []
                    if self.web_searches:
                        tools.append({**WEB_SEARCH, "max_uses": self.web_searches})
                    if self.web_fetches:
                        tools.append({**WEB_FETCH, "max_uses": self.web_fetches, "max_content_tokens": 8000})
                    decision, sources = submit_call(b.client, b.model, self._system(), dossier, DECISION_TOOL, usage,
                                                    server_tools=tools, max_tokens=48000, effort=self.effort)
                else:  # free tiers: no web tools and small context, so a compact dossier
                    dossier = self._dossier(obs, symbols, review, reports, compact=True)
                    decision = b.json(self._system().replace("Finish by calling submit_decision exactly once with",
                                                             "Answer with"), dossier, DECISION_TOOL["input_schema"],
                                      usage, max_tokens=6000)
                if not isinstance(decision.get("positions"), list):
                    raise LLMError(f"{b.name}: answer without a positions list")
                engine = b.label
                break
            except LLMError as exc:
                errors.append(str(exc)[:240])
                decision = None
        self._account(usage)
        if decision is None:
            if not self.local_fallback:
                status = "inactive" if not errors else "error"
                self._last = {"status": status, "usage": usage.to_dict(), "engine_errors": errors,
                              "message": ("sin motor de IA configurado: el analista no opera (mantiene)" if not errors
                                          else f"sin decisión: {errors[-1]}")}
                return []
            decision = self._local_decision(obs, symbols, reports)
            engine = "local (gratis, sin IA)"
        self.last_engine = engine
        targets, notes = self._targets(decision, obs, symbols)
        actions, changes = self._orders(obs, targets, decision, reports)
        self._remember(obs, decision, targets, review)
        self._last = {"status": "ok", "engine": engine, "engine_errors": errors,
                      "market_view": decision.get("market_view"), "regime": decision.get("regime"),
                      "key_risks": decision.get("key_risks"), "targets": {s: round(w, 4) for s, w in targets.items()},
                      "positions": [p for p in decision.get("positions", []) if p.get("symbol") in targets],
                      "changes": changes, "watchlist": decision.get("watchlist"), "lessons_new": decision.get("lessons"),
                      "adjustments": notes, "sources": sources[:15], "usage": usage.to_dict(),
                      "dossier_chars": len(dossier)}
        return actions

    def _reports(self, obs: Observation, symbols: list[str]) -> dict[str, Any]:
        out = {}
        for sym in symbols:
            try:
                out[sym] = obs.patterns(sym)
            except (DataUnavailableError, KeyError, ValueError) as exc:
                log.warning("%s: no patterns for %s (%s)", self.name, sym, exc)
        return out

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
            "- Every position gets a real stop-loss and take-profit order that executes during the session without "
            "you: give stop_loss at the level that invalidates the thesis (usually just below a support) and "
            "take_profit at a realistic target (a resistance or a pattern target), aiming for at least twice the "
            "risk. Stops are only ever raised, never lowered. Use 0 to let the system derive them.\n"
            f"- Limits: at most {self.max_positions} positions, at most {self.max_weight:.0%} of equity each, total "
            f"invested at most {self.invested_fraction:.0%}.\n"
            "- Be honest about uncertainty. Write lessons only when a past decision teaches something specific.\n"
            "- Use web search only to check something specific that matters for a decision (a fresh development, a "
            "figure you lack); the dossier already has news, macro and calendar.\n\n"
            f"Write every text field (market view, theses, catalysts, invalidation, risks, watchlist, lessons, notes) "
            f"in {lang}. Finish by calling submit_decision exactly once with the complete target portfolio."
        )

    def _dossier(self, obs: Observation, symbols: list[str], review: list[dict[str, Any]],
                 reports: dict[str, Any] | None = None, compact: bool = False) -> str:
        """Everything the model reads. ``compact`` keeps it within free tiers' small request size."""
        now = obs.timestamp
        pf = obs.portfolio()
        parts = [f"DATE: {now:%Y-%m-%d %H:%M} UTC ({now:%A})",
                 f"ACCOUNT: equity {pf['equity']}, cash {pf['cash']}, drawdown {pf.get('current_drawdown_pct', 0):.2f}% "
                 f"(max {pf.get('max_drawdown_pct', 0):.2f}%), fees paid {pf.get('total_fees')}"]
        rows = []
        for p in pf.get("positions", []):
            sym = p["symbol"]
            th = self.theses.get(sym, {})
            ex = obs.sim.get_exits(sym) if hasattr(obs.sim, "get_exits") else {}
            rows.append(f"- {sym}: weight {p['weight']:.1%}, qty {p['quantity']}, avg {p.get('avg_price')}, "
                        f"uPnL {p.get('unrealized_pnl')}, stop-loss {ex.get('stop_loss') or 'none'}, "
                        f"take-profit {ex.get('take_profit') or 'none'}"
                        + (f", held since {th['opened_at'][:10]}, thesis: {th.get('thesis')}; invalidation: "
                           f"{th.get('invalidation')}; catalysts: {', '.join(th.get('catalysts') or [])}" if th else ""))
        parts.append("OPEN POSITIONS:\n" + ("\n".join(rows) or "(none, all cash)"))
        if review:
            parts.append("REVIEW OF PAST DECISIONS (outcomes since your last decision):\n" + "\n".join(
                f"- {r['symbol']}: {r['outcome']}" for r in review))
        if self.decisions:
            parts.append("YOUR RECENT DECISIONS:\n" + "\n".join(
                f"- {d['date'][:10]} ({d.get('regime')}, equity {d.get('equity'):.0f}): {d.get('summary', '')[:300]}"
                for d in self.decisions[-(3 if compact else 5):]))
        if self.lessons:
            parts.append("YOUR LESSONS SO FAR:\n" + "\n".join(f"- {x}" for x in self.lessons[-(8 if compact else 12):]))
        if self.notes:
            parts.append(f"YOUR NOTES FROM LAST TIME: {self.notes}")
        parts += self._research_sections(now, symbols, compact)
        held = {s for s, q in obs.positions.items() if q > 0}
        parts.append("TECHNICALS (daily bars; score = net pattern bullishness -1..1)"
                     + (": your holdings plus the most bullish and bearish assets" if compact else "") + ":\n"
                     + self._technicals(obs, symbols, reports, held if compact else None))
        return "\n\n".join(parts)

    def _research_sections(self, now: datetime, symbols: list[str], compact: bool = False) -> list[str]:
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
            soon = upcoming(cal, now, 7, "medium" if compact else "low")
            later = [e for e in upcoming(cal, now, 31, "high" if compact else "medium") if e not in soon]
            text = "NEXT 7 DAYS:\n" + calendar_brief(soon) + "\n\nREST OF THE MONTH (medium/high):\n" + calendar_brief(later)
            past = [e for e in cal["events"] if e["date"] < now.date().isoformat() and e.get("actual")]
            if past:
                text += "\n\nRECENT RESULTS:\n" + "\n".join(f"- {e['date']} {e['title']}: {e['actual']}" for e in past[-15:])
            if cal.get("research_summary"):
                text += f"\n\nCALENDAR RESEARCHER'S NOTE: {cal['research_summary']}"
            out.append("EVENT CALENDAR (ids in brackets; ET = New York time):\n" + text)
        news = self.research.news()
        recent = news.recent(72, now)
        deep = [i for i in recent if i.analysis][: (12 if compact else self.max_news)]
        if deep:
            out.append("NEWS ANALYSES (most recent first):\n" + "\n".join(_fmt_analysis(i) for i in deep))
        lines = []
        for sym in symbols:
            items = [i for i in news.for_symbol(sym) if i.triage and i.triage["relevance"] >= 0.45
                     and i.published >= (now - timedelta(hours=48)).isoformat() and not i.analysis][:(1 if compact else 4)]
            for i in items:
                t = i.triage
                lines.append(f"- {sym} {i.published[5:16]} [{t['impact']}, sent {t['sentiment']:+.1f}, rel "
                             f"{t['relevance']:.1f}] {t['summary']}")
        macro_items = [i for i in recent if i.triage and not i.triage.get("symbols") and i.triage["relevance"] >= 0.5
                       and not i.analysis][:(6 if compact else 15)]
        lines += [f"- MACRO {i.published[5:16]} [{i.triage['impact']}] {i.triage['summary']}" for i in macro_items]
        if lines:
            out.append("OTHER NEWS (triage, last 48h):\n" + "\n".join(lines))
        return out

    def _technicals(self, obs: Observation, symbols: list[str], reports: dict[str, Any] | None = None,
                    only_held: set[str] | None = None) -> str:
        from tradesim.research.macro import asset_profile

        sim = obs.sim
        reports = reports if reports is not None else self._reports(obs, symbols)
        if only_held is not None:  # compact: holdings + 12 most bullish + 6 most bearish
            ranked = sorted((s for s in symbols if s in reports), key=lambda s: -reports[s].score())
            symbols = list(dict.fromkeys([s for s in symbols if s in only_held] + ranked[:12] + ranked[-6:]))
        lines = []
        for sym in symbols:
            rep = reports.get(sym)
            if rep is None:
                lines.append(f"{sym}: no data")
                continue
            try:
                profile = asset_profile(sim.universe.get(sym))
            except (AttributeError, KeyError):
                profile = "?"
            lines.append(f"[{profile}] {rep.summary(max_patterns=3 if only_held is not None else 5)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ local engine (free, no AI)
    def _news_scores(self, now: datetime, symbols: list[str]) -> dict[str, tuple[float, int, list[str]]]:
        """Per symbol: relevance- and recency-weighted sentiment of the last 72 h (-1..1), item count, headlines."""
        out: dict[str, tuple[float, int, list[str]]] = {}
        if self.research is None:
            return out
        news = self.research.news()
        for sym in symbols:
            num = den = 0.0
            ranked: list[tuple[float, str]] = []
            n = 0
            for i in news.for_symbol(sym):
                t = i.triage or {}
                age_h = (now - datetime.fromisoformat(i.published)).total_seconds() / 3600
                if age_h > 72 or t.get("relevance", 0) < 0.3:
                    continue
                named = t.get("named")
                sent = t.get("sentiment", 0.0) if named is None or sym in named else 0.0
                a = next((x for x in (i.analysis or {}).get("affected", []) if x.get("symbol") == sym), None)
                if a is not None and a.get("direction") in ("up", "down"):
                    sent = (1 if a["direction"] == "up" else -1) * {"low": 0.4, "medium": 0.7, "high": 1.0}.get(
                        a.get("magnitude"), 0.5)
                w = t.get("relevance", 0.3) * math.exp(-age_h / 48)
                num += w * sent
                den += w
                n += 1
                if t.get("category") != "opinion":
                    ranked.append((w, i.title[:90]))
            if n:
                heads = [title for _, title in sorted(ranked, reverse=True)[:2]]  # most relevant and recent first
                out[sym] = (num / den if den else 0.0, n, heads)
        return out

    def _regime(self) -> tuple[str, float, str]:
        """(regime, max invested fraction, explanation) from the macro gauges."""
        if self.research is None:
            return "neutral", self.invested_fraction * 0.85, "sin datos macro"
        ind = (self.research.load("macro") or {}).get("indicators", {})
        vix = (ind.get("vix") or {}).get("value")
        hy = ind.get("hy_spread") or {}
        hy_jump = (hy.get("change") or 0) >= 0.3
        why = []
        if vix is not None:
            why.append(f"VIX {vix:.1f}")
        if hy.get("value") is not None:
            why.append(f"diferencial high yield {hy['value']:.2f} pp")
        if (vix is not None and vix >= 28) or hy_jump:
            return "risk_off", self.invested_fraction * 0.5, ", ".join(why)
        if vix is not None and vix < 17:
            return "risk_on", self.invested_fraction, ", ".join(why)
        return "neutral", self.invested_fraction * 0.8, ", ".join(why)

    def _events(self, now: datetime) -> list[dict[str, Any]]:
        if self.research is None:
            return []
        from tradesim.research.calendar import upcoming

        return upcoming(self.research.load("calendar") or {}, now, 30, "medium")

    def _local_decision(self, obs: Observation, symbols: list[str], reports: dict[str, Any]) -> dict[str, Any]:
        """Free decision: weigh technicals, news, trend, macro regime and the calendar for every asset."""
        from tradesim.patterns import PATTERN_NAMES_ES

        now = obs.timestamp
        news = self._news_scores(now, symbols)
        regime, exposure, regime_why = self._regime()
        events = self._events(now)
        today = now.date().isoformat()
        soon = (now + timedelta(days=2)).date().isoformat()
        market_events = [e for e in events if e["date"] <= (now + timedelta(days=1)).date().isoformat()
                         and e["importance"] == "high" and "*all" in (e.get("symbols") or [])]
        scored: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            rep = reports.get(sym)
            if rep is None:
                continue
            tech = rep.score()
            trend = rep.trend or {}
            tr = {"bullish": 1.0, "bearish": -1.0}.get(trend.get("direction"), 0.0) * float(trend.get("quality") or 0)
            ns, nn, heads = news.get(sym, (0.0, 0, []))
            score = 0.55 * tech + 0.3 * ns + 0.15 * tr
            sym_events = [e for e in events if sym in (e.get("symbols") or [])]
            binary_soon = [e for e in sym_events if today <= e["date"] <= soon and e["category"] == "earnings"]
            scored[sym] = {"score": score, "tech": tech, "news": ns, "n_news": nn, "heads": heads, "trend": tr,
                           "rep": rep, "events": sym_events, "binary_soon": binary_soon}
        held = {s for s, q in obs.positions.items() if q > 0}
        keep = {s for s in held if s in scored and scored[s]["score"] > -0.1}
        fresh = [s for s in sorted(scored, key=lambda x: -scored[x]["score"])
                 if s not in held and scored[s]["score"] >= self.min_score and not scored[s]["binary_soon"]]
        chosen = sorted(keep, key=lambda x: -scored[x]["score"]) + fresh
        chosen = chosen[: self.max_positions]
        size = 0.5 if market_events else 1.0  # smaller new positions just before a market-wide binary event
        raw = {s: max(scored[s]["score"], 0.05) * (size if s not in held else 1.0) for s in chosen}
        total = sum(raw.values()) or 1.0
        positions = []
        for s in chosen:
            w = min(self.max_weight, exposure * raw[s] / total)
            d = scored[s]
            rep = d["rep"]
            pats = [PATTERN_NAMES_ES.get(p.name, p.name) for p in rep.patterns if p.direction == "bullish"][:2]
            sup = (rep.support or [{}])[0].get("price")
            thesis = (f"Puntuación {d['score']:+.2f}: técnico {d['tech']:+.2f}"
                      + (f" ({', '.join(pats)})" if pats else "")
                      + f", noticias {d['news']:+.2f} ({d['n_news']} relevantes), tendencia "
                      + ("alcista" if d["trend"] > 0.1 else "bajista" if d["trend"] < -0.1 else "lateral")
                      + (f". Titular: «{d['heads'][0]}»" if d["heads"] else ""))
            positions.append({
                "symbol": s, "target_weight": round(w, 4),
                "conviction": "high" if d["score"] >= 0.5 else "medium" if d["score"] >= 0.3 else "low",
                "thesis": thesis, "catalysts": [f"{e['date']} {e['title']}" for e in d["events"][:3]],
                "invalidation": f"pérdida del soporte {sup:g} (stop-loss justo debajo)" if sup
                else "stop-loss a 2 ATR bajo la entrada",
                "horizon_days": 20, "stop_loss": 0, "take_profit": 0})
        best = sorted(scored, key=lambda x: -scored[x]["score"])
        worst = [s for s in reversed(best) if scored[s]["score"] < -0.2][:3]
        regime_es = {"risk_on": "de apetito por el riesgo", "neutral": "neutral", "risk_off": "de aversión al riesgo"}
        fmt = lambda syms: ", ".join(f"{x} {scored[x]['score']:+.2f}" for x in syms)  # noqa: E731
        view = (f"Decisión del motor local (sin IA). Régimen {regime_es[regime]} ({regime_why}); exposición máxima "
                f"{exposure:.0%}. Más fuertes: {fmt(best[:4])}"
                + (f". Más débiles: {fmt(worst)}" if worst else "")
                + (f". Evento clave en 24 h: {market_events[0]['title']} (posiciones nuevas a mitad de tamaño)."
                   if market_events else "."))
        risks = [f"{e['date']} {e['title']}" for e in events if e["importance"] == "high"
                 and "*all" in (e.get("symbols") or [])][:4]
        watch = [f"{s}: puntuación {scored[s]['score']:+.2f}, resultados {scored[s]['binary_soon'][0]['date']} (espera)"
                 for s in best[:10] if scored[s]["binary_soon"] and s not in held][:4]
        return {"market_view": view, "regime": regime, "key_risks": risks, "positions": positions,
                "watchlist": watch, "lessons": [], "notes_for_next_time": ""}

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

    def _exit_levels(self, obs: Observation, sym: str, p: dict[str, Any] | None,
                     rep: Any) -> tuple[float | None, float | None, str]:
        """Stop-loss and take-profit for ``sym``: the model's levels when valid, else support / resistance and
        volatility (ATR) based ones; the stop never moves down (trailing). Returns (stop, target, note)."""
        price = obs.prices.get(sym, float("nan"))
        if not price > 0:
            return None, None, ""
        ind = getattr(rep, "indicators", {}) or {}
        atr = (ind.get("atr_pct") or 3.0) / 100 * price
        p = p or {}

        def num(k: str) -> float | None:
            try:
                v = float(p.get(k) or 0)
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None

        stop, target, src = num("stop_loss"), num("take_profit"), "modelo"
        if stop is None or not (price * 0.6 < stop < price - 0.3 * atr):
            sup = (getattr(rep, "support", None) or [{}])[0].get("price") if rep is not None else None
            stop = sup - 0.25 * atr if sup and 0.5 * atr <= price - sup <= 4 * atr else price - 2 * atr
            src = "soporte/ATR"
        risk = price - stop
        if target is None or not (price + 0.3 * atr < target < price * 2):
            res = (getattr(rep, "resistance", None) or [{}])[0].get("price") if rep is not None else None
            target = res - 0.1 * atr if res and res - price >= 1.5 * risk else price + 2 * risk
        current = obs.sim.get_exits(sym) if hasattr(obs.sim, "get_exits") else {}
        held = obs.position(sym) > 0  # the stop only trails within one position, not across re-entries
        old = (current.get("stop_loss") or (self.theses.get(sym) or {}).get("stop_loss")) if held else None
        note = f"stop {stop:.4g} ({src}), objetivo {target:.4g}"
        if old and old > stop:
            stop, note = old, f"stop {old:.4g} (se mantiene: nunca se baja), objetivo {target:.4g}"
            if target <= price:
                target = price + 2 * max(price - stop, 0.5 * atr)
        return round(stop, 6), round(target, 6), note

    def _orders(self, obs: Observation, targets: dict[str, float], decision: dict[str, Any],
                reports: dict[str, Any] | None = None) -> tuple[list[Any], list[str]]:
        by_sym = {p.get("symbol"): p for p in decision.get("positions", [])}
        reports = reports or {}
        actions: list[Any] = []
        changes: list[str] = []
        self._exits: dict[str, tuple[float | None, float | None]] = {}
        held = {s for s, q in obs.positions.items() if q > 0}
        for sym in sorted(held & set(targets)):
            # The price is already through the stop (the order executes on the next update): stops are never
            # lowered, so the position is closed instead of re-protected lower.
            stop = (obs.sim.get_exits(sym) if hasattr(obs.sim, "get_exits") else {}).get("stop_loss") \
                or (self.theses.get(sym) or {}).get("stop_loss")
            try:
                bid = float(obs.quote(sym).get("bid") or obs.prices.get(sym, float("inf")))
            except Exception:  # noqa: BLE001 - no quote: fall back to the last price
                bid = obs.prices.get(sym, float("inf"))
            if stop and bid <= stop:
                targets.pop(sym)
                actions.append(ClosePosition(sym))
                changes.append(f"vender {sym}: el precio está bajo su stop-loss {stop:.4g}")
        for sym in sorted(held - set(targets)):
            if any(isinstance(a, ClosePosition) and a.symbol == sym for a in actions):
                continue
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
            stop, take, note = self._exit_levels(obs, sym, by_sym.get(sym), reports.get(sym))
            self._exits[sym] = (stop, take)
            if tgt > cur + self.min_change:  # exits are attached when the purchase fills
                actions.append(OrderIntent.buy(sym, notional=round((tgt - cur) * obs.equity, 2),
                                               tag=_tag("comprar" if sym not in held else "ampliar", by_sym.get(sym)),
                                               stop_loss=stop, take_profit=take))
                changes.append(f"{'comprar' if sym not in held else 'ampliar'} {sym} {cur:.1%} -> {tgt:.1%}; {note}")
            elif sym in held and stop is not None:  # held position: (re)set its exits, raising the stop if due
                actions.append(SetExits(sym, stop, take, tag=(by_sym.get(sym) or {}).get("thesis", "")[:120]))
                prev = obs.sim.get_exits(sym) if hasattr(obs.sim, "get_exits") else {}
                if prev.get("stop_loss") != stop or prev.get("take_profit") != take:
                    changes.append(f"salidas de {sym}: {note}")
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
                "invalidation": p.get("invalidation", ""), "horizon_days": p.get("horizon_days"), "updated_at": now,
                "stop_loss": (getattr(self, "_exits", {}).get(sym) or (None, None))[0],
                "take_profit": (getattr(self, "_exits", {}).get(sym) or (None, None))[1]}
        for sym in [s for s in self.theses if s not in targets and obs.position(s) <= 0]:
            del self.theses[sym]  # never bought (or sold now): nothing to track
        known = {x.lower() for x in self.lessons}
        new_lessons = [x.strip() for x in decision.get("lessons", []) if isinstance(x, str) and x.strip()
                       and x.strip().lower() not in known]
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
        return {"method": "Discretionary analysis (technicals + macro + news + calendar) with memory",
                "engines": self.engines, "last_engine": self.last_engine, "model": self.model, "decisions": len(self.decisions), "open_theses": len(self.theses),
                "lessons": self.lessons[-8:], "last_regime": last.get("regime"), "last_view": last.get("summary"),
                "recent_reviews": self.reviews[-5:], "llm_cost_usd_total": self.usage_total.get("cost_usd_estimate"),
                "theses": self.theses}

    def state_dict(self) -> dict[str, Any]:
        return {"theses": self.theses, "decisions": self.decisions, "lessons": self.lessons, "reviews": self.reviews,
                "notes": self.notes, "usage_total": self.usage_total, "steps": self.steps,
                "last_engine": self.last_engine}

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
        self.last_engine = state.get("last_engine")

    def reset(self) -> None:
        self.steps = 0


class ClaudePortfolioManager(DiscretionaryAgent):
    """:class:`DiscretionaryAgent` on Claude only (paid; with web search). Holds when Claude is unavailable."""

    def __init__(self, *args: Any, engines: Sequence[str] = ("anthropic",), local_fallback: bool = False, **kw: Any):
        super().__init__(*args, engines=engines, local_fallback=local_fallback, **kw)


def _tag(verb: str, p: dict[str, Any] | None) -> str:
    thesis = (p or {}).get("thesis", "")
    return f"{verb}: {thesis}"[:240]


def _fmt_analysis(i: Any) -> str:
    a = i.analysis
    aff = "; ".join(f"{x['symbol']} {x['direction']}/{x['magnitude']}/{x['horizon']}" for x in a.get("affected", []))
    return (f"- [{i.published[5:16]}] {a.get('headline') or i.title} | {a.get('what_happened', '')[:400]} | "
            f"why: {a.get('why_it_matters', '')[:300]} | affects: {aff or 'n/a'} | priced in: {a.get('priced_in')} | "
            f"confidence {a.get('confidence')} | watch: {a.get('what_to_watch', '')[:200]}")
