"""Claude analysts: read every news item, study the important ones in depth, and keep the calendar right.

* :class:`NewsAnalyst` works in two passes so every item is analysed without spending model requests on
  noise. Each pass runs on the configured engines in order (free Gemini / Groq tiers, Claude) and falls back
  to the free local engine (:mod:`tradesim.research.local_nlp`). **Triage** (fast model, batches of ~25) scores each item: relevance to our assets, sentiment,
  expected impact, which symbols it concerns, and a one-sentence summary. **Deep analysis** (stronger
  model, may open the article with web fetch) covers the items that matter most: what happened, why
  it matters, the transmission channel to each affected asset, horizon, whether it is priced in,
  confidence, and what to watch next (linked to calendar events).
* :class:`CalendarResearcher` (daily, with web search) confirms estimated release dates, adds market-moving
  events no feed covers (other central banks, Treasury auctions, elections, crypto events, index
  rebalances...) and records the actual results of events that already happened.

All text written for people is in Spanish (``language``); prompts are in English.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from tradesim.research.calendar import NY, Event, calendar_brief, event_id
from tradesim.research.llm import WEB_FETCH, WEB_SEARCH, LLMError, Usage, json_with_fallback, submit_call
from tradesim.research.news import NewsItem

log = logging.getLogger(__name__)

LANG = {"es": "Spanish", "en": "English"}

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "relevance": {"type": "number", "description": "0-1"},
                    "sentiment": {"type": "number", "description": "-1 (very negative) to 1 (very positive)"},
                    "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string", "enum": ["earnings", "guidance", "macro", "central_bank", "policy",
                                                            "regulation", "geopolitics", "corporate", "product",
                                                            "analyst_rating", "markets", "crypto", "commodities",
                                                            "opinion", "other"]},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                },
                "required": ["id", "relevance", "sentiment", "impact", "category", "symbols", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

DEEP_TOOL = {
    "name": "submit_analyses",
    "description": "Submit the detailed analysis of every news item in the batch (one entry per item id).",
    "input_schema": {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "headline": {"type": "string", "description": "Headline translated to the output language"},
                        "what_happened": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "affected": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "direction": {"type": "string", "enum": ["up", "down", "mixed", "unclear"]},
                                    "magnitude": {"type": "string", "enum": ["low", "medium", "high"]},
                                    "horizon": {"type": "string", "enum": ["intraday", "days", "weeks", "months"]},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["symbol", "direction", "magnitude", "horizon", "rationale"],
                            },
                        },
                        "macro_link": {"type": "string", "description": "How it relates to the macro picture"},
                        "priced_in": {"type": "string", "enum": ["likely", "partially", "unlikely", "unknown"]},
                        "confidence": {"type": "number", "description": "0-1"},
                        "what_to_watch": {"type": "string"},
                        "related_events": {"type": "array", "items": {"type": "string"},
                                           "description": "ids of calendar events this news relates to"},
                        "read_full_article": {"type": "boolean"},
                    },
                    "required": ["id", "headline", "what_happened", "why_it_matters", "affected", "macro_link",
                                 "priced_in", "confidence", "what_to_watch", "related_events", "read_full_article"],
                },
            }
        },
        "required": ["analyses"],
    },
}

CALENDAR_TOOL = {
    "name": "submit_calendar_updates",
    "description": "Submit verified corrections, missing events and actual results for the market calendar.",
    "input_schema": {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "description": "Estimated or wrong entries whose real date/time you verified",
                "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time_et": {"type": "string", "description": "HH:MM New York time, or empty"},
                    "note": {"type": "string"}, "source_url": {"type": "string"}},
                    "required": ["id", "date", "time_et", "note", "source_url"]},
            },
            "additions": {
                "type": "array",
                "description": "Market-moving events in the window that are missing from the calendar",
                "items": {"type": "object", "properties": {
                    "date": {"type": "string"}, "time_et": {"type": "string"}, "title": {"type": "string"},
                    "category": {"type": "string", "enum": ["central_bank", "inflation", "employment", "growth",
                                                            "consumer", "earnings", "dividend", "options_expiry",
                                                            "holiday", "politics", "crypto", "commodities", "other"]},
                    "importance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "country": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"}, "source_url": {"type": "string"}},
                    "required": ["date", "time_et", "title", "category", "importance", "country", "symbols", "notes",
                                 "source_url"]},
            },
            "results": {
                "type": "array",
                "description": "Actual outcomes of events from the last days that have no result yet",
                "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "actual": {"type": "string"}, "note": {"type": "string"}},
                    "required": ["id", "actual", "note"]},
            },
            "summary": {"type": "string", "description": "What to keep an eye on this month, in the output language"},
        },
        "required": ["corrections", "additions", "results", "summary"],
    },
}


def _clip(x: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


class NewsAnalyst:
    """Two-pass news analysis (see module docstring).

    ``triage_backends`` / ``deep_backends`` are engines (:mod:`tradesim.research.llm`) tried in order;
    when a list is empty or every engine fails (quota, outage) the free local engine
    (:class:`~tradesim.research.local_nlp.LocalNewsAnalyst`) does the work, so every item is always covered.
    """

    def __init__(self, universe_lines: list[str], triage_backends: list[Any] | None = None,
                 deep_backends: list[Any] | None = None, language: str = "es", triage_batch: int = 25,
                 deep_batch: int = 4, read_articles: bool = True):
        from tradesim.research.local_nlp import LocalNewsAnalyst

        self.universe_lines = universe_lines
        self.symbols = {line.split(" ", 1)[0] for line in universe_lines}
        self.triage_backends, self.deep_backends = list(triage_backends or []), list(deep_backends or [])
        self.language = LANG.get(language, language)
        self.triage_batch, self.deep_batch = triage_batch, deep_batch
        self.read_articles = read_articles
        self.local = LocalNewsAnalyst(universe_lines)
        self.usage = Usage()
        self.errors: list[str] = []
        self.engines: dict[str, int] = {}  # items analysed per engine

    def _count(self, engine: str, n: int) -> None:
        self.engines[engine] = self.engines.get(engine, 0) + n

    # ------------------------------------------------------------------ triage
    def _triage_system(self) -> str:
        return (
            "You are a buy-side news analyst for a portfolio that trades only these assets "
            "(symbol, class/sector, name):\n" + "\n".join(self.universe_lines) + "\n\n"
            "For every news item you receive, return one entry with:\n"
            "- relevance (0-1): how likely the item is to move any of these assets in the coming days/weeks. "
            "Macro, central-bank, regulation and geopolitics news can matter for many assets. Clickbait, generic "
            "listicles, personal-finance advice and stale recaps score below 0.2.\n"
            "- sentiment (-1..1) for the assets concerned; impact (low/medium/high); category.\n"
            "- symbols: the assets above that the item concerns (empty if none). Use exactly the symbols listed.\n"
            f"- summary: one factual sentence in {self.language} with the key fact and figures. No hype.\n"
            "Judge from the title and summary only; do not invent facts that are not there."
        )

    def triage(self, items: list[NewsItem], now: datetime) -> int:
        """Triage every item (LLM batches when available, the local engine otherwise)."""
        done = 0
        for k in range(0, len(items), self.triage_batch):
            batch = items[k:k + self.triage_batch]
            out, engine = None, "local"
            if self.triage_backends:
                lines = [json.dumps({"id": i.id, "published": i.published[:16], "source": i.publisher or i.source,
                                     "feed": i.symbols or i.topics, "title": i.title, "summary": i.summary[:400]},
                                    ensure_ascii=False) for i in batch]
                try:
                    out, b = json_with_fallback(self.triage_backends, self._triage_system(), "\n".join(lines),
                                                TRIAGE_SCHEMA, self.usage, max_tokens=12000, effort="low",
                                                errors=self.errors)
                    engine = f"{b.name}:{b.model}"
                except LLMError:
                    out = None
            by_id = {i.id: i for i in batch}
            if out is not None:
                for row in out.get("items", []):
                    item = by_id.pop(row.get("id"), None)
                    if item is None:
                        continue
                    item.triage = {
                        "relevance": round(_clip(row.get("relevance"), 0, 1), 3),
                        "sentiment": round(_clip(row.get("sentiment"), -1, 1), 3),
                        "impact": row.get("impact") if row.get("impact") in ("low", "medium", "high") else "low",
                        "category": row.get("category", "other"),
                        "symbols": [s for s in row.get("symbols", []) if s in self.symbols],
                        "summary": str(row.get("summary", ""))[:500],
                        "model": engine, "at": now.isoformat(),
                    }
                    done += 1
                self._count(engine, len(batch) - len(by_id))
            if by_id:  # no engine, or items the model skipped
                done += self.local.triage(list(by_id.values()), now)
                self._count("local", len(by_id))
        return done

    # ------------------------------------------------------------------ deep analysis
    def _deep_system(self, context: str) -> str:
        return (
            "You are a senior cross-asset strategist. You analyse news for a portfolio that trades only these "
            "assets:\n" + "\n".join(self.universe_lines) + "\n\n"
            f"Current context (macro data and the event calendar):\n{context}\n\n"
            "For each news item: explain what happened (facts and figures), why it matters, and for each affected "
            "asset the direction, magnitude, horizon and transmission channel (earnings, rates, dollar, risk "
            "appetite, flows, regulation...). Say whether it is likely priced in, your confidence (0-1), what to "
            "watch next and which calendar events (by id) it relates to. Be specific and sceptical: separate facts "
            "from speculation, flag opinion pieces, and do not overstate small news."
            + (" When the title and summary are not enough to judge an important item, open the article with "
               "web_fetch (at most once per item) and set read_full_article accordingly." if self.read_articles else "")
            + f"\nWrite every text field in {self.language}. Finish by calling submit_analyses once with all items."
        )

    def _store(self, item: NewsItem, row: dict[str, Any], engine: str, now: datetime) -> None:
        row = dict(row)
        row["affected"] = [a for a in row.get("affected", []) if isinstance(a, dict) and a.get("symbol") in self.symbols]
        row["confidence"] = round(_clip(row.get("confidence"), 0, 1, 0.5), 2)
        row["model"], row["at"] = engine, now.isoformat()
        item.analysis = row

    def deep(self, items: list[NewsItem], context: str, now: datetime, calendar: dict[str, Any] | None = None) -> int:
        """Detailed analysis of ``items``: LLM engines in order, the local engine for what they leave."""
        done = 0
        system = self._deep_system(context)
        for k in range(0, len(items), self.deep_batch):
            batch = items[k:k + self.deep_batch]
            payload = "\n\n".join(json.dumps({
                "id": i.id, "published": i.published, "source": i.publisher or i.source, "url": i.url,
                "title": i.title, "summary": i.summary, "triage": i.triage}, ensure_ascii=False) for i in batch)
            user = f"Analyse these {len(batch)} news items:\n\n{payload}"
            by_id = {i.id: i for i in batch}
            for b in self.deep_backends:
                try:
                    if b.name == "anthropic":  # Claude can open the articles (web fetch)
                        if getattr(b, "budget", None):
                            b.budget.take(b.name)
                        tools = [{**WEB_FETCH, "max_uses": len(batch), "max_content_tokens": 6000}] \
                            if self.read_articles else []
                        out, _ = submit_call(b.client, b.model, system, user, DEEP_TOOL, self.usage,
                                             server_tools=tools, max_tokens=24000, effort="medium")
                    else:
                        out = b.json(system.replace("submit_analyses", "the JSON answer"), user,
                                     DEEP_TOOL["input_schema"], self.usage, max_tokens=8000)
                except LLMError as exc:
                    self.errors.append(f"deep {b.name}: {exc}"[:240])
                    log.warning("deep analysis with %s failed: %s", b.name, exc)
                    continue
                engine = f"{b.name}:{b.model}"
                for row in out.get("analyses", []):
                    item = by_id.pop(row.get("id"), None)
                    if item is not None:
                        self._store(item, row, engine, now)
                        done += 1
                        self._count(engine, 1)
                break
            for item in by_id.values():  # no engine answered: local analysis
                item.analysis = self.local.analyse_item(item, now, calendar)
                done += 1
                self._count("local", 1)
        return done


def pick_for_deep(items: Iterable[NewsItem], holdings: set[str], limit: int, min_relevance: float = 0.6,
                  max_age_hours: float = 48, now: datetime | None = None) -> list[NewsItem]:
    """Triaged, not yet analysed items worth a deep look, most important first."""
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=max_age_hours)).isoformat()
    scored = []
    for i in items:
        t = i.triage
        if not t or i.analysis is not None or i.published < cutoff:
            continue
        held = bool(holdings & set(t.get("symbols") or []))
        score = t["relevance"] + (0.25 if t["impact"] == "high" else 0.1 if t["impact"] == "medium" else 0) \
            + (0.2 if held else 0)
        if t["relevance"] >= min_relevance or (held and t["relevance"] >= min_relevance - 0.2) or t["impact"] == "high":
            scored.append((score, i.published, i))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [i for _, _, i in scored[:limit]]


# --------------------------------------------------------------------------- calendar
class CalendarResearcher:
    """Daily web research that verifies and completes the calendar (see module docstring)."""

    def __init__(self, client: Any, universe_lines: list[str], model: str = "claude-opus-5-5", language: str = "es",
                 max_searches: int = 10):
        self.client = client
        self.universe_lines = universe_lines
        self.model = model
        self.language = LANG.get(language, language)
        self.max_searches = max_searches
        self.usage = Usage()

    def research(self, cal: dict[str, Any], now: datetime) -> dict[str, Any]:
        """Returns the calendar with the updates applied (and ``researched_at`` / ``research_summary``)."""
        today = now.astimezone(NY).date()
        events = cal.get("events", [])
        future = [e for e in events if e["date"] >= today.isoformat()]
        past = [e for e in events if e["date"] < today.isoformat() and not e.get("actual")
                and e["importance"] in ("high", "medium") and e["category"] not in ("holiday", "dividend")]
        system = (
            "You maintain the event calendar of a portfolio that trades:\n" + "\n".join(self.universe_lines) +
            f"\n\nToday is {today.isoformat()} ({now.astimezone(NY):%A}, New York). The calendar window ends "
            f"{cal.get('window', ['', ''])[1]}. Use web search to:\n"
            "1. Verify the entries marked 'estimated date' against official schedules (BLS, BEA, Census, ISM, "
            "University of Michigan, company investor-relations pages) and correct date/time when they differ.\n"
            "2. Add market-moving events in the window that are missing: other central-bank decisions (ECB, BoJ, "
            "BoE, PBoC) not yet listed, US Treasury refunding and large auctions, elections and major political "
            "deadlines, OPEC+ meetings, earnings of the listed companies not yet in the calendar, crypto-specific "
            "events (ETF decisions, network upgrades, large token unlocks, regulation hearings), index "
            "rebalances. Only events with a known or officially announced date; no speculation.\n"
            "3. For the past events listed without a result, give the actual figure/outcome and one line of "
            "context (vs forecast).\n"
            f"Keep notes short and factual, in {self.language}. Cite a source URL for each correction and addition. "
            "Finish by calling submit_calendar_updates exactly once."
        )
        user = ("Upcoming calendar:\n" + calendar_brief(future) + "\n\nPast events without results:\n"
                + (calendar_brief(past) or "(none)"))
        out, sources = submit_call(self.client, self.model, system, user, CALENDAR_TOOL, self.usage,
                                   server_tools=[{**WEB_SEARCH, "max_uses": self.max_searches}], max_tokens=32000,
                                   effort="medium")
        return apply_calendar_updates(cal, out, now, sources)


def _when(day: str, hhmm: str) -> str | None:
    try:
        h, m = (int(x) for x in hhmm.split(":")[:2])
        return datetime.combine(datetime.fromisoformat(day).date(), datetime.min.time().replace(hour=h, minute=m),
                                NY).astimezone(timezone.utc).isoformat()
    except (ValueError, AttributeError):
        return None


def apply_calendar_updates(cal: dict[str, Any], out: dict[str, Any], now: datetime,
                           sources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cal = json.loads(json.dumps(cal))
    events = {e["id"]: e for e in cal.get("events", [])}
    overrides = cal.setdefault("overrides", {})
    for c in out.get("corrections", []):
        e = events.get(c.get("id"))
        if e is None or not c.get("date"):
            continue
        key = override_key(e)
        e["date"] = c["date"]
        e["when"] = _when(c["date"], c.get("time_et", "")) or e.get("when")
        e["confirmed"] = True
        e["notes"] = f"verified: {c.get('note', '')}".strip()
        if e.get("source") == "rules":
            overrides[key] = {"date": e["date"], "when": e["when"], "note": e["notes"], "url": c.get("source_url", "")}
    for a in out.get("additions", []):
        if not a.get("date") or not a.get("title"):
            continue
        ev = Event(a["date"], a["title"], a.get("category", "other"), a.get("importance", "medium"),
                   a.get("country", "US"), _when(a["date"], a.get("time_et", "")), list(a.get("symbols") or []),
                   source="claude-web", notes=(a.get("notes", "") + (f" ({a['source_url']})" if a.get("source_url") else "")))
        if ev.id not in events:
            events[ev.id] = ev.to_dict()
    for r in out.get("results", []):
        e = events.get(r.get("id"))
        if e is not None and r.get("actual"):
            e["actual"] = r["actual"]
            if r.get("note"):
                e["notes"] = (e.get("notes", "") + " | " + r["note"]).strip(" |")
    cal["events"] = sorted(events.values(), key=lambda e: (e["date"], e.get("when") or "99"))
    cal["researched_at"] = now.isoformat()
    cal["research_summary"] = out.get("summary", "")
    cal["research_sources"] = (sources or [])[:30]
    return cal


def override_key(e: dict[str, Any]) -> str:
    """Identifies a rule-based release across rebuilds (title + month of the estimated date)."""
    return f"{e['title']}|{e['date'][:7]}"


def apply_overrides(cal: dict[str, Any]) -> dict[str, Any]:
    """Re-apply researcher corrections to freshly rebuilt rule-based entries."""
    ov = cal.get("overrides") or {}
    if not ov:
        return cal
    for e in cal.get("events", []):
        o = ov.get(override_key(e)) if e.get("source") == "rules" else None
        if o:
            e.update(date=o["date"], when=o.get("when") or e.get("when"), confirmed=True, notes=o.get("note", ""))
            e["id"] = event_id(e["date"], e["title"], e["country"])
    cal["events"].sort(key=lambda e: (e["date"], e.get("when") or "99"))
    return cal
