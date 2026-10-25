"""One research refresh: news -> triage -> deep analysis, macro snapshot, calendar (+ web verification),
technical patterns of every symbol. Everything is persisted as JSON in ``ResearchConfig.dir`` so agents,
the dashboard and later runs share it.

Run it with ``python scripts/live_run.py --mode research`` (hourly in GitHub Actions) or as part of the
daily decision run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradesim.assets.universe import AssetUniverse
from tradesim.persistence import read_json, write_json
from tradesim.research.analyst import (CalendarResearcher, NewsAnalyst, apply_overrides, pick_for_deep)
from tradesim.research.calendar import CalendarBuilder, calendar_brief, upcoming
from tradesim.research.llm import LLMError, Usage, make_client
from tradesim.research.macro import MacroMonitor, macro_brief
from tradesim.research.news import NewsCollector, NewsStore
from tradesim.types import AssetClass

log = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """``research`` section of ``live/config.json``.

    Attributes:
        enabled: run research at all.
        dir: where the research files live (relative to the repository root).
        news / macro / calendar / technicals: switch each part on or off.
        analyze: use AI engines when available; without them the free local engine analyses everything.
        engines: AI engines in order of preference: ``gemini`` / ``groq`` (free tiers, need the free keys
            ``GEMINI_API_KEY`` / ``GROQ_API_KEY``) and ``anthropic`` (Claude, paid, ``ANTHROPIC_API_KEY``).
            Missing keys are skipped; when all fail the local engine takes over.
        free_models: model per free engine (defaults in ``tradesim.research.llm.FREE_PROVIDERS``).
        daily_requests: request budget per engine per UTC day (keeps free tiers within quota).
        triage_engine: ``"local"`` (free keyword engine for every item, default) or ``"llm"``.
        triage_model / deep_model / calendar_model: Claude models for each task (when ``anthropic`` is used).
        max_triage_per_run / max_deep_per_run: caps per refresh for AI engines; the backlog carries over.
        max_local_deep_per_run: detailed analyses per refresh when only the local engine is available.
        deep_min_relevance: triage relevance needed for a deep analysis (held assets need 0.2 less).
        read_articles: let the deep analyst open articles with web fetch.
        calendar_days / keep_news_days / news_max_age_days: windows.
        calendar_research_hours: minimum hours between web verifications of the calendar.
        language: language of the texts written for people.
    """

    enabled: bool = True
    dir: str = "live/research"
    news: bool = True
    macro: bool = True
    calendar: bool = True
    technicals: bool = True
    analyze: bool = True
    engines: list[str] = field(default_factory=lambda: ["gemini", "groq", "anthropic"])
    free_models: dict[str, str] = field(default_factory=dict)
    daily_requests: dict[str, int] = field(default_factory=lambda: {"gemini": 150, "groq": 400, "anthropic": 2000})
    triage_engine: str = "local"
    max_local_deep_per_run: int = 25
    triage_model: str = "claude-haiku-5-5"
    deep_model: str = "claude-sonnet-5-5"
    calendar_model: str = "claude-opus-5-5"
    max_triage_per_run: int = 300
    max_deep_per_run: int = 12
    deep_min_relevance: float = 0.6
    read_articles: bool = True
    calendar_days: int = 30
    keep_news_days: float = 7.0
    news_max_age_days: float = 3.0
    calendar_research_hours: float = 20.0
    language: str = "es"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ResearchConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ResearchStore:
    """Read/write access to the research files."""

    FILES = ("news", "macro", "calendar", "technicals", "status")

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        p = self.path(name)
        try:
            return read_json(p) if p.exists() else {}
        except (OSError, ValueError) as exc:
            log.warning("research file %s unreadable (%s); starting fresh", p, exc)
            return {}

    def save(self, name: str, data: dict[str, Any]) -> None:
        write_json(self.path(name), data)

    def news(self) -> NewsStore:
        return NewsStore.from_dict(self.load("news"))

    def exists(self) -> bool:
        return any(self.path(n).exists() for n in self.FILES)


def universe_lines(symbols: list[str], universe: AssetUniverse) -> list[str]:
    out = []
    for s in symbols:
        a = universe.get(s)
        kind = a.asset_class.value + (f"/{a.sector}" if a.sector else "")
        out.append(f"{s} ({kind}) {a.name}")
    return out


def analysis_context(store: ResearchStore, now: datetime) -> str:
    macro = store.load("macro")
    cal = store.load("calendar")
    return ("MACRO:\n" + (macro_brief(macro) or "(none)") + "\n\nCALENDAR (next 30 days, medium/high importance):\n"
            + (calendar_brief(upcoming(cal, now, 30, "medium")) or "(none)"))


def refresh_research(cfg: ResearchConfig, symbols: list[str], universe: AssetUniverse, store: ResearchStore,
                     provider: Any = None, now: datetime | None = None, client: Any = "auto",
                     holdings: set[str] | None = None) -> dict[str, Any]:
    """Run one refresh and return a status dict (also saved as ``status.json``).

    ``client`` is the Anthropic client: ``"auto"`` builds it from ``ANTHROPIC_API_KEY`` when ``anthropic`` is
    among ``cfg.engines``; ``None`` disables every AI engine (local engine only). Free engines are built from
    their keys. ``provider`` (a DataProvider) is needed for the technical patterns.
    """
    now = now or datetime.now(timezone.utc)
    t0 = time.perf_counter()
    holdings = holdings or set()
    lines = universe_lines(symbols, universe)
    engines = engine_chain(cfg, store, client)
    if client == "auto":
        client = next((b.client for b in engines["deep"] if b.name == "anthropic"), None)
    status: dict[str, Any] = {"started_at": now.isoformat(), "errors": {},
                              "llm": "enabled" if engines["deep"] else "local",
                              "engines": [b.label for b in engines["deep"]] + ["local (gratis, sin IA)"]}
    usage = Usage()

    if cfg.macro:
        mm = MacroMonitor()
        macro = mm.refresh(store.load("macro"), now)
        store.save("macro", macro)
        status["macro"] = {"indicators": len(macro["indicators"])}
        status["errors"].update({f"macro:{k}": v for k, v in mm.errors.items()})

    if cfg.calendar:
        equities = [s for s in symbols if universe.get(s).asset_class is AssetClass.EQUITY]
        previous = store.load("calendar")
        builder = CalendarBuilder(symbols, equities, days_ahead=cfg.calendar_days)
        cal = builder.build(previous, now)
        for k in ("overrides", "research_summary", "research_sources", "researched_at"):
            if previous.get(k) is not None:
                cal[k] = previous[k]
        cal = apply_overrides(cal)
        status["errors"].update({f"calendar:{k}": v for k, v in builder.errors.items()})
        due = not cal.get("researched_at") or \
            now - datetime.fromisoformat(cal["researched_at"]) >= timedelta(hours=cfg.calendar_research_hours)
        if client is not None and cfg.analyze and "anthropic" in cfg.engines and due:
            researcher = CalendarResearcher(client, lines, cfg.calendar_model, cfg.language)
            try:
                cal = researcher.research(cal, now)
                status["calendar_research"] = "done"
            except LLMError as exc:
                status["errors"]["calendar_research"] = str(exc)[:300]
            usage.merge(researcher.usage)
        store.save("calendar", cal)
        status["calendar"] = {"events": len(cal["events"]), "researched_at": cal.get("researched_at")}

    if cfg.news:
        news = store.news()
        collector = NewsCollector(symbols, [s for s in symbols if universe.get(s).asset_class is AssetClass.CRYPTO],
                                  max_age_days=cfg.news_max_age_days)
        fetched = collector.collect(now)
        added = news.merge(fetched)
        news.prune(now, cfg.keep_news_days)
        status["errors"].update({f"news:{k}": v for k, v in collector.errors.items()})
        analyst = NewsAnalyst(lines, engines["triage"], engines["deep"], cfg.language, read_articles=cfg.read_articles)
        pending = sorted((i for i in news.items.values() if i.triage is None), key=lambda i: i.published, reverse=True)
        if engines["triage"]:
            pending = pending[: cfg.max_triage_per_run]
        triaged = analyst.triage(pending, now)
        store.save("news", news.to_dict())  # keep the triage even if the deep pass fails
        limit = cfg.max_deep_per_run if engines["deep"] else cfg.max_local_deep_per_run
        chosen = pick_for_deep(news.items.values(), holdings, limit, cfg.deep_min_relevance, now=now)
        deep = analyst.deep(chosen, analysis_context(store, now), now, calendar=store.load("calendar")) if chosen else 0
        usage.merge(analyst.usage)
        status["news_engines"] = analyst.engines
        if analyst.errors:
            status["errors"]["analyst"] = "; ".join(analyst.errors)[:500]
        store.save("news", news.to_dict())
        backlog = sum(1 for i in news.items.values() if i.triage is None)
        status["news"] = {"fetched": len(fetched), "new": added, "stored": len(news.items), "triaged": triaged,
                          "deep_analyses": deep, "untriaged_backlog": backlog}

    if cfg.technicals and provider is not None:
        status["technicals"] = refresh_technicals(store, symbols, universe, provider, now, status["errors"])

    status["llm_usage"] = usage.to_dict()
    status["duration_s"] = round(time.perf_counter() - t0, 1)
    prev = store.load("status")
    totals = Usage()
    totals.by_model = (prev.get("llm_usage_total") or {}).get("by_model", {})
    totals.merge(usage)
    status["llm_usage_total"] = totals.to_dict()
    status["config"] = asdict(cfg)
    store.save("status", status)
    return status


def engine_chain(cfg: ResearchConfig, store: ResearchStore, client: Any = "auto") -> dict[str, list[Any]]:
    """AI engines per task, in preference order (empty lists = local engine only)."""
    from tradesim.research.llm import AnthropicBackend, Budget, make_backends

    if not cfg.analyze or client is None:
        return {"triage": [], "deep": []}
    budget = Budget(store.dir / "llm_budget.json", cfg.daily_requests)
    free = make_backends([e for e in cfg.engines if e != "anthropic"], cfg.free_models, budget)
    anthropic_client = make_client() if client == "auto" else client
    claude = anthropic_client is not None and "anthropic" in cfg.engines

    def chain(claude_model: str) -> list[Any]:
        out = []
        for name in cfg.engines:
            if name == "anthropic" and claude:
                out.append(AnthropicBackend(anthropic_client, claude_model, budget))
            else:
                out += [b for b in free if b.name == name]
        return out

    return {"triage": chain(cfg.triage_model) if cfg.triage_engine == "llm" else [], "deep": chain(cfg.deep_model)}


def refresh_technicals(store: ResearchStore, symbols: list[str], universe: AssetUniverse, provider: Any,
                       now: datetime, errors: dict[str, str]) -> dict[str, Any]:
    import pandas as pd

    from tradesim.data.base import DataUnavailableError
    from tradesim.live import is_market_open
    from tradesim.patterns import detect_patterns

    out: dict[str, Any] = {}
    start = now - timedelta(days=420)
    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    for sym in symbols:
        asset = universe.get(sym)
        try:
            bars = provider.get_bars(asset, "1d", start, now)
            if is_market_open(asset, now):  # today's bar is still forming: patterns use completed bars only
                bars = bars[bars.index < today]
            out[sym] = detect_patterns(bars, sym).to_dict()
        except (DataUnavailableError, KeyError, ValueError) as exc:
            errors[f"technicals:{sym}"] = str(exc)[:200]
    store.save("technicals", {"generated_at": now.isoformat(), "symbols": out})
    return {"symbols": len(out)}
