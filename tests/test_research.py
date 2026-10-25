"""Research layer and the discretionary agent, fully offline (fake feeds and a fake Anthropic client)."""

import json
import re
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator, run_episode
from tradesim.agents.discretionary import ClaudePortfolioManager
from tradesim.assets.universe import AssetUniverse
from tradesim.research import analyst as analyst_mod
from tradesim.research import calendar as cal_mod
from tradesim.research import macro as macro_mod
from tradesim.research import news as news_mod
from tradesim.research.analyst import NewsAnalyst, apply_calendar_updates, apply_overrides, pick_for_deep
from tradesim.research.calendar import (CalendarBuilder, Event, jobs_report_date, merge_events, nth_weekday,
                                        parse_fomc_page, parse_forex_factory)
from tradesim.research.llm import Usage
from tradesim.research.macro import MacroMonitor, asset_profile, relevant_indicators
from tradesim.research.news import NewsCollector, NewsItem, NewsStore, item_id, parse_rss
from tradesim.research.refresh import ResearchConfig, ResearchStore, refresh_research

NOW = datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc)  # a Monday


# --------------------------------------------------------------------------- fixtures
def rss(items):
    from html import escape

    body = "".join(f"<item><title>{escape(t)}</title><link>https://ex.com/{i}</link><description>{escape(d)}</description>"
                   f"<pubDate>{p}</pubDate></item>" for i, (t, d, p) in enumerate(items))
    return f'﻿<?xml version="1.0"?><rss version="2.0"><channel>{body}</channel></rss>'


FOMC_PAGE = """
<h4>2026 FOMC Meetings</h4>
<div class="fomc-meeting__month col-xs-5"><strong>October</strong></div><div class="fomc-meeting__date col-xs-4">27-28</div>
<div class="fomc-meeting__month col-xs-5"><strong>December</strong></div><div class="fomc-meeting__date col-xs-4">8-9*</div>
<h4>2027 FOMC Meetings</h4>
<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div><div class="fomc-meeting__date col-xs-4">26-27</div>
"""

FF_WEEK = json.dumps([
    {"title": "CPI m/m", "country": "USD", "date": "2026-10-29T08:30:00-04:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.4%"},
    {"title": "FOMC Statement", "country": "USD", "date": "2026-10-28T14:00:00-04:00", "impact": "High"},
    {"title": "Federal Funds Rate", "country": "USD", "date": "2026-10-28T14:00:00-04:00", "impact": "High",
     "forecast": "4.00%", "previous": "4.00%"},
    {"title": "German Ifo", "country": "EUR", "date": "2026-10-26T05:00:00-04:00", "impact": "Low"},
    {"title": "Main Refinancing Rate", "country": "EUR", "date": "2026-10-29T09:15:00-04:00", "impact": "High"},
])


def fred_csv(n=30, start=100.0, step=1.0, freq="MS"):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return "observation_date,X\n" + "\n".join(f"{d.date()},{start + k * step}" for k, d in enumerate(idx))


def fake_fetch(url):
    if "fomccalendars" in url:
        return FOMC_PAGE
    if "faireconomy" in url:
        return FF_WEEK
    if "fred.stlouisfed.org" in url:
        return fred_csv()
    if "finance.yahoo.com" in url:
        sym = re.search(r"s=([^&]+)", url).group(1)
        return rss([(f"{sym} beats estimates as demand surges", "Revenue up 20%.", "Mon, 26 Oct 2026 11:00:00 +0000"),
                    ("Markets rally ahead of the Fed", "Stocks rose.", "Mon, 26 Oct 2026 10:00:00 +0000")])
    if "news.google.com" in url:
        return rss([("Fed expected to hold rates - Reuters", "<a>x</a>", "Mon, 26 Oct 2026 09:00:00 GMT")])
    if "coindesk" in url or "federalreserve.gov/feeds" in url:
        return rss([("Bitcoin ETF inflows hit record", "Inflows of $1B.", "Mon, 26 Oct 2026 08:00:00 +0000")])
    raise AssertionError(f"unexpected url {url}")


def fake_yahoo(tickers):
    idx = pd.date_range("2026-07-01", periods=60, freq="B")
    return {t: pd.Series([10.0 + k * 0.1 for k in range(60)], index=idx) for t in tickers}


def fake_company_calendar(sym):
    return {"Earnings Date": [date(2026, 10, 29)], "Earnings Average": 1.5, "Revenue Average": 1e10} if sym == "AAPL" else {}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(news_mod, "fetch_text", lambda url, **kw: fake_fetch(url))
    monkeypatch.setattr(cal_mod, "fetch_text", lambda url, **kw: fake_fetch(url))
    monkeypatch.setattr(macro_mod, "fetch_text", lambda url, **kw: fake_fetch(url))
    monkeypatch.setattr(macro_mod, "_yahoo_closes", fake_yahoo)
    monkeypatch.setattr(cal_mod, "_yahoo_calendar", fake_company_calendar)


# ------------------------------------------------------------------ fake Anthropic client
class FakeClient:
    """Answers like the Messages API for the four kinds of request the code makes."""

    def __init__(self, decision=None, fail=False):
        self.calls = []
        self.decision = decision
        self.fail = fail
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        return _Stream(self._respond(kw))

    def _respond(self, kw):
        usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0, server_tool_use=None)
        user = kw["messages"][0]["content"]
        if "format" in kw.get("output_config", {}):  # triage
            ids = re.findall(r'"id": "([0-9a-f]{16})"', user)
            items = [{"id": i, "relevance": 0.8, "sentiment": 0.5, "impact": "high", "category": "earnings",
                      "symbols": ["AAPL", "NOPE"], "summary": "Resumen."} for i in ids]
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"items": items}))],
                                   stop_reason="end_turn", usage=usage)
        tool = kw["tools"][-1]["name"]
        if tool == "submit_analyses":
            ids = re.findall(r'"id": "([0-9a-f]{16})"', user)
            inp = {"analyses": [{"id": i, "headline": "Titular", "what_happened": "Pasó algo.", "why_it_matters": "Importa.",
                                 "affected": [{"symbol": "AAPL", "direction": "up", "magnitude": "medium",
                                               "horizon": "days", "rationale": "Ventas."},
                                              {"symbol": "ZZZ", "direction": "up", "magnitude": "low",
                                               "horizon": "days", "rationale": "x"}],
                                 "macro_link": "", "priced_in": "partially", "confidence": 1.7, "what_to_watch": "",
                                 "related_events": [], "read_full_article": False} for i in ids]}
        elif tool == "submit_calendar_updates":
            eid = re.search(r"\[([0-9a-f]{12})\] \d{4}-\d{2}-\d{2}[^\n]*Employment Situation", user).group(1)
            inp = {"corrections": [{"id": eid, "date": "2026-11-06", "time_et": "08:30", "note": "BLS schedule",
                                    "source_url": "https://bls.gov"}],
                   "additions": [{"date": "2026-11-03", "time_et": "", "title": "US midterm elections",
                                  "category": "politics", "importance": "high", "country": "US", "symbols": ["*all"],
                                  "notes": "Elecciones de mitad de mandato.", "source_url": "https://ex.com"}],
                   "results": [], "summary": "Mes cargado: Fed, empleo y elecciones."}
        elif tool == "submit_decision":
            if self.fail:
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="thinking aloud")],
                                       stop_reason="end_turn", usage=usage)
            inp = self.decision
        else:
            raise AssertionError(tool)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=tool, input=inp, id="t1")],
                               stop_reason="tool_use", usage=usage)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


# ------------------------------------------------------------------------------ news
def test_parse_rss_and_dedupe():
    items = parse_rss(rss([("Apple beats", "<b>Strong</b> quarter", "Mon, 26 Oct 2026 11:00:00 +0000")]), "yahoo",
                      symbols=["AAPL"])
    assert items[0].summary == "Strong quarter" and items[0].published.startswith("2026-10-26T11:00")
    store = NewsStore(items)
    dup = NewsItem(item_id("Apple beats"), "Apple beats", "u", items[0].published, "yahoo", symbols=["QQQ"])
    assert store.add(dup) is False and store.items[dup.id].symbols == ["AAPL", "QQQ"]
    g = parse_rss(rss([("Fed holds - Reuters", "x", "Mon, 26 Oct 2026 09:00:00 GMT")]), "google-news")
    assert g[0].title == "Fed holds" and g[0].publisher == "Reuters" and g[0].summary == ""


def test_news_collector_with_fake_feeds():
    nc = NewsCollector(["AAPL", "BTC-USD"], ["BTC-USD"], fetch=fake_fetch, max_age_days=3)
    store = NewsStore()
    store.merge(nc.collect(NOW))
    shared = [i for i in store.items.values() if i.title.startswith("Markets rally")]
    assert len(shared) == 1 and set(shared[0].symbols) == {"AAPL", "BTC-USD"}
    assert not nc.errors
    store.prune(NOW + timedelta(days=10), keep_days=7)
    assert not store.items


# ----------------------------------------------------------------------------- macro
def test_macro_snapshot_and_relevance():
    m = MacroMonitor(fetch=fake_fetch, yahoo=fake_yahoo).refresh(now=NOW)
    ind = m["indicators"]
    assert ind["cpi_yoy"]["value"] == pytest.approx((129 / 117 - 1) * 100, rel=1e-3)  # yoy on monthly data
    assert ind["payrolls_change"]["value"] == 1.0 and ind["vix"]["change_1m_pct"] is not None
    # cached official data is reused within the TTL; failures keep stale values
    m2 = MacroMonitor(fetch=lambda u: (_ for _ in ()).throw(RuntimeError("down")), yahoo=fake_yahoo).refresh(m, NOW)
    assert m2["indicators"]["cpi_yoy"]["value"] == ind["cpi_yoy"]["value"]
    u = AssetUniverse.default()
    assert asset_profile(u.get("TLT")) == "bonds" and "real_10y" in relevant_indicators(u.get("GLD"))
    assert asset_profile(u.get("BTC-USD")) == "crypto" and asset_profile(u.get("XOM")) == "Energy"


# --------------------------------------------------------------------------- calendar
def test_calendar_rules():
    assert jobs_report_date(2025, 8) == date(2025, 8, 1) and jobs_report_date(2025, 11) == date(2025, 11, 7)
    assert nth_weekday(2026, 10, 4, 3) == date(2026, 10, 16)  # monthly options expiry
    sched = parse_fomc_page(FOMC_PAGE)
    assert sched[2026] == [(10, 27, 28, False), (12, 8, 9, True)] and sched[2027][0] == (1, 26, 27, False)
    ff = parse_forex_factory(FF_WEEK)
    titles = {e.title for e in ff}
    assert "FOMC Statement" not in titles and "German Ifo" not in titles and "Main Refinancing Rate" in titles
    assert {e.country for e in ff} == {"US", "EU"}


def test_calendar_builder_merges_sources():
    b = CalendarBuilder(["AAPL", "SPY"], ["AAPL"], fetch=fake_fetch, calendar_fn=fake_company_calendar)
    cal = b.build(now=NOW)
    ev = cal["events"]
    titles = [e["title"] for e in ev]
    assert any(t.startswith("FOMC rate decision") for t in titles)
    assert "AAPL quarterly earnings" in titles
    assert "CPI m/m" in titles and "CPI (consumer prices)" not in [e["title"] for e in ev if e["date"] < "2026-11-01"]
    nfp = [e for e in ev if e["title"].startswith("Employment Situation")]
    assert nfp and nfp[0]["confirmed"] is False and nfp[0]["date"] == "2026-11-06"
    assert all(cal["window"][0] <= e["date"] <= cal["window"][1] for e in ev)
    # Forex Factory rows survive the next week's rebuild (the feed only covers the current week)
    b2 = CalendarBuilder(["AAPL"], ["AAPL"], fetch=lambda u: "[]" if "faireconomy" in u else fake_fetch(u),
                         calendar_fn=fake_company_calendar)
    cal2 = b2.build(cal, NOW + timedelta(hours=1))
    assert "CPI m/m" in [e["title"] for e in cal2["events"]]


def test_merge_prefers_confirmed_sources():
    est = Event("2026-10-28", "PCE inflation / personal income and outlays", "inflation", source="rules", confirmed=False)
    real = Event("2026-10-29", "Core PCE Price Index m/m", "inflation", "high", source="forexfactory")
    out = merge_events([est, real])
    assert [e.title for e in out] == ["Core PCE Price Index m/m"]


def test_calendar_research_updates_and_overrides(offline):
    b = CalendarBuilder(["AAPL"], ["AAPL"], fetch=fake_fetch, calendar_fn=fake_company_calendar)
    cal = b.build(now=NOW)
    researcher = analyst_mod.CalendarResearcher(FakeClient(), ["AAPL (equity) Apple"])
    cal = researcher.research(cal, NOW)
    nfp = next(e for e in cal["events"] if e["title"].startswith("Employment Situation"))
    assert nfp["confirmed"] and nfp["when"].startswith("2026-11-06T13:30")
    assert any(e["title"] == "US midterm elections" and e["source"] == "claude-web" for e in cal["events"])
    assert cal["research_summary"] and cal["overrides"]
    # a rebuild regenerates the estimated entry; the stored override re-applies the verified date
    rebuilt = b.build(cal, NOW + timedelta(hours=2))
    rebuilt["overrides"] = cal["overrides"]
    rebuilt = apply_overrides(rebuilt)
    nfp2 = next(e for e in rebuilt["events"] if e["title"].startswith("Employment Situation"))
    assert nfp2["confirmed"] and nfp2["date"] == "2026-11-06"
    assert any(e["title"] == "US midterm elections" for e in rebuilt["events"])


# ---------------------------------------------------------------------------- analyst
def test_news_triage_and_deep_analysis():
    from tradesim.research.llm import AnthropicBackend

    items = parse_rss(rss([("Apple beats", "Strong quarter", "Mon, 26 Oct 2026 11:00:00 +0000"),
                           ("Listicle: 5 stocks", "", "Mon, 26 Oct 2026 10:00:00 +0000")]), "yahoo", ["AAPL"])
    client = FakeClient()
    claude = AnthropicBackend(client, "claude-haiku-5-5")
    a = NewsAnalyst(["AAPL (equity/IT) Apple", "SPY (etf) S&P 500"], triage_backends=[claude], deep_backends=[claude])
    assert a.triage(items, NOW) == 2
    assert items[0].triage["symbols"] == ["AAPL"] and items[0].triage["model"].startswith("anthropic")
    chosen = pick_for_deep(items, {"AAPL"}, limit=1, now=NOW)
    assert len(chosen) == 1
    assert a.deep(chosen, "context", NOW) == 1
    an = chosen[0].analysis
    assert an["confidence"] == 1.0 and [x["symbol"] for x in an["affected"]] == ["AAPL"]
    assert a.usage.cost_usd() > 0
    deep_call = client.calls[-1]
    assert deep_call["tools"][0]["type"] == "web_fetch_20260209" and deep_call["tools"][-1]["eager_input_streaming"]


def test_local_engine_triages_and_analyses_without_any_key():
    from tradesim.research.local_nlp import LocalNewsAnalyst

    lines = ["AAPL (equity/IT) Apple Inc.", "XOM (equity/Energy) ExxonMobil", "BTC-USD (crypto) Bitcoin",
             "ETH-USD (crypto) Ethereum", "SPY (etf) SPDR S&P 500 ETF Trust"]
    la = LocalNewsAnalyst(lines)
    mk = lambda t, s="", syms=(): NewsItem(item_id(t), t, "u", NOW.isoformat(), "yahoo", s, "Reuters", list(syms))
    good = la.triage_item(mk("Apple beats estimates and raises guidance as iPhone sales surge"), NOW)
    assert good["symbols"] == ["AAPL"] and good["sentiment"] > 0.5 and good["category"] == "earnings"
    bad = la.triage_item(mk("Exxon cuts guidance after refinery outage; shares plunge"), NOW)
    assert bad["symbols"] == ["XOM"] and bad["sentiment"] < -0.5
    neg = la.triage_item(mk("Apple did not miss estimates"), NOW)
    assert neg["sentiment"] >= 0  # negation flips "miss"
    bait = la.triage_item(mk("3 High Quality Stocks to Buy Now and Hold Forever", syms=["AAPL"]), NOW)
    assert bait["category"] == "opinion" and bait["relevance"] < 0.3
    crypto = la.triage_item(mk("South Korea makes stablecoins an official means of payment"), NOW)
    assert set(crypto["symbols"]) == {"BTC-USD", "ETH-USD"}
    fed = la.triage_item(mk("Fed signals rate hike as inflation runs hotter than expected"), NOW)
    assert fed["category"] == "central_bank" and fed["relevance"] >= 0.6 and fed["sentiment"] < 0
    # every item analysed; deep pass falls back to the local engine when no AI engine answers
    items = [mk("Apple beats estimates", "Revenue up 20%.", ["AAPL"]), mk("Fed holds rates")]
    a = NewsAnalyst(lines, deep_backends=[])
    assert a.triage(items, NOW) == 2 and all(i.triage["model"] == "local" for i in items)
    cal = {"events": [{"id": "e1", "date": "2026-10-29", "symbols": ["AAPL"], "category": "earnings"}]}
    assert a.deep(items, "", NOW, cal) == 2
    assert items[0].analysis["model"] == "local" and items[0].analysis["related_events"] == ["e1"]
    assert items[0].analysis["why_it_matters"] and a.engines == {"local": 4}


def test_free_engine_backend_budget_and_fallback(tmp_path):
    from tradesim.research.llm import (Budget, LLMError, OpenAICompatBackend, QuotaError, json_with_fallback,
                                       parse_json_text)

    calls = []

    def post(url, headers, payload, timeout):
        calls.append((url, headers, payload))
        if "groq" in url:
            return 429, {"error": {"message": "rate limited"}}
        return 200, {"choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    budget = Budget(tmp_path / "budget.json", {"gemini": 2, "groq": 5})
    groq = OpenAICompatBackend("groq", "k1", budget=budget, post=post)
    gemini = OpenAICompatBackend("gemini", "k2", budget=budget, post=post)
    usage, errors = Usage(), []
    out, used = json_with_fallback([groq, gemini], "sys", "user", {"type": "object"}, usage, errors=errors)
    assert out == {"ok": True} and used is gemini and "rate limit" in errors[0]
    url, headers, payload = calls[-1]
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert headers["Authorization"] == "Bearer k2" and payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "gemini-flash-lite-latest" and "JSON schema" in payload["messages"][0]["content"]
    assert usage.cost_usd() == 0  # free tiers cost nothing
    gemini.json("s", "u", {}, usage)  # second request of the day: still within budget
    with pytest.raises(QuotaError):
        gemini.json("s", "u", {}, usage)  # budget shared with the file: a third one is refused
    assert Budget(tmp_path / "budget.json", {"gemini": 2}).used("gemini") == 2
    with pytest.raises(LLMError):
        json_with_fallback([groq], "s", "u", {}, usage)
    assert parse_json_text('Sure! {"a": 1} hope it helps') == {"a": 1}


def test_refresh_research_end_to_end(tmp_path, offline):
    store = ResearchStore(tmp_path / "research")
    cfg = ResearchConfig(technicals=False)
    st = refresh_research(cfg, ["AAPL", "BTC-USD"], AssetUniverse.default(), store, now=NOW, client=FakeClient())
    assert st["news"]["stored"] > 0 and st["news"]["triaged"] == st["news"]["stored"]
    assert st["engines"][0].startswith("Claude") and st["news_engines"].get("local") == st["news"]["stored"]
    assert st["news"]["deep_analyses"] > 0 and st["calendar_research"] == "done"
    assert st["macro"]["indicators"] > 10 and st["llm_usage"]["cost_usd_estimate"] > 0
    assert {p.stem for p in (tmp_path / "research").iterdir()} >= {"news", "macro", "calendar", "status"}
    # second run: nothing new to analyse, calendar research not due yet
    st2 = refresh_research(cfg, ["AAPL", "BTC-USD"], AssetUniverse.default(), store, now=NOW + timedelta(hours=1),
                           client=FakeClient())
    assert st2["news"]["triaged"] == 0 and "calendar_research" not in st2
    assert st2["llm_usage_total"]["cost_usd_estimate"] >= st["llm_usage"]["cost_usd_estimate"]
    # without a client the data is still collected
    st3 = refresh_research(cfg, ["AAPL"], AssetUniverse.default(), ResearchStore(tmp_path / "r2"), now=NOW, client=None)
    assert st3["llm"] == "local" and st3["news"]["untriaged_backlog"] == 0 and st3["news"]["deep_analyses"] > 0


# ------------------------------------------------------------------- discretionary agent
def _sim():
    return TradingSimulator(provider=SyntheticProvider(seed=4), config=SimulatorConfig(initial_cash=10_000),
                            symbols=["SPY", "AAPL", "GLD", "BTC-USD"], start="2022-01-01", end="2023-06-30")


DECISION = {
    "market_view": "Tendencia alcista con riesgos de tipos.", "regime": "risk_on", "key_risks": ["FOMC"],
    "positions": [
        {"symbol": "AAPL", "target_weight": 0.30, "conviction": "high", "thesis": "Resultados fuertes y ruptura.",
         "catalysts": ["earnings"], "invalidation": "cierre bajo 150", "horizon_days": 20},
        {"symbol": "GLD", "target_weight": 8, "conviction": "medium", "thesis": "Cobertura.", "catalysts": [],
         "invalidation": "dólar fuerte", "horizon_days": 30},
        {"symbol": "FAKE", "target_weight": 0.1, "conviction": "low", "thesis": "x", "catalysts": [],
         "invalidation": "x", "horizon_days": 5},
    ],
    "watchlist": ["BTC-USD si rompe resistencia"], "lessons": ["No perseguir gaps."], "notes_for_next_time": "Vigilar CPI.",
}


def test_discretionary_agent_decides_and_remembers(tmp_path, offline):
    store = ResearchStore(tmp_path / "research")
    refresh_research(ResearchConfig(technicals=False), ["SPY", "AAPL", "GLD", "BTC-USD"], AssetUniverse.default(),
                     store, now=NOW, client=FakeClient())
    client = FakeClient(decision=DECISION)
    agent = ClaudePortfolioManager(client=client, max_weight=0.15, every=5)
    agent.attach_research(store)
    env = _sim()
    res = run_episode(agent, env, seed=0, episode_length=12)
    assert res.metrics["num_fills"] >= 2
    weights = {p["symbol"]: p["weight"] for p in env.get_portfolio()["positions"]}
    assert set(weights) == {"AAPL", "GLD"} and weights["AAPL"] <= 0.16  # capped at max_weight; % -> fraction
    prompt = client.calls[0]["messages"][0]["content"]
    for section in ("TECHNICALS", "MACRO DATA", "EVENT CALENDAR", "NEWS ANALYSES", "OPEN POSITIONS"):
        assert section in prompt
    assert "ignored FAKE: not in the tradable universe" in agent.explain()["adjustments"]
    assert agent.explain()["status"] == "ok" and agent.decisions and agent.lessons == []  # unverified free-text lessons are not kept
    assert {t["type"] for t in client.calls[0]["tools"][:-1]} == {"web_search_20260209", "web_fetch_20260209"}
    # the second decision sees its theses, the review and its notes
    later = client.calls[-1]["messages"][0]["content"]
    assert "Resultados fuertes" in later and "YOUR TRACK RECORD" in later and "Vigilar CPI." in later
    restored = ClaudePortfolioManager()
    restored.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    assert restored.theses.keys() == agent.theses.keys() and restored.usage_total["cost_usd_estimate"] > 0
    env.close()


def test_discretionary_agent_without_key_or_answer_holds(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    env = _sim()
    agent = ClaudePortfolioManager()
    res = run_episode(agent, env, seed=0, episode_length=3)
    assert res.metrics["num_fills"] == 0 and agent.explain()["status"] == "inactive"
    agent2 = ClaudePortfolioManager(client=FakeClient(fail=True))
    run_episode(agent2, env, seed=0, episode_length=2)
    assert agent2.explain()["status"] == "error" and "submit_decision" in agent2.explain()["message"]
    env.close()


def test_usage_cost():
    u = Usage()
    u.add("claude-opus-5-5", SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=0,
                                             cache_creation_input_tokens=0,
                                             server_tool_use=SimpleNamespace(web_search_requests=10)))
    assert u.cost_usd() == pytest.approx(4 + 2 + 0.1)


def test_runner_research_mode_exports_dashboard(tmp_path, offline):
    from tradesim.trading import AccountSpec, LiveConfig, LiveRunner

    cfg = LiveConfig(symbols=["AAPL", "BTC-USD"], accounts=[AccountSpec("analista", "Analista", "ClaudePortfolioManager")],
                     state_dir=str(tmp_path / "accounts"), runs_log=str(tmp_path / "runs.jsonl"),
                     dashboard_file=str(tmp_path / "live.json"),
                     research={"dir": str(tmp_path / "research"), "technicals": False})
    runner = LiveRunner(cfg, provider=SyntheticProvider(seed=1), clock=lambda: NOW, llm_client=FakeClient())
    rec = runner.run("research")
    assert rec["mode"] == "research" and rec["research"]["llm"] == "enabled"
    assert rec["accounts"]["analista"]["status"] == "synced"  # research runs also sync (stops checked hourly)
    data = json.loads((tmp_path / "live.json").read_text())
    r = data["research"]
    assert r["news"]["analyses"] and r["calendar"]["events"] and r["macro"]["indicators"]
    assert r["status"]["llm"] == "enabled" and r["llm_cost_usd_total"] > 0
    # research is off unless the config has a research section
    assert LiveConfig(symbols=["AAPL"], accounts=[]).research_config().enabled is False


def test_discretionary_agent_local_engine_without_keys(tmp_path, offline, monkeypatch):
    from tradesim.agents.discretionary import DiscretionaryAgent

    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    store = ResearchStore(tmp_path / "research")
    refresh_research(ResearchConfig(technicals=False), ["SPY", "AAPL", "GLD", "BTC-USD"], AssetUniverse.default(),
                     store, now=NOW, client=None)
    agent = DiscretionaryAgent(min_score=-1.0, max_weight=0.2, every=5)  # min_score -1: always pick something
    agent.attach_research(store)
    env = _sim()
    res = run_episode(agent, env, seed=0, episode_length=6)
    ex = agent.explain()
    assert ex["status"] == "ok" and ex["engine"].startswith("local") and res.metrics["num_fills"] > 0
    assert "motor local" in ex["market_view"] and ex["positions"]
    p = ex["positions"][0]
    assert p["thesis"].startswith("Puntuación") and p["invalidation"] and p["target_weight"] <= 0.2
    assert sum(x["target_weight"] for x in ex["positions"]) <= 0.95 + 1e-9
    assert agent.learning_summary()["last_engine"].startswith("local")
    env.close()


def test_discretionary_agent_free_engine_then_fallback(tmp_path, offline, monkeypatch):
    from tradesim.agents.discretionary import DiscretionaryAgent
    from tradesim.research.llm import OpenAICompatBackend

    monkeypatch.setenv("GEMINI_API_KEY", "free-key")
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    sent = []

    def fake_send(self, payload):
        sent.append(payload)
        return 200, {"choices": [{"message": {"content": json.dumps(DECISION)}}],
                     "usage": {"prompt_tokens": 3000, "completion_tokens": 800}}

    monkeypatch.setattr(OpenAICompatBackend, "_send", fake_send)
    store = ResearchStore(tmp_path / "research")
    refresh_research(ResearchConfig(technicals=False, engines=[]), ["SPY", "AAPL", "GLD", "BTC-USD"],
                     AssetUniverse.default(), store, now=NOW, client=None)
    agent = DiscretionaryAgent(max_weight=0.15, every=5, daily_requests={"gemini": 1})
    agent.attach_research(store)
    env = _sim()
    run_episode(agent, env, seed=0, episode_length=7)  # decisions on steps 1 and 6
    assert len(sent) == 1  # the daily budget allowed one request...
    assert agent.decisions[0]["summary"] == DECISION["market_view"]
    assert agent.explain()["engine"].startswith("local")  # ...the second decision fell back to the local engine
    assert "daily budget" in agent.explain()["engine_errors"][0]
    prompt = sent[0]["messages"][1]["content"]
    assert len(prompt) < 30_000 and "TECHNICALS" in prompt  # compact dossier for free tiers
    env.close()


def test_local_analysis_only_directs_named_assets():
    from tradesim.research.local_nlp import LocalNewsAnalyst

    la = LocalNewsAnalyst(["GLD (etf) SPDR Gold Shares", "XOM (equity/Energy) ExxonMobil", "TLT (etf) iShares 20+ Year Treasury Bond"])
    theme = NewsItem("a", "Gold price outlook as US bond yields rise ahead of FOMC", "u", NOW.isoformat(), "yahoo",
                     "Oil prices climb too.")
    theme.triage = la.triage_item(theme, NOW)
    assert {x["direction"] for x in la.analyse_item(theme, NOW)["affected"]} == {"unclear"}
    named = NewsItem("b", "Exxon beats estimates", "u", NOW.isoformat(), "yahoo")
    named.triage = la.triage_item(named, NOW)
    assert [(x["symbol"], x["direction"]) for x in la.analyse_item(named, NOW)["affected"]] == [("XOM", "up")]


def test_forecast_book_and_signal_learner_only_keep_supported_lessons():
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from tradesim.agents.forecasts import ForecastBook, SignalLearner

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    book = ForecastBook()
    rng = np.random.default_rng(0)
    for i in range(40):  # always says 80 % but only ~35 % go up: overconfident
        now = t0 + timedelta(days=i)
        book.record(now, [{"symbol": f"S{i}", "prob_up": 0.8, "horizon_days": 5, "conviction": "high"}],
                    {f"S{i}": 100.0}, "risk_on", {})
        book.resolve(now + timedelta(days=5), {f"S{i}": 100.0 * (1.01 if rng.random() < 0.35 else 0.99)})
    st = book.stats()
    assert st["overall"]["n"] == 40 and st["overall"]["brier"] > 0.25
    assert any(x.startswith("Sobreestimo") for x in book.lessons())
    few = ForecastBook()
    few.closed = book.closed[:10]
    assert few.lessons() == []  # n < 30: nothing is concluded

    sl = SignalLearner(horizon_days=1, min_samples=200)
    syms = [f"A{k}" for k in range(10)]
    prices = {s: 100.0 for s in syms}
    for d in range(120):  # the news score predicts tomorrow's relative return; technicals are noise
        now = t0 + timedelta(days=d)
        sl.resolve(now, prices)
        sig = {s: {"tech": rng.normal(), "news": rng.normal(), "trend": 0.0} for s in syms}
        sl.record(now, sig, prices, {s: 0.01 for s in syms})
        prices = {s: prices[s] * float(np.exp(0.01 * sig[s]["news"] + rng.normal(0, 0.005))) for s in syms}
    w, src = sl.weights()
    assert src == "aprendidos" and w["news"] > abs(w["tech"])


def test_signal_ic_by_horizon_drops_useless_news_and_pattern_edges():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    import numpy as np

    from tradesim.agents.forecasts import PatternStats, SignalLearner

    rng = np.random.default_rng(3)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    syms = [f"A{k}" for k in range(10)]
    sl = SignalLearner(min_days=10)
    ps = PatternStats(min_n=30)
    prices = {s: 100.0 for s in syms}
    for d in range(160):  # technicals predict, news is noise; a "hammer" works, a "doji" does not
        now = t0 + timedelta(days=d)
        sl.resolve(now, prices)
        ps.resolve(now, prices)
        sig = {s: {"tech": rng.normal(), "news": rng.normal(), "trend": 0.0} for s in syms}
        sl.record(now, sig, prices, {s: 0.01 for s in syms})
        reps = {s: SimpleNamespace(patterns=[SimpleNamespace(name="hammer" if sig[s]["tech"] > 0 else "doji",
                                                             direction="bullish", bars_ago=0)]) for s in syms}
        ps.record(now, reps, prices, {s: 0.01 for s in syms}, "neutral")
        prices = {s: prices[s] * float(np.exp(0.003 * sig[s]["tech"] + rng.normal(0, 0.004))) for s in syms}
    ic = sl.signal_ic()
    assert ic["tech"][5]["rank_ic_mean"] > 0.1 and ic["tech"][1]["days"] > 10
    w, src = sl.weights()
    assert w["news"] < 0.3 and src != "fijos"  # unproven news loses weight (learned or verified)
    assert ps.edge("hammer", "neutral") == 1.0 and ps.edge("doji", "neutral") in (0.0, -1.0)
