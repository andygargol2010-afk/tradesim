"""Daily chronicle of the live simulation: what the market did, which patterns showed up and what each agent did
given that situation. Written once per UTC day by an AI engine (free Gemini/Groq, or Claude) with a local fallback,
into ``dashboard/diario.json`` (read by ``diario.html``) and ``dashboard/diario.md`` (the same text as a document).

Each entry tags the day with a situation (``calma · alcista``, ``tensión · bajista``, ``solo cripto``...) and the
running statistics count, per agent and situation, how often it bought, sold or stayed put and how it did: the
behavioural patterns of every agent build up day after day.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from tradesim.persistence import read_json, write_json

log = logging.getLogger(__name__)

WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
KEY_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD"]
MACRO_KEYS = ["vix", "hy_spread", "ust_10y_live", "ust_10y", "dxy", "wti", "gold"]
KEEP_ENTRIES = 400
DIRECTION_ES = {"bullish": "alcista", "bearish": "bajista", "neutral": "neutral"}


def _r(x: Any) -> Any:
    return round(x, 2) if isinstance(x, float) else x

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "situacion": {"type": "string"},
        "patrones_mercado": {"type": "array", "items": {"type": "object", "properties": {
            "patron": {"type": "string"}, "activos": {"type": "array", "items": {"type": "string"}},
            "lectura": {"type": "string"}}, "required": ["patron", "activos", "lectura"]}},
        "agentes": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "que_hizo": {"type": "string"}, "por_que": {"type": "string"},
            "segun_situacion": {"type": "string"}, "resultado": {"type": "string"}},
            "required": ["id", "que_hizo", "por_que", "segun_situacion", "resultado"]}},
        "comparativa": {"type": "string"},
        "patrones_recurrentes": {"type": "array", "items": {"type": "string"}},
        "a_vigilar": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["titulo", "situacion", "patrones_mercado", "agentes", "comparativa", "patrones_recurrentes",
                 "a_vigilar"],
}

SYSTEM = """Eres el cronista de una simulación de trading con varios agentes autónomos, cada uno con 100 USD.
Cada día escribes en español la entrada del diario: qué situación de mercado hubo, qué patrones aparecieron y qué hizo
cada agente según esa situación. Usa solo los datos que recibes (no inventes precios, noticias ni operaciones).
- situacion: 2-3 frases sobre el día (índices, cripto, macro, eventos, noticias relevantes).
- patrones_mercado: los patrones técnicos más relevantes del día, agrupados, con su lectura.
- agentes: una entrada por agente (todos, con su id): que_hizo (operaciones concretas o que no operó), por_que (sus
  motivos según sus señales), segun_situacion (cómo encaja su comportamiento con la situación del día y con lo que
  suele hacer en situaciones parecidas según las estadísticas acumuladas), resultado (cambio de valor del día).
- comparativa: qué estilo funcionó mejor hoy y por qué.
- patrones_recurrentes: comportamientos que se repiten en los agentes a lo largo de los días (usa las estadísticas y
  los días anteriores; si aún hay pocos datos, dilo).
- a_vigilar: eventos o niveles de los próximos días.
Frases cortas y concretas, con cifras."""


# ------------------------------------------------------------------ facts
def _day_of(ts: str) -> date | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except (ValueError, AttributeError):
        return None


def _journal(state_dir: Path, account_id: str, day: date) -> list[dict[str, Any]]:
    path = state_dir / account_id / "journal.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _day_of(e.get("time", "")) == day:
            out.append(e)
    return out


def _equity_at(curve: dict[str, Any], until: datetime) -> float | None:
    vals = curve.get("equity") or curve.get("v") or []
    last = None
    for t, v in zip(curve.get("t", []), vals):
        if datetime.fromisoformat(t.replace("Z", "+00:00")) <= until:
            last = v
    return last


def situation_tag(market: dict[str, Any]) -> str:
    """Short label of the day: volatility regime · direction (or "solo cripto" when stock markets were shut)."""
    vix = (market.get("macro", {}).get("vix") or {}).get("value")
    regime = None if vix is None else "calma" if vix < 16 else "normal" if vix <= 24 else "tensión"
    ref = "SPY" if market.get("stocks_open") else "BTC-USD"
    chg = (market.get("changes_pct") or {}).get(ref)
    if chg is None:
        chg = ((market.get("technicals") or {}).get(ref) or {}).get("trend_change_pct")
        trend = None if chg is None else "alcista" if chg > 2 else "bajista" if chg < -2 else "lateral"
    else:
        trend = "alcista" if chg > 0.5 else "bajista" if chg < -0.5 else "lateral"
    parts = ([] if market.get("stocks_open") else ["solo cripto"]) + [x for x in (regime, trend) if x]
    return " · ".join(parts) or "sin referencia"


def collect_facts(config, day: date, data: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """Everything the chronicler needs about ``day`` (UTC), from the dashboard export and the account journals."""
    from tradesim.assets.universe import AssetUniverse
    from tradesim.live import is_market_open
    from tradesim.patterns import PATTERN_NAMES_ES

    end = datetime.combine(day, time.max, tzinfo=timezone.utc)
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    research = data.get("research") or {}
    prices = {s: p.get("last") for s, p in (data.get("prices") or {}).items() if p.get("last")}
    prev_closes = (previous or {}).get("closes") or {}
    changes = {s: round((p / prev_closes[s] - 1) * 100, 2) for s, p in prices.items() if prev_closes.get(s)}
    spy = AssetUniverse.default().get("SPY")
    stocks_open = is_market_open(spy, datetime.combine(day, time(15, 0), tzinfo=timezone.utc))

    techs: dict[str, Any] = {}
    for sym, t in ((research.get("technicals") or {}).get("symbols") or {}).items():
        pats = list(dict.fromkeys(
            PATTERN_NAMES_ES.get(p["name"], p["name"]) + f" ({DIRECTION_ES.get(p.get('direction'), p.get('direction'))})"
            for p in t.get("patterns", []) if p.get("bars_ago", 99) <= 2 and p.get("status") != "invalidated"))
        techs[sym] = {"trend": (t.get("trend") or {}).get("direction"),
                      "trend_change_pct": (t.get("trend") or {}).get("change_pct"),
                      "rsi": t.get("rsi"), "score": t.get("score"), "patterns": pats[:4]}
    macro = {k: {"name": v.get("name"), "value": _r(v.get("value")), "change": _r(v.get("change"))}
             for k, v in ((research.get("macro") or {}).get("indicators") or {}).items() if k in MACRO_KEYS}
    soon = {(day + timedelta(days=i)).isoformat() for i in range(0, 4)}
    events = [{"date": e.get("date"), "title": e.get("title"), "importance": e.get("importance")}
              for e in ((research.get("calendar") or {}).get("events") or [])
              if e.get("date") in soon and e.get("importance") == "high"][:8]
    news = [n for n in ((research.get("news") or {}).get("analyses") or []) if _day_of(n.get("published", "")) == day]
    news.sort(key=lambda n: -abs((n.get("triage") or {}).get("sentiment", 0) or 0))
    news = [{"title": n.get("title"), "symbols": (n.get("triage") or {}).get("symbols"),
             "sentiment": (n.get("triage") or {}).get("sentiment")} for n in news[:6]]
    movers = sorted(changes.items(), key=lambda kv: -abs(kv[1]))[:8]
    market = {"date": day.isoformat(), "weekday": WEEKDAYS_ES[day.weekday()], "stocks_open": stocks_open,
              "changes_pct": {s: changes[s] for s in KEY_SYMBOLS if s in changes}, "top_movers": dict(movers),
              "macro": macro, "events_next_days": events, "news": news, "technicals": techs}
    market["tag"] = situation_tag(market)

    state_dir = Path(config.state_dir)
    agents = []
    for acc in data.get("accounts", []):
        entries = _journal(state_dir, acc["id"], day)
        decisions = []
        for e in entries:
            orders = [{"side": o.get("side"), "symbol": o.get("symbol"), "status": o.get("status"),
                       "notional": round(float(o.get("filled_quantity") or o.get("quantity") or 0)
                                         * float(o.get("avg_fill_price") or 0), 2) or None,
                       "tag": (o.get("tag") or "")[:120]} for o in e.get("orders", [])]
            if e.get("mode") != "decide" and not orders:
                continue
            ex = e.get("explain") or {}
            why = ex.get("signals") or ex.get("reasons") or []
            decisions.append({"time": e.get("time", "")[11:16], "status": e.get("status"),
                              "message": (e.get("message") or "")[:200], "orders": orders[:20],
                              "signals": [str(s)[:160] for s in why[:6]],
                              "market_view": (ex.get("market_view") or "")[:500] or None})
        curve = acc.get("equity_curve") or {}
        eq0 = _equity_at(curve, start - timedelta(microseconds=1)) or acc.get("initial_cash")
        eq1 = _equity_at(curve, end) or (acc.get("summary") or {}).get("equity")
        filled = [o for d in decisions for o in d["orders"] if o["status"] in ("filled", "partially_filled")]
        agents.append({
            "id": acc["id"], "name": acc.get("name"), "agent": acc.get("agent"),
            "style": (acc.get("description") or "")[:220], "decisions": decisions,
            "buys": sorted({o["symbol"] for o in filled if o["side"] == "buy"}),
            "sells": sorted({o["symbol"] for o in filled if o["side"] == "sell"}),
            "closed_today": [{"symbol": t.get("symbol"), "pnl": t.get("pnl"), "exit": t.get("exit_kind")}
                             for t in acc.get("closed_trades", []) if _day_of(t.get("closed_at") or "") == day][:10],
            "equity_start": eq0, "equity_end": eq1,
            "day_pnl_pct": round((eq1 / eq0 - 1) * 100, 2) if eq0 and eq1 else None,
            "invested_pct": (acc.get("summary") or {}).get("invested_pct"),
            "top_positions": [p.get("symbol") for p in sorted(acc.get("positions", []),
                                                               key=lambda p: -(p.get("market_value") or 0))[:6]],
        })
    return {"market": market, "agents": agents, "closes": prices}


# ------------------------------------------------------------------ statistics
def running_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Per agent and situation: days, days with trades, buys, sells and average daily result."""
    acc: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    names: dict[str, str] = {}
    for e in entries:
        tag = e.get("situation_tag", "?")
        for a in e.get("facts", {}).get("agents", []):
            s = acc[a["id"]][tag]
            names[a["id"]] = a.get("name") or a["id"]
            s["days"] += 1
            s["days_traded"] += bool(a["buys"] or a["sells"])
            s["buys"] += len(a["buys"])
            s["sells"] += len(a["sells"])
            if a.get("day_pnl_pct") is not None:
                s["pnl_sum"] += a["day_pnl_pct"]
                s["pnl_n"] += 1
    out = {}
    for aid, tags in acc.items():
        out[aid] = {"name": names[aid], "by_situation": {
            tag: {"days": int(s["days"]), "days_traded": int(s["days_traded"]), "buys": int(s["buys"]),
                  "sells": int(s["sells"]),
                  "avg_day_pnl_pct": round(s["pnl_sum"] / s["pnl_n"], 2) if s["pnl_n"] else None}
            for tag, s in sorted(tags.items())}}
    return out


# ------------------------------------------------------------------ writing
def _fmt_pct(x: float | None) -> str:
    return "sin dato" if x is None else f"{x:+.2f} %".replace(".", ",")


def local_text(facts: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Deterministic entry, used when no AI engine is available (same structure as the AI one)."""
    m = facts["market"]
    ch = ", ".join(f"{s} {_fmt_pct(v)}" for s, v in m["changes_pct"].items()) or "sin variaciones registradas aún"
    macro = ", ".join(f"{v['name']} {v['value']}" for v in m["macro"].values() if v.get("value") is not None)
    by_pattern: dict[str, list[str]] = defaultdict(list)
    for sym, t in m["technicals"].items():
        for p in t["patterns"]:
            by_pattern[p].append(sym)
    pats = sorted(by_pattern.items(), key=lambda kv: -len(kv[1]))[:6]
    agents = []
    for a in facts["agents"]:
        did = []
        if a["buys"]:
            did.append("compró " + ", ".join(a["buys"]))
        if a["sells"]:
            did.append("vendió " + ", ".join(a["sells"]))
        msgs = [d["message"] for d in a["decisions"] if d["message"] and d["status"] != "skipped"]
        if not msgs and a["decisions"] and all(d["status"] == "skipped" for d in a["decisions"]):
            msgs = ["no tocaba decidir: bolsa cerrada o ya había decidido hoy"]
        why = [s for d in a["decisions"] for s in d["signals"]][:3] or [o["tag"] for d in a["decisions"]
                                                                        for o in d["orders"] if o["tag"]][:3]
        hist = stats.get(a["id"], {}).get("by_situation", {}).get(m["tag"])
        habit = (f"En días «{m['tag']}» ha operado {hist['days_traded']} de {hist['days']} veces "
                 f"(resultado medio {_fmt_pct(hist['avg_day_pnl_pct'])})." if hist else "Primer día con esta situación.")
        agents.append({"id": a["id"], "que_hizo": "; ".join(did) or ("no operó" + (f" ({msgs[0]})" if msgs else "")),
                       "por_que": "; ".join(why) or "sin señales para operar", "segun_situacion": habit,
                       "resultado": f"{_fmt_pct(a['day_pnl_pct'])} (valor {a['equity_end']:.2f} USD)"
                       if a.get("equity_end") else _fmt_pct(a["day_pnl_pct"])})
    ranked = sorted([a for a in facts["agents"] if a["day_pnl_pct"] is not None], key=lambda a: -a["day_pnl_pct"])
    return {
        "titulo": f"{m['weekday'].capitalize()}: {m['tag']}",
        "situacion": ". ".join(x for x in ("Sesión normal" if m["stocks_open"] else "Bolsa cerrada, solo cripto",
                                           ch, macro) if x) + ".",
        "patrones_mercado": [{"patron": p, "activos": syms[:8], "lectura": f"aparece en {len(syms)} activos"}
                             for p, syms in pats],
        "agentes": agents,
        "comparativa": (f"Mejor: {ranked[0]['name']} ({_fmt_pct(ranked[0]['day_pnl_pct'])}); peor: {ranked[-1]['name']} "
                        f"({_fmt_pct(ranked[-1]['day_pnl_pct'])}).") if ranked else "Sin resultados aún.",
        "patrones_recurrentes": [],
        "a_vigilar": [f"{e['date']}: {e['title']}" for e in m["events_next_days"][:5]],
    }


def _ai_text(config, facts: dict[str, Any], stats: dict[str, Any], recent: list[dict[str, Any]]):
    from tradesim.research.llm import Budget, LLMError, Usage, json_with_fallback, make_backends

    rc = config.research_config()
    if not rc.enabled or not rc.analyze:
        return None, None
    backends = make_backends(list(rc.engines), rc.free_models, Budget(Path(rc.dir) / "llm_budget.json",
                                                                      rc.daily_requests))
    if not backends:
        return None, None
    payload = {"hoy": {k: v for k, v in facts.items() if k != "closes"}, "estadisticas_acumuladas": stats,
               "dias_anteriores": [{"fecha": e["date"], "situacion": e["situation_tag"],
                                    "titulo": e["text"].get("titulo"), "comparativa": e["text"].get("comparativa")}
                                   for e in recent]}
    try:
        text, backend = json_with_fallback(backends, SYSTEM, json.dumps(payload, ensure_ascii=False, default=str),
                                           SCHEMA, Usage(), max_tokens=6000, effort="low")
        return text, backend.name
    except LLMError as exc:
        log.warning("chronicle: AI engines failed (%s); local text", exc)
        return None, None


def write_entry(config, day: date, use_ai: bool = True) -> dict[str, Any]:
    """Write (or rewrite) the chronicle entry of ``day`` and regenerate diario.json / diario.md."""
    out_dir = Path(config.dashboard_file).parent
    store = read_json(out_dir / "diario.json") if (out_dir / "diario.json").exists() else {"entries": []}
    entries = [e for e in store.get("entries", []) if e["date"] != day.isoformat()]
    previous = next((e for e in entries if e["date"] < day.isoformat()), None)
    data = read_json(config.dashboard_file)
    facts = collect_facts(config, day, data, previous.get("facts") if previous else None)
    tag = facts["market"]["tag"]
    stats = running_stats([e for e in entries if e["date"] < day.isoformat()]
                          + [{"situation_tag": tag, "facts": facts}])
    text, engine = _ai_text(config, facts, stats, [e for e in entries if e["date"] < day.isoformat()][:5]) \
        if use_ai else (None, None)
    local = local_text(facts, stats)
    if text:  # every agent must appear even if the model skipped one
        have = {a.get("id") for a in text.get("agentes", [])}
        text["agentes"] = text.get("agentes", []) + [a for a in local["agentes"] if a["id"] not in have]
    entry = {"date": day.isoformat(), "written_at": datetime.now(timezone.utc).isoformat(), "engine": engine or "local",
             "situation_tag": tag, "text": text or local, "facts": facts}
    entries = sorted(entries + [entry], key=lambda e: e["date"], reverse=True)[:KEEP_ENTRIES]
    store = {"updated_at": entry["written_at"], "stats": running_stats(entries), "entries": entries}
    write_json(out_dir / "diario.json", store)
    (out_dir / "diario.md").write_text(to_markdown(store, {a["id"]: a.get("name") for a in data.get("accounts", [])}))
    return entry


def maybe_write(config, now: datetime) -> dict[str, Any] | None:
    """Chronicle of yesterday (UTC) if it is not written yet; called after every live run."""
    out = Path(config.dashboard_file).parent / "diario.json"
    day = (now.astimezone(timezone.utc) - timedelta(days=1)).date()
    if out.exists() and any(e["date"] == day.isoformat() for e in read_json(out).get("entries", [])):
        return None
    if not Path(config.dashboard_file).exists():
        return None
    starts = [_day_of(a.get("started_at") or "") for a in read_json(config.dashboard_file).get("accounts", [])]
    starts = [d for d in starts if d]
    if not starts or day < min(starts):
        return None  # the simulation had not started yet
    return write_entry(config, day)


# ------------------------------------------------------------------ markdown
def to_markdown(store: dict[str, Any], names: dict[str, str]) -> str:
    lines = ["# Diario de la simulación", "",
             f"_Actualizado: {store['updated_at'][:16].replace('T', ' ')} UTC. Una entrada por día, la más reciente "
             "primero. Escrito por un agente de IA a partir de los datos de la simulación._", "",
             "## Patrones acumulados: qué hace cada agente según la situación", "",
             "| Agente | Situación | Días | Días con operaciones | Compras | Ventas | Resultado medio del día |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for aid, s in store.get("stats", {}).items():
        for tag, v in s["by_situation"].items():
            lines.append(f"| {s['name']} | {tag} | {v['days']} | {v['days_traded']} | {v['buys']} | {v['sells']} | "
                         f"{_fmt_pct(v['avg_day_pnl_pct'])} |")
    for e in store["entries"]:
        t = e["text"]
        lines += ["", f"## {e['date']} · {t.get('titulo', '')}", "",
                  f"**Situación ({e['situation_tag']}).** {t.get('situacion', '')}", ""]
        if t.get("patrones_mercado"):
            lines += ["**Patrones del mercado**", ""]
            lines += [f"- **{p['patron']}** ({', '.join(p['activos'])}): {p['lectura']}" for p in t["patrones_mercado"]]
            lines.append("")
        lines += ["**Qué hizo cada agente**", ""]
        for a in t.get("agentes", []):
            lines += [f"### {names.get(a['id'], a['id'])}", "",
                      f"- **Qué hizo:** {a['que_hizo']}", f"- **Por qué:** {a['por_que']}",
                      f"- **Según la situación:** {a['segun_situacion']}", f"- **Resultado:** {a['resultado']}", ""]
        lines += [f"**Comparativa.** {t.get('comparativa', '')}", ""]
        for title, key in (("Patrones recurrentes", "patrones_recurrentes"), ("A vigilar", "a_vigilar")):
            if t.get(key):
                lines += [f"**{title}**", ""] + [f"- {x}" for x in t[key]] + [""]
        lines.append(f"_Escrito con: {e['engine']}._")
    return "\n".join(lines) + "\n"
