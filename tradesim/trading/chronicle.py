"""Daily chronicle of the live simulation: what the market did, which patterns showed up and what each agent did
given that situation. The market part is compiled from the research data; the part of each agent is written by the
agent itself (:meth:`tradesim.agents.base.Agent.diary`, stored with every decision in its journal; the analyst writes
it with its own AI engine). Compiled once per UTC day into ``dashboard/diario.json`` (read by ``diario.html``) and
``dashboard/diario.md`` (the same text as a document).

Each entry tags the day with a situation (``calma · alcista``, ``tensión · bajista``, ``solo cripto``...) and the
running statistics count, per agent and situation, how often it bought, sold or stayed put and how it did: the
behavioural patterns of every agent build up day after day.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from tradesim.persistence import read_json, write_json

WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
KEY_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD"]
MACRO_KEYS = ["vix", "hy_spread", "ust_10y_live", "ust_10y", "dxy", "wti", "gold"]
KEEP_ENTRIES = 400
DIRECTION_ES = {"bullish": "alcista", "bearish": "bajista", "neutral": "neutral"}


def _r(x: Any) -> Any:
    return round(x, 2) if isinstance(x, float) else x


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


def situation(market: dict[str, Any]) -> dict[str, Any]:
    """Volatility regime (VIX), direction (SPY, or BTC with the stock market shut) and a short label of both."""
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
    return {"tag": " · ".join(parts) or "sin referencia", "regime": regime, "trend": trend, "vix": vix,
            "stocks_open": bool(market.get("stocks_open"))}


def situation_tag(market: dict[str, Any]) -> str:
    return situation(market)["tag"]


def situation_now(store, stocks_open: bool) -> dict[str, Any]:
    """The situation an agent sees when it decides (latest research data), passed to its ``diary``."""
    macro = store.load("macro").get("indicators", {}) if store is not None else {}
    techs = store.load("technicals").get("symbols", {}) if store is not None else {}
    return situation({"stocks_open": stocks_open, "macro": {"vix": macro.get("vix") or {}},
                      "technicals": {s: {"trend_change_pct": (t.get("trend") or {}).get("change_pct")}
                                     for s, t in techs.items() if s in ("SPY", "BTC-USD")}})


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
                              "market_view": (ex.get("market_view") or "")[:500] or None,
                              "diary": e.get("diary")})
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
    """The day's entry: market part compiled from the data, each agent's part from its own diary notes."""
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
        hist = stats.get(a["id"], {}).get("by_situation", {}).get(m["tag"])
        habit = (f"Historial en días «{m['tag']}»: operó {hist['days_traded']} de {hist['days']} "
                 f"(resultado medio {_fmt_pct(hist['avg_day_pnl_pct'])})." if hist else "")
        notes = [d["diary"] for d in a["decisions"] if d.get("diary")]
        extra = sorted({f"{'compra' if o['side'] == 'buy' else 'venta'} de {o['symbol']} ({o['tag'] or 'orden'})"
                        for d in a["decisions"] if not d.get("diary") for o in d["orders"]
                        if o["status"] in ("filled", "partially_filled")})
        if notes:  # written by the agent itself when it decided
            que = " ".join(n["que_hice"] for n in notes)
            por = " ".join(n["por_que"] for n in notes if n.get("por_que"))
            segun = notes[-1].get("segun_situacion", "")
            author = "agente"
        else:
            skipped = [d["message"] for d in a["decisions"] if d["status"] == "skipped"]
            que = "No tomó ninguna decisión hoy."
            por = ("No le tocaba decidir: bolsa cerrada o ya había decidido hoy." if skipped
                   else "No hubo ejecución de decisión este día.")
            segun = ""
            author = "registro"
        if extra:
            que += " Además se ejecutaron órdenes que tenía puestas: " + "; ".join(extra) + "."
        agents.append({"id": a["id"], "autor": author, "que_hizo": que, "por_que": por,
                       "segun_situacion": " ".join(x for x in (segun, habit) if x) or "—",
                       "resultado": f"{_fmt_pct(a['day_pnl_pct'])} (valor {a['equity_end']:.2f} USD)"
                       if a.get("equity_end") else _fmt_pct(a["day_pnl_pct"])})
    recurring = []
    for aid, st in stats.items():
        rows = st["by_situation"]
        days = sum(v["days"] for v in rows.values())
        if days < 3:
            continue
        traded = sum(v["days_traded"] for v in rows.values())
        best = max((kv for kv in rows.items() if kv[1]["avg_day_pnl_pct"] is not None),
                   key=lambda kv: kv[1]["avg_day_pnl_pct"], default=None)
        busiest = max(rows.items(), key=lambda kv: kv[1]["days_traded"] / kv[1]["days"])
        recurring.append(f"{st['name']}: opera en {traded} de {days} días; donde más opera es en «{busiest[0]}»"
                         + (f"; le va mejor en «{best[0]}» ({_fmt_pct(best[1]['avg_day_pnl_pct'])} de media)."
                            if best else "."))
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
        "patrones_recurrentes": recurring or ["Aún hay pocos días para ver patrones de comportamiento."],
        "a_vigilar": [f"{e['date']}: {e['title']}" for e in m["events_next_days"][:5]],
    }


def write_entry(config, day: date) -> dict[str, Any]:
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
    entry = {"date": day.isoformat(), "written_at": datetime.now(timezone.utc).isoformat(), "engine": "agentes",
             "situation_tag": tag, "text": local_text(facts, stats), "facts": facts}
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
             "primero. Cada agente escribe su propia parte al decidir; la situación del mercado se compila de los "
             "datos._", "",
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
            lines += [f"### {names.get(a['id'], a['id'])}" + (" (escrito por el agente)" if a.get("autor") == "agente"
                                                              else ""), "",
                      f"- **Qué hizo:** {a['que_hizo']}", f"- **Por qué:** {a['por_que']}",
                      f"- **Según la situación:** {a['segun_situacion']}", f"- **Resultado:** {a['resultado']}", ""]
        lines += [f"**Comparativa.** {t.get('comparativa', '')}", ""]
        for title, key in (("Patrones recurrentes", "patrones_recurrentes"), ("A vigilar", "a_vigilar")):
            if t.get(key):
                lines += [f"**{title}**", ""] + [f"- {x}" for x in t[key]] + [""]
    return "\n".join(lines) + "\n"
