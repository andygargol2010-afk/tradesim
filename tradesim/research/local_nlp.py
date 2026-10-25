"""Free, offline news analysis (no LLM, no API key): every item gets a triage and the important ones a
structured analysis, with the same fields the LLM analysts produce, so the agents and the dashboard work
the same way whichever engine ran.

* **Which assets**: tickers and company / coin names in the text (``Apple``, ``Nvidia``, ``bitcoin``...),
  the feed the item came from, and macro themes mapped to the asset types they move (oil -> energy,
  the Fed -> everything, a crypto regulation story -> crypto...).
* **Sentiment**: a finance word list in the spirit of Loughran-McDonald plus phrases that matter in
  markets ("beats estimates", "cuts guidance", "downgrade", "rate hike"...), with simple negation.
* **Relevance / impact**: event type, magnitude words, source quality, and a penalty for clickbait and
  listicles ("3 stocks to buy", "should you buy...?", "prediction:").
* **Analysis** (important items): the facts are the headline and summary; the channel, horizon and
  what to watch come from the event type; related calendar events are matched by asset and date.

It is a keyword engine: it reads headlines well and misses irony and nuance. The LLM analysts replace
it when a (free or paid) model is configured; it remains the fallback when a quota runs out.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Iterable

from tradesim.research.news import NewsItem

ENGINE = "local"

POSITIVE = set("""
beat beats beating exceeded exceeds surpass surpassed record records strong stronger strongest robust solid gain gains
gained rally rallies rallied surge surges surged soar soars soared jump jumps jumped climb climbs climbed rise rises rose
upgrade upgrades upgraded outperform outperforms bullish optimistic optimism boost boosts boosted expand expands expansion
growth growing grew profitable profit profits rebound rebounds rebounded recover recovers recovery approval approved
approves win wins won breakthrough raise raises raised higher accelerate accelerates accelerating buyback buybacks
dividend inflows inflow upbeat resilient easing eased cools cooled cooling dovish cut-rate stimulus deal partnership
""".split())
NEGATIVE = set("""
miss misses missed weak weaker weakest decline declines declined drop drops dropped fall falls fell plunge plunges plunged
slump slumps slumped tumble tumbles tumbled sink sinks sank crash crashes crashed downgrade downgrades downgraded
underperform bearish pessimistic warning warns warned cut cuts lower lowers lowered loss losses lawsuit lawsuits sue sued
probe investigation investigated fine fined penalty recall recalls recalled bankruptcy bankrupt default defaults layoffs
layoff fraud halt halted halts delay delays delayed shortage shortfall concern concerns fear fears risk risks volatile
selloff sell-off outflows outflow hawkish tariff tariffs sanction sanctions strike strikes hack hacked exploit
inflation recession slowdown contraction downturn disappointing disappoints disappointed slashes slashed
gloom gloomy gloomier worst turmoil dent dents rattle rattles rattled slide slides sliding slid tumbling plunging
stoke stokes fled flee crisis panic stress distress
""".split())
PHRASES_POS = ["beats estimates", "beat estimates", "tops estimates", "raises guidance", "raised guidance", "raises outlook",
               "record high", "all-time high", "price target raised", "upgraded to buy", "rate cut", "cuts rates",
               "better than expected", "stronger than expected", "share buyback", "etf inflows", "approves etf"]
PHRASES_NEG = ["misses estimates", "missed estimates", "cuts guidance", "lowers guidance", "cut guidance", "price target cut",
               "downgraded to sell", "rate hike", "raises rates", "worse than expected", "weaker than expected",
               "files for bankruptcy", "under investigation", "job cuts", "profit warning", "etf outflows", "hotter than expected"]
NEGATORS = {"not", "no", "never", "without", "fails", "failed", "despite"}

CLICKBAIT = re.compile(
    r"(\b\d+\s+(?:[\w-]+\s+){0,3}(stocks?|etfs?|reasons?|things?|picks?|shares)\b|\bshould you (buy|sell)\b|"
    r"\bis it time to\b|\bbetter buy\b|\bbuy now\b|\bmillionaire|\bforever\b|\bprediction:|^opinion:|\bcould make you\b|"
    r"\bretire\w*|\bdividend stocks? to\b|\bmy top\b|\bi'm buying\b|\bhere's why\b|\bno-brainer\b|\bhidden gem|"
    r"\bmotley fool\b|\bwhat (it|this) means for (you|investors)\b|\bstocks? and other\b|\bsmart (move|buy)\b)", re.I)
QUALITY_SOURCES = {"reuters", "bloomberg", "wall street journal", "wsj", "financial times", "ft", "cnbc", "associated press",
                   "ap", "federalreserve", "federal reserve", "barron's", "marketwatch", "the economist", "coindesk"}

CATEGORY_RULES: list[tuple[str, str]] = [
    ("central_bank", r"\b(fed|fomc|powell|federal reserve|ecb|lagarde|boj|bank of japan|boe|rate (cut|hike|decision)|interest rates?)\b"),
    ("earnings", r"\b(earnings|quarterly results|eps|revenue|q[1-4] results|profit|beats|misses|guidance)\b"),
    ("macro", r"\b(cpi|inflation|pce|payrolls|jobs report|unemployment|gdp|retail sales|ism|pmi|jobless claims|consumer sentiment|recession)\b"),
    ("regulation", r"\b(sec|regulat\w*|antitrust|doj|ftc|lawsuit|probe|investigation|ban|approval|approves|fda)\b"),
    ("geopolitics", r"\b(war|sanction\w*|tariff\w*|election|geopolitic\w*|china|russia|iran|israel|ukraine|opec)\b"),
    ("analyst_rating", r"\b(upgrade\w*|downgrade\w*|price target|initiat\w* coverage|overweight|underweight|outperform rating)\b"),
    ("crypto", r"\b(bitcoin|ethereum|crypto\w*|stablecoins?|blockchain|tokens?|solana|xrp|dogecoin|etf flows?)\b"),
    ("commodities", r"\b(oil|crude|brent|wti|gold|silver|copper|natural gas|opec)\b"),
    ("corporate", r"\b(acquisition|acquire\w*|merger|buyback|layoffs?|ceo|split|spin-?off|ipo|dividend)\b"),
    ("product", r"\b(launch\w*|unveil\w*|product|iphone|chip|model|release\w*)\b"),
    ("markets", r"\b(stocks?|s&p 500|nasdaq|dow|markets?|wall street|treasur\w*|yields?|dollar)\b"),
]
CATEGORY_WEIGHT = {"central_bank": 0.85, "macro": 0.8, "earnings": 0.75, "regulation": 0.6, "geopolitics": 0.6,
                   "analyst_rating": 0.45, "crypto": 0.55, "commodities": 0.55, "corporate": 0.5, "product": 0.4,
                   "markets": 0.35, "opinion": 0.15, "other": 0.2}

# Macro themes -> asset groups they move ("*all" = every asset, else symbols or profile names).
THEME_ASSETS = {
    "central_bank": ["*all"], "macro": ["*all"],
    "oil": ["XLE", "XOM", "CVX"], "gold": ["GLD", "SLV"], "silver": ["SLV"], "copper": ["EEM", "CAT"],
    "crypto": ["*crypto"], "dollar": ["GLD", "EEM"], "yields": ["TLT", "XLF", "QQQ"],
}

# Spanish explanations by event type (the "why it matters" of the local analysis).
CHANNEL_ES = {
    "earnings": "Los resultados y las previsiones mueven directamente la valoración: el mercado compara cifras y guía con lo esperado.",
    "central_bank": "Los tipos de interés cambian el coste del dinero y el valor presente de los beneficios futuros; afectan a todos los activos, sobre todo a bonos, tecnológicas y cripto.",
    "macro": "Los datos macro cambian las expectativas sobre crecimiento, inflación y tipos de la Fed, y con ellas el apetito por el riesgo.",
    "regulation": "Las decisiones regulatorias o judiciales pueden alterar el negocio, multar o abrir/cerrar mercados.",
    "geopolitics": "Los riesgos geopolíticos y comerciales afectan a cadenas de suministro, materias primas y apetito por el riesgo.",
    "analyst_rating": "Los cambios de recomendación mueven flujos a corto plazo; su efecto suele diluirse en días.",
    "crypto": "Las noticias de flujos, regulación o adopción mueven el sentimiento de todo el mercado cripto.",
    "commodities": "Los precios de las materias primas afectan a márgenes de productores y a la inflación.",
    "corporate": "Las operaciones corporativas (fusiones, recompras, despidos) cambian el valor por acción o la percepción del negocio.",
    "product": "Los lanzamientos pueden cambiar expectativas de ventas, normalmente con efecto gradual.",
    "markets": "Refleja el tono general del mercado más que un hecho concreto.",
    "opinion": "Es un artículo de opinión o recomendación genérica: poco valor informativo.",
    "other": "Noticia de impacto incierto.",
}
HORIZON_BY_CATEGORY = {"earnings": "days", "central_bank": "weeks", "macro": "days", "regulation": "weeks",
                       "geopolitics": "weeks", "analyst_rating": "days", "crypto": "days", "commodities": "weeks",
                       "corporate": "weeks", "product": "weeks", "markets": "days", "opinion": "days", "other": "days"}
CATEGORY_ES = {"earnings": "resultados", "central_bank": "banco central", "macro": "dato macro", "regulation": "regulación",
               "geopolitics": "geopolítica", "analyst_rating": "recomendación de analistas", "crypto": "cripto",
               "commodities": "materias primas", "corporate": "operación corporativa", "product": "producto",
               "markets": "mercado", "opinion": "opinión", "other": "otros"}

_WORD = re.compile(r"[a-z][a-z\-']+")

# Names the press uses that differ from the listed company / coin name.
EXTRA_ALIASES = {
    "XOM": ["exxon"], "GOOGL": ["google", "alphabet"], "META": ["facebook", "meta platforms", "instagram"],
    "LLY": ["eli lilly", "lilly"], "JPM": ["jpmorgan", "jp morgan"], "BAC": ["bank of america", "bofa"],
    "GS": ["goldman"], "V": ["visa"], "MA": ["mastercard"], "JNJ": ["johnson & johnson", "j&j"],
    "UNH": ["unitedhealth"], "PG": ["procter", "p&g"], "KO": ["coca-cola", "coca cola", "coke"],
    "PEP": ["pepsico", "pepsi"], "WMT": ["walmart"], "COST": ["costco"], "HD": ["home depot"],
    "MCD": ["mcdonald's", "mcdonalds"], "CAT": ["caterpillar"], "BA": ["boeing"], "TSLA": ["tesla"],
    "NVDA": ["nvidia"], "AVGO": ["broadcom"], "ORCL": ["oracle"], "CRM": ["salesforce"], "NFLX": ["netflix"],
    "AMZN": ["amazon", "aws"], "CVX": ["chevron"], "PFE": ["pfizer"], "MSFT": ["microsoft"], "AAPL": ["apple", "iphone"],
    "AMD": ["advanced micro devices"], "BTC-USD": ["bitcoin", "btc"], "ETH-USD": ["ethereum", "ether"],
    "SOL-USD": ["solana"], "XRP-USD": ["xrp", "ripple"], "ADA-USD": ["cardano"], "DOGE-USD": ["dogecoin", "doge"],
    "LINK-USD": ["chainlink"], "LTC-USD": ["litecoin"], "GLD": ["gold etf"], "TLT": ["long-dated treasuries"],
}


def _aliases(universe_lines: Iterable[str]) -> dict[str, list[str]]:
    """symbol -> lower-case names to look for (ticker as a word, company / coin name without suffixes)."""
    out: dict[str, list[str]] = {}
    for line in universe_lines:
        sym, rest = line.split(" ", 1)
        name = rest.split(") ", 1)[-1] if ") " in rest else rest
        base = sym.split("-")[0].lower()
        clean = re.sub(r"\b(inc|corp|corporation|company|co|plc|ltd|the|trust|class [a-z]|platforms|holdings|group|"
                       r"incorporated|\(the\)|\(class a\))\b\.?", "", name.lower())
        clean = re.sub(r"[^a-z0-9& ]+", " ", clean).strip()
        names = {base} if len(base) >= 3 or sym.endswith("-USD") else set()
        first = clean.split(" ")[0] if clean else ""
        if clean and len(clean) >= 4:
            names.add(clean)
        if first and len(first) >= 4 and first not in {"spdr", "ishares", "invesco", "select", "energy", "financial"}:
            names.add(first)
        names |= set(EXTRA_ALIASES.get(sym, []))
        out[sym] = sorted(names)
    return out


class LocalNewsAnalyst:
    """Keyword-based triage and analysis (see module docstring). Same outputs as the LLM analysts."""

    def __init__(self, universe_lines: list[str]):
        self.universe_lines = universe_lines
        self.symbols = [line.split(" ", 1)[0] for line in universe_lines]
        self.aliases = _aliases(universe_lines)
        self.crypto = [s for s in self.symbols if s.endswith("-USD")]

    # ------------------------------------------------------------------ pieces
    def sentiment(self, text: str) -> tuple[float, int]:
        """(-1..1 score, number of sentiment hits)."""
        t = text.lower()
        score, hits = 0.0, 0
        for p in PHRASES_POS:
            if p in t:
                score += 2
                hits += 1
        for p in PHRASES_NEG:
            if p in t:
                score -= 2
                hits += 1
        words = _WORD.findall(t)
        for i, w in enumerate(words):
            sign = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
            if not sign:
                continue
            if any(x in NEGATORS for x in words[max(0, i - 3):i]):
                sign = -sign
            score += sign
            hits += 1
        return (math.tanh(score / 3.0) if hits else 0.0), hits

    def category(self, text: str) -> str:
        t = text.lower()
        if CLICKBAIT.search(t):
            return "opinion"
        for cat, pat in CATEGORY_RULES:
            if re.search(pat, t):
                return cat
        return "other"

    def assets(self, item: NewsItem, text: str, category: str) -> list[str]:
        t = " " + text.lower() + " "
        found = []
        for sym, names in self.aliases.items():
            if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", t) for n in names):
                found.append(sym)
            elif re.search(rf"(?<![A-Za-z]){re.escape(sym.split('-')[0])}(?![A-Za-z])", text) and len(sym.split("-")[0]) >= 2 \
                    and sym.split("-")[0].isupper() and len(sym) <= 5:
                found.append(sym)  # ticker written in capitals, e.g. "NVDA"
        if not found:
            found = [s for s in item.symbols if s in self.symbols]
        themes = set(item.topics)
        if category in ("central_bank", "macro", "crypto"):
            themes.add(category)
        for kw, theme in (("oil", "oil"), ("crude", "oil"), ("opec", "oil"), ("gold", "gold"), ("silver", "silver"),
                          ("copper", "copper"), ("bitcoin", "crypto"), ("crypto", "crypto"), ("dollar", "dollar"),
                          ("treasury", "yields"), ("yield", "yields")):
            if kw in t:
                themes.add(theme)
        broad: list[str] = []
        for th in themes:
            for a in THEME_ASSETS.get(th, []):
                if a == "*crypto":
                    broad += self.crypto
                elif a != "*all" and a in self.symbols:
                    broad.append(a)
        self._named = found  # assets the text names (vs. assets only touched by the theme)
        broad = [b for b in dict.fromkeys(broad) if b not in found]
        return (found + broad[:4])[:8]

    # ------------------------------------------------------------------ triage
    def triage_item(self, item: NewsItem, now: datetime) -> dict[str, Any]:
        text = f"{item.title}. {item.summary}"
        cat = self.category(item.title + " " + item.summary[:200])
        sent, hits = self.sentiment(text)
        syms = self.assets(item, text, cat)
        named = list(self._named)
        quality = (item.publisher or item.source or "").lower()
        good_source = any(q == quality or q in quality.split() for q in QUALITY_SOURCES) or item.source == "federalreserve"
        specific = [s for s in syms if s not in item.symbols] or syms  # beyond the feed it came from
        rel = 0.7 * CATEGORY_WEIGHT.get(cat, 0.2)
        rel += 0.12 if good_source else 0.0
        rel += 0.04 * min(hits, 3) + (0.08 if specific and cat not in ("opinion", "other") else 0.0)
        rel -= 0.25 if not syms and cat not in ("central_bank", "macro", "geopolitics") else 0.0
        rel = max(0.0, min(1.0, rel))
        strength = abs(sent) * (1.2 if good_source else 1.0)
        impact = "high" if rel >= 0.75 and strength >= 0.5 else "medium" if rel >= 0.5 and (strength >= 0.3 or cat in (
            "central_bank", "macro", "earnings")) else "low"
        return {"relevance": round(rel, 3), "sentiment": round(sent, 3), "impact": impact, "category": cat,
                "symbols": syms, "named": named,
                "summary": item.title if not item.summary else f"{item.title}. {item.summary[:220]}",
                "model": ENGINE, "at": now.isoformat()}

    def triage(self, items: list[NewsItem], now: datetime) -> int:
        for it in items:
            it.triage = self.triage_item(it, now)
        return len(items)

    # ------------------------------------------------------------------ analysis
    def analyse_item(self, item: NewsItem, now: datetime, calendar: dict[str, Any] | None = None) -> dict[str, Any]:
        t = item.triage or self.triage_item(item, now)
        cat, sent = t["category"], t["sentiment"]
        direction = "up" if sent > 0.15 else "down" if sent < -0.15 else "unclear"
        magnitude = "high" if t["impact"] == "high" else "medium" if t["impact"] == "medium" else "low"
        horizon = HORIZON_BY_CATEGORY.get(cat, "days")
        named = set(t["named"] if "named" in t else t["symbols"])
        tone = 'positivo' if direction == 'up' else 'negativo' if direction == 'down' else 'neutro'
        affected = []
        for s in t["symbols"][:6]:
            if s in named:  # the headline is about this asset: its tone applies
                affected.append({"symbol": s, "direction": direction, "magnitude": magnitude, "horizon": horizon,
                                 "rationale": f"{CATEGORY_ES.get(cat, cat).capitalize()}: tono {tone} en el titular "
                                              f"({sent:+.2f})."})
            else:  # only touched through the theme (oil, rates, crypto...): direction not inferable from keywords
                affected.append({"symbol": s, "direction": "unclear", "magnitude": "low", "horizon": horizon,
                                 "rationale": f"Afectado por el tema ({CATEGORY_ES.get(cat, cat)}); el sentido no se "
                                              "deduce del titular."})
        related = []
        if calendar:
            day = now.date()
            for e in calendar.get("events", []):
                try:
                    d = datetime.fromisoformat(e["date"]).date()
                except (KeyError, ValueError):
                    continue
                if not (day - timedelta(days=3) <= d <= day + timedelta(days=10)):
                    continue
                syms = set(e.get("symbols") or [])
                if (syms & set(t["symbols"])) or (cat in ("central_bank", "macro") and e.get("category") == cat.replace(
                        "macro", "inflation") or (cat == "central_bank" and e.get("category") == "central_bank")):
                    related.append(e["id"])
        watch = {"earnings": "La reacción del precio en la sesión siguiente y si la previsión cambia.",
                 "central_bank": "El próximo comunicado y el discurso de los miembros del banco central.",
                 "macro": "La revisión del dato y cómo cambian las expectativas de tipos.",
                 "crypto": "Los flujos de los ETF y la reacción de BTC/ETH."}.get(cat, "Si otras fuentes confirman la noticia.")
        return {
            "id": item.id, "headline": item.title,
            "what_happened": (item.summary or item.title)[:600],
            "why_it_matters": CHANNEL_ES.get(cat, CHANNEL_ES["other"]),
            "affected": affected,
            "macro_link": "", "priced_in": "unknown",
            "confidence": round(min(0.6, 0.2 + abs(sent) * 0.5), 2),
            "what_to_watch": watch, "related_events": related[:5], "read_full_article": False,
            "model": ENGINE, "at": now.isoformat(),
        }
