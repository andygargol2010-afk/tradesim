"""News collection: free RSS feeds per asset and for the macro picture, deduplicated into one store.

Sources (no API keys):

* Yahoo Finance headlines per symbol (stocks, ETFs and crypto);
* CoinDesk (crypto market news);
* Federal Reserve monetary-policy press releases;
* Google News searches for macro themes (Fed, inflation, jobs, yields, oil, trade, crypto regulation...).

The same story often appears in several feeds; items are merged by normalised title and keep the
union of the symbols they were found for. Analysis (``item.analysis``) is filled later by
:class:`tradesim.research.analyst.NewsAnalyst`.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus

from tradesim.research.http import FetchError, fetch_text

log = logging.getLogger(__name__)

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
FED_RSS = "https://www.federalreserve.gov/feeds/press_monetary.xml"
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"

# Macro themes searched on Google News (last 2 days). Each maps to the asset groups it usually moves.
MACRO_QUERIES: dict[str, str] = {
    "fed": '"Federal Reserve" OR FOMC OR Powell rates',
    "inflation": "CPI OR PCE OR inflation report United States",
    "jobs": '"jobs report" OR payrolls OR unemployment rate OR "jobless claims"',
    "growth": "US GDP OR recession OR ISM OR \"retail sales\"",
    "yields": "Treasury yields OR bond market OR \"10-year yield\"",
    "oil": "oil prices OR OPEC OR crude",
    "trade": "tariffs OR \"trade war\" OR sanctions markets",
    "crypto_policy": "SEC crypto OR bitcoin ETF OR stablecoin regulation",
    "geopolitics": "geopolitical risk markets OR election markets",
    "dollar": "\"US dollar\" OR \"dollar index\" currency markets",
}


@dataclass
class NewsItem:
    """One article. ``id`` is stable across runs (hash of the normalised title)."""

    id: str
    title: str
    url: str
    published: str  # ISO UTC
    source: str
    summary: str = ""
    publisher: str = ""
    symbols: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    fetched_at: str = ""
    triage: dict[str, Any] | None = None  # quick analysis of every item (relevance, sentiment, assets)
    analysis: dict[str, Any] | None = None  # detailed analysis of the important ones

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NewsItem":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def published_dt(self) -> datetime:
        return datetime.fromisoformat(self.published)


def normalise_title(title: str) -> str:
    t = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)  # Google News appends " - Publisher"
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()[:120]


def item_id(title: str) -> str:
    return hashlib.sha1(normalise_title(title).encode()).hexdigest()[:16]


def _clean(text: str | None, limit: int = 600) -> str:
    if not text:
        return ""
    t = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", t).strip()[:limit]


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def parse_rss(xml_text: str, source: str, symbols: Iterable[str] = (), topics: Iterable[str] = (),
              now: datetime | None = None) -> list[NewsItem]:
    """RSS 2.0 -> items (Google News titles keep their publisher in ``publisher``)."""
    now = now or datetime.now(timezone.utc)
    try:
        root = ET.fromstring(xml_text.lstrip("\ufeff \r\n\t"))
    except ET.ParseError as exc:
        raise FetchError(f"{source}: invalid RSS ({exc})") from exc
    out = []
    for it in root.iter("item"):
        title = _clean(it.findtext("title"), 300)
        if not title:
            continue
        published = _parse_date(it.findtext("pubDate")) or now
        publisher = _clean(it.findtext("source"), 80)
        if not publisher and " - " in title and source == "google-news":
            publisher = title.rsplit(" - ", 1)[1]
        summary = _clean(it.findtext("description"))
        if source == "google-news":
            summary = ""  # Google's description only repeats the title as a link
            title = re.sub(r"\s+-\s+[^-]{2,60}$", "", title)
        out.append(NewsItem(id=item_id(title), title=title, url=(it.findtext("link") or "").strip(),
                            published=published.isoformat(), source=source, summary=summary,
                            publisher=publisher, symbols=list(symbols), topics=list(topics),
                            fetched_at=now.isoformat()))
    return out


Fetcher = Callable[[str], str]


class NewsCollector:
    """Fetches all configured feeds. ``fetch`` is injectable for tests (``url -> text``)."""

    def __init__(self, symbols: list[str], crypto_symbols: Iterable[str] = (), macro_queries: dict[str, str] | None = None,
                 fetch: Fetcher | None = None, max_age_days: float = 3.0):
        self.symbols = list(symbols)
        self.crypto = set(crypto_symbols)
        self.macro_queries = MACRO_QUERIES if macro_queries is None else macro_queries
        self.fetch = fetch or (lambda url: fetch_text(url, timeout=20, retries=1))
        self.max_age = timedelta(days=max_age_days)
        self.errors: dict[str, str] = {}

    def _get(self, key: str, url: str, **kw: Any) -> list[NewsItem]:
        try:
            return parse_rss(self.fetch(url), **kw)
        except Exception as exc:  # noqa: BLE001 - one dead feed must not stop the others
            self.errors[key] = str(exc)[:200]
            log.warning("news feed %s failed: %s", key, exc)
            return []

    def collect(self, now: datetime | None = None) -> list[NewsItem]:
        now = now or datetime.now(timezone.utc)
        items: list[NewsItem] = []
        for sym in self.symbols:
            items += self._get(f"yahoo:{sym}", YAHOO_RSS.format(symbol=sym), source="yahoo", symbols=[sym], now=now)
        if self.crypto:
            items += self._get("coindesk", COINDESK_RSS, source="coindesk", topics=["crypto"], now=now)
        items += self._get("fed", FED_RSS, source="federalreserve", topics=["fed"], now=now)
        for topic, q in self.macro_queries.items():
            url = GOOGLE_NEWS_RSS.format(query=quote_plus(f"{q} when:2d"))
            items += self._get(f"google:{topic}", url, source="google-news", topics=[topic], now=now)
        cutoff = now - self.max_age
        return [i for i in items if i.published_dt() >= cutoff]


class NewsStore:
    """The deduplicated news of the last ``keep_days`` days, persisted as JSON."""

    def __init__(self, items: Iterable[NewsItem] = ()):
        self.items: dict[str, NewsItem] = {}
        for it in items:
            self.add(it)

    def add(self, item: NewsItem) -> bool:
        """Insert or merge (symbols / topics union). Returns True if the item is new."""
        cur = self.items.get(item.id)
        if cur is None:
            self.items[item.id] = item
            return True
        cur.symbols = sorted(set(cur.symbols) | set(item.symbols))
        cur.topics = sorted(set(cur.topics) | set(item.topics))
        if not cur.summary and item.summary:
            cur.summary = item.summary
        if item.published < cur.published:
            cur.published = item.published
        return False

    def merge(self, items: Iterable[NewsItem]) -> int:
        return sum(self.add(i) for i in items)

    def prune(self, now: datetime, keep_days: float) -> None:
        cutoff = (now - timedelta(days=keep_days)).isoformat()
        self.items = {k: v for k, v in self.items.items() if v.published >= cutoff}

    def recent(self, hours: float, now: datetime) -> list[NewsItem]:
        cutoff = (now - timedelta(hours=hours)).isoformat()
        return sorted((i for i in self.items.values() if i.published >= cutoff), key=lambda i: i.published,
                      reverse=True)

    def for_symbol(self, symbol: str) -> list[NewsItem]:
        return sorted((i for i in self.items.values() if symbol in i.symbols or
                       symbol in ((i.triage or {}).get("symbols") or [])),
                      key=lambda i: i.published, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [i.to_dict() for i in sorted(self.items.values(), key=lambda i: i.published, reverse=True)]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NewsStore":
        return cls(NewsItem.from_dict(x) for x in d.get("items", []))
