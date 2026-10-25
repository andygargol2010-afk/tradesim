"""Calendar of market-moving events for the next month, rebuilt on every refresh.

Sources, from most to least authoritative:

1. **Federal Reserve**: FOMC meeting dates scraped from federalreserve.gov (built-in fallback list);
   minutes are three weeks after each meeting.
2. **Forex Factory weekly calendar** (this week): exact times, forecasts and previous values of the
   week's releases for the US and the other big economies.
3. **Company calendars** (Yahoo): earnings dates with consensus EPS, ex-dividend dates.
4. **Exchange calendar**: US market holidays, monthly options expiry (3rd Friday; quarterly = quad witching).
5. **Release rules** for the main US statistics when no confirmed date is known yet (jobs report on the
   first Friday, CPI around the 12th...). These are flagged ``confirmed: False`` and are replaced as soon
   as a confirmed source lists the release, or corrected by the Claude calendar researcher
   (:class:`tradesim.research.analyst.CalendarResearcher`) which also adds events none of the feeds
   cover (other central banks, Treasury refunding, elections, crypto-specific events...).
"""

from __future__ import annotations

import calendar as _cal
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from tradesim.research.http import fetch_text

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MONTHS = {m: i for i, m in enumerate(_cal.month_name) if m}

# Published schedule (fallback if the Fed page cannot be read). (month, first day, second day, SEP?)
FOMC_FALLBACK: dict[int, list[tuple[int, int, int, bool]]] = {
    2026: [(1, 27, 28, False), (3, 17, 18, True), (4, 28, 29, False), (6, 16, 17, True), (7, 28, 29, False),
           (9, 15, 16, True), (10, 27, 28, False), (12, 8, 9, True)],
    2027: [(1, 26, 27, False), (3, 16, 17, True), (4, 27, 28, False), (6, 8, 9, True), (7, 27, 28, False),
           (9, 14, 15, True), (10, 26, 27, False), (12, 7, 8, True)],
}

ALL_EQUITIES = "*equities"  # placeholder symbol tags expanded by consumers
ALL_CRYPTO = "*crypto"
ALL = "*all"

IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Event:
    """One calendar entry. ``when`` is ISO (UTC) when the time is known, else just the date."""

    date: str  # YYYY-MM-DD (New York date)
    title: str
    category: str  # central_bank | inflation | employment | growth | consumer | earnings | dividend |
    #                options_expiry | holiday | politics | crypto | commodities | other
    importance: str = "medium"  # high | medium | low
    country: str = "US"
    when: str | None = None
    symbols: list[str] = field(default_factory=list)
    forecast: str | None = None
    previous: str | None = None
    actual: str | None = None
    confirmed: bool = True
    source: str = ""
    notes: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = event_id(self.date, self.title, self.country)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def event_id(day: str, title: str, country: str = "") -> str:
    key = f"{day}|{country}|{re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _ny(d: date, hh: int, mm: int) -> str:
    return datetime.combine(d, time(hh, mm), NY).astimezone(timezone.utc).isoformat()


# ------------------------------------------------------------------------- date rules
def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of the month; n = -1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year, month, _cal.monthrange(year, month)[1])
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def jobs_report_date(year: int, month: int) -> date:
    """BLS rule: the third Friday after the end of the reference week (the week containing the 12th) of the
    previous month. Usually the first Friday of the month."""
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    d12 = date(py, pm, 12)
    saturday = d12 + timedelta(days=(5 - d12.weekday()) % 7)
    return saturday + timedelta(days=20)


def business_day(year: int, month: int, n: int, holidays: set[date]) -> date:
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() < 5 and d not in holidays:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def closest_weekday(year: int, month: int, day: int, allowed: Iterable[int], holidays: set[date]) -> date:
    target = date(year, month, min(day, _cal.monthrange(year, month)[1]))
    for k in range(0, 10):
        for d in (target + timedelta(days=k), target - timedelta(days=k)):
            if d.month == month and d.weekday() in allowed and d not in holidays:
                return d
    return target


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def rule_based_events(start: date, end: date, holidays: set[date]) -> list[Event]:
    """Approximate dates of the main US releases (``confirmed=False``) plus exact exchange events."""
    ev: list[Event] = []
    est = "estimated date (BLS/BEA usual schedule); confirmed when the week's calendar lists it"

    def add(d: date, title: str, cat: str, imp: str, hh: int, mm: int, syms: list[str], confirmed=False, notes=est):
        if start <= d <= end:
            ev.append(Event(d.isoformat(), title, cat, imp, "US", _ny(d, hh, mm), syms, confirmed=confirmed,
                            source="rules", notes=notes))

    for y, m in _months(start, end):
        add(jobs_report_date(y, m), "Employment Situation (nonfarm payrolls, unemployment rate)", "employment",
            "high", 8, 30, [ALL])
        add(closest_weekday(y, m, 12, (1, 2, 3), holidays), "CPI (consumer prices)", "inflation", "high", 8, 30, [ALL])
        add(closest_weekday(y, m, 14, (1, 2, 3, 4), holidays), "PPI (producer prices)", "inflation", "medium", 8, 30,
            [ALL_EQUITIES, "TLT"])
        add(closest_weekday(y, m, 16, (1, 2, 3, 4), holidays), "Retail sales", "consumer", "medium", 8, 30,
            [ALL_EQUITIES])
        add(nth_weekday(y, m, 4, -1), "PCE inflation / personal income and outlays", "inflation", "high", 8, 30, [ALL])
        gdp_kind = {1: "advance", 4: "advance", 7: "advance", 10: "advance", 2: "second", 5: "second", 8: "second",
                    11: "second"}.get(m, "third")
        add(nth_weekday(y, m, 3, -1), f"GDP ({gdp_kind} estimate)", "growth", "high" if gdp_kind == "advance" else "medium",
            8, 30, [ALL])
        add(business_day(y, m, 1, holidays), "ISM manufacturing PMI", "growth", "medium", 10, 0, [ALL_EQUITIES])
        add(business_day(y, m, 3, holidays), "ISM services PMI", "growth", "medium", 10, 0, [ALL_EQUITIES])
        add(nth_weekday(y, m, 4, 2), "UMich consumer sentiment (preliminary)", "consumer", "low", 10, 0,
            [ALL_EQUITIES])
        opex = nth_weekday(y, m, 4, 3)
        while opex in holidays:
            opex -= timedelta(days=1)
        quad = m in (3, 6, 9, 12)
        add(opex, "Quarterly options/futures expiry (quad witching)" if quad else "Monthly options expiry", "options_expiry",
            "medium" if quad else "low", 16, 0, [ALL_EQUITIES], confirmed=True,
            notes="expiry of index and equity options; volume and volatility spikes are common")
    d = start
    while d <= end:
        if d.weekday() == 3 and d not in holidays:
            add(d, "Initial jobless claims", "employment", "low", 8, 30, [ALL_EQUITIES, "TLT"], confirmed=True,
                notes="published every Thursday")
        d += timedelta(days=1)
    return ev


def holiday_events(start: date, end: date) -> tuple[list[Event], set[date]]:
    from tradesim.live import nyse_holidays

    hols: set[date] = set()
    for y in range(start.year, end.year + 1):
        hols |= set(nyse_holidays(y))
    ev = [Event(h.isoformat(), "US stock market holiday (NYSE/Nasdaq closed)", "holiday", "medium", "US",
                symbols=[ALL_EQUITIES], source="exchange", notes="no stock trading; crypto keeps trading")
          for h in sorted(hols) if start <= h <= end]
    return ev, hols


# ----------------------------------------------------------------------------- the Fed
def parse_fomc_page(page: str) -> dict[int, list[tuple[int, int, int, bool]]]:
    """Meeting dates by year from the Fed's FOMC calendar page."""
    out: dict[int, list[tuple[int, int, int, bool]]] = {}
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"(\d{4}) FOMC Meetings", page)]
    for k, (pos, year) in enumerate(heads):
        seg = page[pos: heads[k + 1][0] if k + 1 < len(heads) else len(page)]
        rows = re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<',
                          seg, re.S)
        meetings = []
        for month_txt, days_txt in rows:
            month_name = month_txt.strip().split("/")[-1].strip()
            nums = [int(x) for x in re.findall(r"\d+", days_txt)]
            if month_name not in MONTHS or not nums:
                continue
            m = MONTHS[month_name]
            d1, d2 = (nums[0], nums[-1]) if len(nums) > 1 else (nums[0], nums[0])
            if d2 < d1:  # meeting spanning two months (e.g. "Apr/May 30-1")
                m = MONTHS.get(month_txt.strip().split("/")[0].strip(), m)
            meetings.append((m, d1, d2, "*" in days_txt))
        if meetings:
            out[year] = meetings
    return out


def fomc_events(start: date, end: date, schedule: dict[int, list[tuple[int, int, int, bool]]],
                source: str) -> list[Event]:
    ev = []
    for year, meetings in schedule.items():
        for m, d1, d2, sep in meetings:
            try:
                first = date(year, m, d1)
                last = date(year, m, d2) if d2 >= d1 else date(year + (m == 12), m % 12 + 1, d2)
            except ValueError:
                continue
            if start <= last <= end:
                ev.append(Event(last.isoformat(), "FOMC rate decision" + (" + economic projections (SEP)" if sep else "")
                                + " and Powell press conference", "central_bank", "high", "US", _ny(last, 14, 0),
                                [ALL], source=source, notes=f"meeting {first.isoformat()} to {last.isoformat()}; "
                                "statement 14:00 ET, press conference 14:30 ET"))
            minutes = last + timedelta(weeks=3)
            if start <= minutes <= end:
                ev.append(Event(minutes.isoformat(), "FOMC minutes", "central_bank", "medium", "US",
                                _ny(minutes, 14, 0), [ALL], confirmed=True, source=source,
                                notes=f"minutes of the {last.isoformat()} meeting (published 3 weeks later)"))
    return ev


# ------------------------------------------------------------------------ Forex Factory
_FF_CATEGORY = [
    (r"FOMC|Fed |Federal Funds|Powell|Rate Decision|Monetary Policy|Cash Rate|Official Bank Rate|Main Refinancing|"
     r"BOJ|ECB|BOE|SNB|RBA|BOC|RBNZ|Press Conference|Minutes", "central_bank"),
    (r"CPI|PPI|PCE|Inflation|Price Index", "inflation"),
    (r"Non-Farm|Unemployment|Jobless|Employment|ADP|JOLTS|Payroll|Earnings", "employment"),
    (r"GDP|PMI|ISM|Industrial|Durable|Factory|Production", "growth"),
    (r"Retail|Consumer|Sentiment|Confidence|Spending|Income", "consumer"),
    (r"Crude|Oil|Natural Gas", "commodities"),
    (r"Holiday", "holiday"),
    (r"Election|Vote|Referendum", "politics"),
]
_FF_COUNTRY_SYMBOLS = {"USD": [ALL], "EUR": [ALL_EQUITIES, "EEM"], "CNY": ["EEM", ALL_EQUITIES],
                       "JPY": [ALL_EQUITIES], "GBP": [ALL_EQUITIES]}
CURRENCY_COUNTRY = {"USD": "US", "EUR": "EU", "GBP": "UK", "JPY": "JP", "CNY": "CN", "CAD": "CA", "AUD": "AU",
                    "NZD": "NZ", "CHF": "CH"}
# Covered by the Fed's own entry (decision + statement + press conference in one event).
_FF_SKIP = re.compile(r"^FOMC (Statement|Press Conference)$", re.I)


def parse_forex_factory(text: str, countries_all: Iterable[str] = ("USD",),
                        countries_high: Iterable[str] = ("EUR", "GBP", "JPY", "CNY", "CAD")) -> list[Event]:
    """This week's events: every US release with medium/high impact, high-impact ones elsewhere."""
    ev = []
    countries_all, countries_high = set(countries_all), set(countries_high)
    for row in json.loads(text):
        country, impact = row.get("country", ""), (row.get("impact") or "").lower()
        if impact == "holiday" or impact == "non-economic":
            continue
        keep = (country in countries_all and impact in ("high", "medium")) or (country in countries_high and impact == "high")
        if not keep:
            continue
        try:
            when = datetime.fromisoformat(row["date"])
        except (KeyError, ValueError):
            continue
        title = row.get("title", "").strip()
        if country == "USD" and _FF_SKIP.match(title):
            continue
        cat = next((c for pat, c in _FF_CATEGORY if re.search(pat, title, re.I)), "other")
        ev.append(Event(when.astimezone(NY).date().isoformat(), title, cat, impact, CURRENCY_COUNTRY.get(country, country),
                        when.astimezone(timezone.utc).isoformat(), list(_FF_COUNTRY_SYMBOLS.get(country, [ALL_EQUITIES])),
                        forecast=row.get("forecast") or None, previous=row.get("previous") or None,
                        source="forexfactory"))
    return ev


# ------------------------------------------------------------------------- companies
def company_events(symbols: list[str], start: date, end: date,
                   calendar_fn: Callable[[str], dict[str, Any]] | None = None) -> tuple[list[Event], dict[str, str]]:
    calendar_fn = calendar_fn or _yahoo_calendar
    ev, errors = [], {}
    for sym in symbols:
        try:
            cal = calendar_fn(sym) or {}
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)[:160]
            continue
        dates = cal.get("Earnings Date") or []
        if isinstance(dates, (date, datetime)):
            dates = [dates]
        if dates:
            d = dates[0].date() if isinstance(dates[0], datetime) else dates[0]
            if start <= d <= end:
                eps = cal.get("Earnings Average")
                rev = cal.get("Revenue Average")
                fc = []
                if eps is not None:
                    fc.append(f"EPS {eps:.2f}")
                if rev:
                    fc.append(f"revenue {rev / 1e9:.1f}B")
                window = len(dates) > 1 and dates[-1] != dates[0]
                ev.append(Event(d.isoformat(), f"{sym} quarterly earnings", "earnings", "high", "US", None, [sym],
                                forecast=", ".join(fc) or None, confirmed=not window, source="yahoo",
                                notes=("date window " + " to ".join(str(x) for x in (dates[0], dates[-1]))) if window
                                else "consensus estimates from Yahoo"))
        exd = cal.get("Ex-Dividend Date")
        if isinstance(exd, datetime):
            exd = exd.date()
        if isinstance(exd, date) and start <= exd <= end:
            ev.append(Event(exd.isoformat(), f"{sym} ex-dividend date", "dividend", "low", "US", None, [sym],
                            source="yahoo", notes="buyers from this date do not receive the next dividend"))
    return ev, errors


def _yahoo_calendar(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    return yf.Ticker(symbol).calendar


# ------------------------------------------------------------------------------ merge
# Releases known under different names by different sources (same group = same release).
_SAME_RELEASE = [("cpi", "consumer price"), ("ppi", "producer price"), ("pce",), ("payroll", "non-farm",
                 "employment situation"), ("jobless", "unemployment claims"), ("gdp",), ("ism manufacturing",
                 "manufacturing pmi"), ("ism services", "services pmi"), ("retail sales",), ("sentiment",),
                 ("fomc rate decision", "federal funds rate"), ("fomc minutes",)]
_SAME_DAY_ONLY = {("jobless", "unemployment claims")}


def _similar(a: Event, b: Event) -> bool:
    if a.category != b.category or a.country != b.country:
        return False
    da, db = date.fromisoformat(a.date), date.fromisoformat(b.date)
    if abs((da - db).days) > 6:
        return False
    ta, tb = a.title.lower(), b.title.lower()
    for group in _SAME_RELEASE:
        if any(k in ta for k in group) and any(k in tb for k in group):
            return da == db if group in _SAME_DAY_ONLY else True
    return False


def merge_events(events: list[Event]) -> list[Event]:
    """Deduplicate: rule-based entries (``source == "rules"``) give way to any other source listing the same
    release; otherwise the first occurrence of an id wins."""
    others = list({e.id: e for e in events if e.source != "rules"}.values())
    out = list(others)
    ids = {e.id for e in out}
    for e in events:
        if e.source == "rules" and e.id not in ids and not any(_similar(e, c) for c in others):
            out.append(e)
            ids.add(e.id)
    return sorted(out, key=lambda e: (e.date, e.when or "99", IMPORTANCE_RANK.get(e.importance, 3)))


class CalendarBuilder:
    """Assembles the calendar. Every network call is injectable (``fetch``, ``calendar_fn``) for tests."""

    def __init__(self, symbols: list[str], equity_symbols: list[str], days_ahead: int = 30, days_back: int = 7,
                 fetch: Callable[[str], str] | None = None, calendar_fn: Callable[[str], dict[str, Any]] | None = None):
        self.symbols = symbols
        self.equities = equity_symbols
        self.days_ahead, self.days_back = days_ahead, days_back
        self.fetch = fetch or (lambda url: fetch_text(url, timeout=20, retries=2))
        self.calendar_fn = calendar_fn
        self.errors: dict[str, str] = {}
        self.sources: dict[str, str] = {}

    def build(self, previous: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        today = now.astimezone(NY).date()
        start, end = today - timedelta(days=self.days_back), today + timedelta(days=self.days_ahead)
        prev_events = [Event.from_dict(e) for e in (previous or {}).get("events", [])]
        hol_ev, hols = holiday_events(start, end)
        confirmed: list[Event] = list(hol_ev)

        schedule, src = FOMC_FALLBACK, "federalreserve (built-in schedule)"
        try:
            parsed = parse_fomc_page(self.fetch(FOMC_URL))
            if parsed:
                schedule, src = {**FOMC_FALLBACK, **parsed}, "federalreserve.gov"
        except Exception as exc:  # noqa: BLE001
            self.errors["fomc"] = str(exc)[:200]
        confirmed += fomc_events(start, end, schedule, src)
        self.sources["fomc"] = src

        try:
            ff = parse_forex_factory(self.fetch(FF_URL))
            confirmed += ff
            self.sources["forexfactory"] = f"{len(ff)} events this week"
        except Exception as exc:  # noqa: BLE001
            self.errors["forexfactory"] = str(exc)[:200]
        # Keep Forex Factory rows from previous refreshes (the feed only shows the current week).
        seen = {e.id for e in confirmed}
        confirmed += [e for e in prev_events if e.source in ("forexfactory", "claude-web") and e.id not in seen
                      and start.isoformat() <= e.date <= end.isoformat()]

        comp, errs = company_events(self.equities, start, end, self.calendar_fn)
        confirmed += comp
        self.errors.update({f"company:{k}": v for k, v in errs.items()})
        self.sources["companies"] = f"{len(comp)} earnings/dividend events"

        events = merge_events(confirmed + rule_based_events(start, end, hols))
        # Carry over what the researcher added or verified (actual results, corrections).
        by_id = {e.id: e for e in prev_events}
        for e in events:
            old = by_id.get(e.id)
            if old is not None:
                e.actual = e.actual or old.actual
                if old.source == "claude-web" and old.notes and old.notes not in e.notes:
                    e.notes = (e.notes + " | " + old.notes).strip(" |")
        return {"generated_at": now.isoformat(), "window": [start.isoformat(), end.isoformat()],
                "events": [e.to_dict() for e in events], "sources": dict(self.sources), "errors": dict(self.errors),
                "researched_at": (previous or {}).get("researched_at")}


def upcoming(cal: dict[str, Any], now: datetime, days: int = 30, min_importance: str = "low") -> list[dict[str, Any]]:
    today = now.astimezone(NY).date()
    lim = IMPORTANCE_RANK[min_importance]
    return [e for e in cal.get("events", []) if today.isoformat() <= e["date"] <= (today + timedelta(days=days)).isoformat()
            and IMPORTANCE_RANK.get(e["importance"], 3) <= lim]


def calendar_brief(events: list[dict[str, Any]]) -> str:
    rows = []
    for e in events:
        t = ""
        if e.get("when"):
            t = datetime.fromisoformat(e["when"]).astimezone(NY).strftime(" %H:%M ET")
        extra = []
        for k in ("forecast", "previous", "actual"):
            if e.get(k):
                extra.append(f"{k} {e[k]}")
        rows.append(f"[{e['id']}] {e['date']}{t} {e['country']} {e['title']} ({e['importance']}"
                    + ("" if e.get("confirmed", True) else ", estimated date") + ")"
                    + (" " + ", ".join(extra) if extra else ""))
    return "\n".join(rows)
