"""Macroeconomic context by asset type: official data (FRED, no key) plus live market gauges (Yahoo).

:class:`MacroMonitor` builds a snapshot of the indicators in :data:`INDICATORS` (latest value, previous
value, change, date). :func:`relevant_indicators` says which of them matter for a given asset: rates and
inflation for bonds, real yields and the dollar for gold and crypto, oil for energy, the yield curve and
credit spreads for banks, and so on. Official series are refreshed at most every ``official_ttl_hours``
(they change monthly or weekly); market gauges on every refresh.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.research.http import fetch_text
from tradesim.types import AssetClass

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"


@dataclass(frozen=True)
class Indicator:
    id: str
    name: str  # Spanish label shown on the dashboard
    source: str  # "fred" | "yahoo"
    series: str
    transform: str = "level"  # level | yoy (% vs 12 obs earlier) | diff (change vs previous obs)
    unit: str = ""
    description: str = ""


INDICATORS: list[Indicator] = [
    # --- monetary policy and rates
    Indicator("fed_funds", "Tipo efectivo de la Fed", "fred", "DFF", unit="%",
              description="Effective federal funds rate (daily)"),
    Indicator("ust_2y", "Bono EE. UU. 2 años", "fred", "DGS2", unit="%"),
    Indicator("ust_10y", "Bono EE. UU. 10 años", "fred", "DGS10", unit="%"),
    Indicator("curve_10y_2y", "Curva 10a-2a", "fred", "T10Y2Y", unit="pp",
              description="10y minus 2y Treasury spread; negative = inverted curve"),
    Indicator("real_10y", "Tipo real 10 años (TIPS)", "fred", "DFII10", unit="%"),
    Indicator("hy_spread", "Diferencial high yield", "fred", "BAMLH0A0HYM2", unit="pp",
              description="ICE BofA US high-yield option-adjusted spread: credit stress"),
    # --- inflation
    Indicator("cpi_yoy", "IPC interanual", "fred", "CPIAUCSL", "yoy", "%"),
    Indicator("core_cpi_yoy", "IPC subyacente interanual", "fred", "CPILFESL", "yoy", "%"),
    Indicator("core_pce_yoy", "PCE subyacente interanual", "fred", "PCEPILFE", "yoy", "%",
              description="The Fed's preferred inflation gauge"),
    # --- labour and activity
    Indicator("unemployment", "Tasa de paro", "fred", "UNRATE", unit="%"),
    Indicator("payrolls_change", "Empleo no agrícola (cambio mensual)", "fred", "PAYEMS", "diff", "miles"),
    Indicator("jobless_claims", "Peticiones semanales de paro", "fred", "ICSA", unit="personas"),
    Indicator("gdp_growth", "PIB real (trimestral anualizado)", "fred", "A191RL1Q225SBEA", unit="%"),
    Indicator("retail_sales_yoy", "Ventas minoristas interanual", "fred", "RSAFS", "yoy", "%"),
    Indicator("consumer_sentiment", "Confianza del consumidor (UMich)", "fred", "UMCSENT"),
    Indicator("m2_yoy", "Masa monetaria M2 interanual", "fred", "M2SL", "yoy", "%",
              description="Liquidity; often linked to crypto and risk appetite"),
    # --- market gauges (live)
    Indicator("vix", "VIX (volatilidad S&P 500)", "yahoo", "^VIX"),
    Indicator("ust_10y_live", "Rendimiento 10 años (mercado)", "yahoo", "^TNX", unit="%"),
    Indicator("ust_3m_live", "Letra 3 meses (mercado)", "yahoo", "^IRX", unit="%"),
    Indicator("dxy", "Índice dólar (DXY)", "yahoo", "DX-Y.NYB"),
    Indicator("wti", "Petróleo WTI", "yahoo", "CL=F", unit="USD"),
    Indicator("gold", "Oro (futuro)", "yahoo", "GC=F", unit="USD"),
    Indicator("copper", "Cobre (futuro)", "yahoo", "HG=F", unit="USD",
              description="Doctor copper: global industrial demand"),
]
BY_ID = {i.id: i for i in INDICATORS}

_CORE = ["fed_funds", "ust_10y_live", "vix", "hy_spread"]
_EQUITY = _CORE + ["curve_10y_2y", "core_cpi_yoy", "core_pce_yoy", "unemployment", "payrolls_change",
                   "jobless_claims", "gdp_growth", "consumer_sentiment"]
RELEVANCE: dict[str, list[str]] = {
    "equity_broad": _EQUITY + ["retail_sales_yoy", "dxy"],
    "Information Technology": _CORE + ["real_10y", "ust_10y", "core_cpi_yoy", "gdp_growth", "dxy"],
    "Communication Services": _CORE + ["real_10y", "consumer_sentiment", "retail_sales_yoy"],
    "Consumer Discretionary": _CORE + ["retail_sales_yoy", "consumer_sentiment", "unemployment", "jobless_claims",
                                       "core_cpi_yoy"],
    "Consumer Staples": _CORE + ["cpi_yoy", "retail_sales_yoy", "consumer_sentiment", "dxy"],
    "Financials": _CORE + ["curve_10y_2y", "ust_2y", "ust_10y", "unemployment", "gdp_growth"],
    "Energy": _CORE + ["wti", "dxy", "gdp_growth", "cpi_yoy"],
    "Health Care": _CORE + ["cpi_yoy", "ust_10y"],
    "Industrials": _CORE + ["copper", "gdp_growth", "dxy", "payrolls_change"],
    "bonds": ["fed_funds", "ust_2y", "ust_10y", "ust_10y_live", "ust_3m_live", "curve_10y_2y", "core_cpi_yoy",
              "core_pce_yoy", "payrolls_change", "unemployment", "gdp_growth"],
    "precious_metals": ["real_10y", "dxy", "fed_funds", "cpi_yoy", "vix", "gold"],
    "emerging": _CORE + ["dxy", "copper", "wti"],
    "crypto": ["fed_funds", "real_10y", "dxy", "m2_yoy", "vix", "hy_spread", "ust_10y_live"],
}
ETF_PROFILE = {"TLT": "bonds", "IEF": "bonds", "SHY": "bonds", "BND": "bonds", "AGG": "bonds",
               "GLD": "precious_metals", "IAU": "precious_metals", "SLV": "precious_metals",
               "EEM": "emerging", "VWO": "emerging", "XLE": "Energy", "XLF": "Financials", "XLK": "Information Technology",
               "XLV": "Health Care", "XLP": "Consumer Staples", "XLY": "Consumer Discretionary", "XLI": "Industrials"}


def asset_profile(asset: Asset) -> str:
    """Key of :data:`RELEVANCE` describing what moves this asset."""
    if asset.asset_class is AssetClass.CRYPTO:
        return "crypto"
    if asset.asset_class is AssetClass.ETF:
        return ETF_PROFILE.get(asset.symbol, "equity_broad")
    if asset.sector in RELEVANCE:
        return asset.sector
    return "equity_broad"


def relevant_indicators(asset: Asset) -> list[str]:
    return RELEVANCE[asset_profile(asset)]


def _fred_frame(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def _summarise(ind: Indicator, s: pd.Series) -> dict[str, Any]:
    if ind.transform == "yoy":
        s = (s / s.shift(12) - 1) * 100
    elif ind.transform == "diff":
        s = s.diff()
    s = s.dropna()
    if s.empty:
        raise ValueError(f"no data for {ind.series}")
    value, prev = float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else None
    out = {"id": ind.id, "name": ind.name, "value": round(value, 4), "previous": None if prev is None else round(prev, 4),
           "change": None if prev is None else round(value - prev, 4), "as_of": s.index[-1].date().isoformat(),
           "unit": ind.unit, "source": ind.source, "series": ind.series}
    if len(s) > 22 and ind.source == "yahoo":
        base = float(s.iloc[-22])
        out["change_1m_pct"] = round((value / base - 1) * 100, 2) if base else None
    if len(s) >= 4:
        last3 = s.iloc[-4:].to_numpy()
        out["trend"] = "up" if last3[-1] > last3[0] else "down" if last3[-1] < last3[0] else "flat"
    return out


class MacroMonitor:
    """Builds and refreshes the macro snapshot. ``fetch`` / ``yahoo`` are injectable for tests."""

    def __init__(self, indicators: list[Indicator] | None = None, fetch: Callable[[str], str] | None = None,
                 yahoo: Callable[[list[str]], dict[str, pd.Series]] | None = None, official_ttl_hours: float = 12.0):
        self.indicators = indicators or INDICATORS
        # FRED's CDN stalls requests with browser-like user agents; a plain client UA is served at once.
        self.fetch = fetch or (lambda url: fetch_text(url, timeout=20, retries=1, backoff=1.0,
                                                      headers={"User-Agent": "curl/8.5.0"}))
        self.yahoo = yahoo or _yahoo_closes
        self.ttl = timedelta(hours=official_ttl_hours)
        self.errors: dict[str, str] = {}

    def refresh(self, previous: dict[str, Any] | None = None, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        prev = (previous or {}).get("indicators", {})
        out: dict[str, Any] = {}
        start = (now - timedelta(days=800)).date().isoformat()
        for ind in self.indicators:
            if ind.source != "fred":
                continue
            old = prev.get(ind.id)
            if old and old.get("fetched_at") and now - datetime.fromisoformat(old["fetched_at"]) < self.ttl:
                out[ind.id] = old
                continue
            try:
                row = _summarise(ind, _fred_frame(self.fetch(FRED_CSV.format(series=ind.series, start=start))))
                row["fetched_at"] = now.isoformat()
                out[ind.id] = row
            except Exception as exc:  # noqa: BLE001 - keep the stale value rather than nothing
                self.errors[ind.id] = str(exc)[:200]
                if old:
                    out[ind.id] = {**old, "stale": True}
        market = [i for i in self.indicators if i.source == "yahoo"]
        if market:
            try:
                closes = self.yahoo([i.series for i in market])
            except Exception as exc:  # noqa: BLE001
                closes = {}
                self.errors["yahoo"] = str(exc)[:200]
            for ind in market:
                s = closes.get(ind.series)
                try:
                    if s is None:
                        raise ValueError("no data")
                    out[ind.id] = {**_summarise(ind, s), "fetched_at": now.isoformat()}
                except Exception as exc:  # noqa: BLE001
                    self.errors[ind.id] = str(exc)[:200]
                    if prev.get(ind.id):
                        out[ind.id] = {**prev[ind.id], "stale": True}
        return {"generated_at": now.isoformat(), "indicators": out, "errors": dict(self.errors)}


def _yahoo_closes(tickers: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf

    df = yf.download(tickers, period="3mo", interval="1d", progress=False, auto_adjust=True, threads=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return {t: df[t].dropna() for t in tickers if t in df}


def macro_brief(snapshot: dict[str, Any], ids: list[str] | None = None) -> str:
    """One line per indicator, for prompts."""
    rows = []
    for k, r in snapshot.get("indicators", {}).items():
        if ids is not None and k not in ids:
            continue
        ch = r.get("change")
        extra = f" (prev {r['previous']:g}, chg {ch:+.3g})" if ch is not None and r.get("previous") is not None else ""
        m = f", 1m {r['change_1m_pct']:+.1f}%" if r.get("change_1m_pct") is not None else ""
        rows.append(f"{k}: {r['value']:g}{r.get('unit') and ' ' + r['unit'] or ''}{extra}{m} as of {r['as_of']}"
                    + (" [stale]" if r.get("stale") else ""))
    return "\n".join(rows)
