"""Build / refresh the bundled asset universe CSV.

Sources:
* Wikipedia constituent tables: S&P 500, Nasdaq-100, S&P MidCap 400, S&P SmallCap 600.
* CoinGecko ``/coins/markets`` (free, no key): top crypto by market cap, excluding
  stablecoins and wrapped / staked / bridged derivatives of other coins.
* Curated lists below: major ETFs, FX pairs, benchmark indices.

Usage::

    python scripts/build_universe.py                      # writes tradesim/assets/data/universe.csv
    python scripts/build_universe.py --out my_universe.csv --crypto 300

Note: today's index constituents applied to past dates introduce survivorship
bias in backtests. See the README section on data caveats.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim.assets.universe import Asset, AssetUniverse  # noqa: E402
from tradesim.types import AssetClass  # noqa: E402

UA = {"User-Agent": "tradesim-universe-builder/0.1 (research; contact: none)"}

WIKI_INDEXES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

# Mega caps with the tightest spreads / deepest books (tier 1).
MEGA_CAPS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY",
    "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX", "WMT", "ABBV", "BAC", "CRM", "ORCL",
    "AMD", "KO", "PEP", "CVX", "MRK", "ADBE", "TMO", "CSCO", "ACN", "MCD", "INTC", "QCOM", "WFC",
    "DIS", "IBM", "PLTR", "GS", "MS", "CAT", "TXN", "AMAT", "MU", "PFE", "T", "VZ", "INTU", "NOW",
}

# (symbol, name, liquidity tier). Tier 1 = deepest ETFs in the world.
ETFS: list[tuple[str, str, int]] = [
    ("SPY", "SPDR S&P 500 ETF Trust", 1), ("IVV", "iShares Core S&P 500", 1), ("VOO", "Vanguard S&P 500", 1),
    ("QQQ", "Invesco QQQ Trust", 1), ("IWM", "iShares Russell 2000", 1), ("DIA", "SPDR Dow Jones Industrial Average", 1),
    ("VTI", "Vanguard Total Stock Market", 1), ("EFA", "iShares MSCI EAFE", 1), ("EEM", "iShares MSCI Emerging Markets", 1),
    ("TLT", "iShares 20+ Year Treasury Bond", 1), ("GLD", "SPDR Gold Shares", 1), ("HYG", "iShares iBoxx High Yield Corp Bond", 1),
    ("LQD", "iShares iBoxx Investment Grade Corp Bond", 1), ("XLF", "Financial Select Sector SPDR", 1),
    ("XLE", "Energy Select Sector SPDR", 1), ("XLK", "Technology Select Sector SPDR", 1), ("XLV", "Health Care Select Sector SPDR", 1),
    ("XLI", "Industrial Select Sector SPDR", 1), ("XLY", "Consumer Discretionary Select Sector SPDR", 1),
    ("XLP", "Consumer Staples Select Sector SPDR", 1), ("XLU", "Utilities Select Sector SPDR", 1),
    ("XLB", "Materials Select Sector SPDR", 2), ("XLRE", "Real Estate Select Sector SPDR", 2),
    ("XLC", "Communication Services Select Sector SPDR", 2), ("SMH", "VanEck Semiconductor", 1),
    ("SOXX", "iShares Semiconductor", 2), ("ARKK", "ARK Innovation", 2), ("VEA", "Vanguard FTSE Developed Markets", 1),
    ("VWO", "Vanguard FTSE Emerging Markets", 1), ("AGG", "iShares Core US Aggregate Bond", 1), ("BND", "Vanguard Total Bond Market", 1),
    ("IEF", "iShares 7-10 Year Treasury Bond", 1), ("SHY", "iShares 1-3 Year Treasury Bond", 1), ("SLV", "iShares Silver Trust", 1),
    ("USO", "United States Oil Fund", 1), ("UNG", "United States Natural Gas Fund", 2), ("GDX", "VanEck Gold Miners", 1),
    ("GDXJ", "VanEck Junior Gold Miners", 2), ("KRE", "SPDR S&P Regional Banking", 1), ("KBE", "SPDR S&P Bank", 2),
    ("XBI", "SPDR S&P Biotech", 1), ("IBB", "iShares Biotechnology", 2), ("XRT", "SPDR S&P Retail", 2),
    ("XHB", "SPDR S&P Homebuilders", 2), ("ITB", "iShares US Home Construction", 2), ("XOP", "SPDR S&P Oil & Gas Exploration", 2),
    ("OIH", "VanEck Oil Services", 2), ("JETS", "US Global Jets", 3), ("TAN", "Invesco Solar", 3), ("ICLN", "iShares Global Clean Energy", 2),
    ("LIT", "Global X Lithium & Battery Tech", 3), ("URA", "Global X Uranium", 2), ("COPX", "Global X Copper Miners", 3),
    ("VNQ", "Vanguard Real Estate", 1), ("IYR", "iShares US Real Estate", 2), ("SCHD", "Schwab US Dividend Equity", 1),
    ("VIG", "Vanguard Dividend Appreciation", 1), ("VYM", "Vanguard High Dividend Yield", 1), ("DVY", "iShares Select Dividend", 2),
    ("SDY", "SPDR S&P Dividend", 2), ("NOBL", "ProShares S&P 500 Dividend Aristocrats", 2), ("VUG", "Vanguard Growth", 1),
    ("VTV", "Vanguard Value", 1), ("IWF", "iShares Russell 1000 Growth", 1), ("IWD", "iShares Russell 1000 Value", 1),
    ("IWB", "iShares Russell 1000", 2), ("IWO", "iShares Russell 2000 Growth", 2), ("IWN", "iShares Russell 2000 Value", 2),
    ("IJH", "iShares Core S&P Mid-Cap", 1), ("IJR", "iShares Core S&P Small-Cap", 1), ("MDY", "SPDR S&P MidCap 400", 2),
    ("RSP", "Invesco S&P 500 Equal Weight", 1), ("SPLG", "SPDR Portfolio S&P 500", 1), ("SCHX", "Schwab US Large-Cap", 2),
    ("SCHB", "Schwab US Broad Market", 2), ("SCHG", "Schwab US Large-Cap Growth", 2), ("MTUM", "iShares MSCI USA Momentum Factor", 2),
    ("QUAL", "iShares MSCI USA Quality Factor", 2), ("USMV", "iShares MSCI USA Min Vol Factor", 2), ("VLUE", "iShares MSCI USA Value Factor", 3),
    ("SPLV", "Invesco S&P 500 Low Volatility", 2), ("SPHB", "Invesco S&P 500 High Beta", 3), ("EWJ", "iShares MSCI Japan", 1),
    ("EWZ", "iShares MSCI Brazil", 1), ("FXI", "iShares China Large-Cap", 1), ("MCHI", "iShares MSCI China", 2), ("KWEB", "KraneShares CSI China Internet", 1),
    ("EWG", "iShares MSCI Germany", 2), ("EWU", "iShares MSCI United Kingdom", 2), ("EWC", "iShares MSCI Canada", 2),
    ("EWA", "iShares MSCI Australia", 2), ("EWY", "iShares MSCI South Korea", 2), ("EWT", "iShares MSCI Taiwan", 2),
    ("INDA", "iShares MSCI India", 2), ("EWW", "iShares MSCI Mexico", 2), ("EZU", "iShares MSCI Eurozone", 2),
    ("VGK", "Vanguard FTSE Europe", 1), ("IEMG", "iShares Core MSCI Emerging Markets", 1), ("IEFA", "iShares Core MSCI EAFE", 1),
    ("ACWI", "iShares MSCI ACWI", 2), ("VXUS", "Vanguard Total International Stock", 1), ("VT", "Vanguard Total World Stock", 2),
    ("BIL", "SPDR Bloomberg 1-3 Month T-Bill", 1), ("SGOV", "iShares 0-3 Month Treasury Bond", 1), ("TIP", "iShares TIPS Bond", 2),
    ("MUB", "iShares National Muni Bond", 2), ("EMB", "iShares JP Morgan USD Emerging Markets Bond", 2), ("JNK", "SPDR Bloomberg High Yield Bond", 1),
    ("BNDX", "Vanguard Total International Bond", 2), ("VCIT", "Vanguard Intermediate-Term Corporate Bond", 2),
    ("VCSH", "Vanguard Short-Term Corporate Bond", 2), ("BSV", "Vanguard Short-Term Bond", 2), ("BIV", "Vanguard Intermediate-Term Bond", 2),
    ("GOVT", "iShares US Treasury Bond", 2), ("TLH", "iShares 10-20 Year Treasury Bond", 2), ("EDV", "Vanguard Extended Duration Treasury", 3),
    ("IAU", "iShares Gold Trust", 1), ("GLDM", "SPDR Gold MiniShares", 2), ("SIVR", "abrdn Physical Silver Shares", 3),
    ("PPLT", "abrdn Physical Platinum Shares", 3), ("DBC", "Invesco DB Commodity Index Tracking", 2), ("DBA", "Invesco DB Agriculture", 3),
    ("PDBC", "Invesco Optimum Yield Diversified Commodity", 2), ("UUP", "Invesco DB US Dollar Index Bullish", 2),
    ("FXE", "Invesco CurrencyShares Euro", 3), ("FXY", "Invesco CurrencyShares Japanese Yen", 3),
    ("TQQQ", "ProShares UltraPro QQQ (3x)", 1), ("SQQQ", "ProShares UltraPro Short QQQ (-3x)", 1),
    ("SPXL", "Direxion Daily S&P 500 Bull 3x", 2), ("SPXS", "Direxion Daily S&P 500 Bear 3x", 2), ("UPRO", "ProShares UltraPro S&P 500 (3x)", 2),
    ("SSO", "ProShares Ultra S&P 500 (2x)", 2), ("SDS", "ProShares UltraShort S&P 500 (-2x)", 2), ("SH", "ProShares Short S&P 500", 2),
    ("PSQ", "ProShares Short QQQ", 2), ("SOXL", "Direxion Daily Semiconductor Bull 3x", 1), ("SOXS", "Direxion Daily Semiconductor Bear 3x", 1),
    ("TNA", "Direxion Daily Small Cap Bull 3x", 2), ("TZA", "Direxion Daily Small Cap Bear 3x", 2), ("UVXY", "ProShares Ultra VIX Short-Term Futures", 1),
    ("VXX", "iPath Series B S&P 500 VIX Short-Term Futures ETN", 1), ("SVXY", "ProShares Short VIX Short-Term Futures", 2),
    ("TMF", "Direxion Daily 20+ Year Treasury Bull 3x", 2), ("TBT", "ProShares UltraShort 20+ Year Treasury", 2),
    ("IBIT", "iShares Bitcoin Trust", 1), ("FBTC", "Fidelity Wise Origin Bitcoin Fund", 1), ("GBTC", "Grayscale Bitcoin Trust", 1),
    ("ARKB", "ARK 21Shares Bitcoin", 2), ("BITB", "Bitwise Bitcoin", 2), ("ETHA", "iShares Ethereum Trust", 1), ("FETH", "Fidelity Ethereum Fund", 2),
    ("BITO", "ProShares Bitcoin Strategy", 2), ("CIBR", "First Trust NASDAQ Cybersecurity", 2), ("HACK", "Amplify Cybersecurity", 3),
    ("SKYY", "First Trust Cloud Computing", 3), ("BOTZ", "Global X Robotics & AI", 3), ("ROBO", "ROBO Global Robotics & Automation", 3),
    ("IGV", "iShares Expanded Tech-Software", 2), ("FDN", "First Trust Dow Jones Internet", 2), ("XME", "SPDR S&P Metals & Mining", 2),
    ("XAR", "SPDR S&P Aerospace & Defense", 3), ("ITA", "iShares US Aerospace & Defense", 2), ("PAVE", "Global X US Infrastructure", 2),
    ("MOAT", "VanEck Morningstar Wide Moat", 3), ("JEPI", "JPMorgan Equity Premium Income", 1), ("JEPQ", "JPMorgan Nasdaq Equity Premium Income", 1),
    ("QYLD", "Global X Nasdaq 100 Covered Call", 2), ("DGRO", "iShares Core Dividend Growth", 2), ("COWZ", "Pacer US Cash Cows 100", 2),
]

# 7 majors + liquid crosses, Yahoo symbols.
FOREX: list[tuple[str, str, int]] = [
    ("EURUSD=X", "EUR/USD", 1), ("USDJPY=X", "USD/JPY", 1), ("GBPUSD=X", "GBP/USD", 1), ("AUDUSD=X", "AUD/USD", 1),
    ("USDCAD=X", "USD/CAD", 1), ("USDCHF=X", "USD/CHF", 1), ("NZDUSD=X", "NZD/USD", 1), ("EURGBP=X", "EUR/GBP", 2),
    ("EURJPY=X", "EUR/JPY", 2), ("GBPJPY=X", "GBP/JPY", 2), ("EURCHF=X", "EUR/CHF", 2), ("AUDJPY=X", "AUD/JPY", 2),
    ("EURAUD=X", "EUR/AUD", 2), ("EURCAD=X", "EUR/CAD", 2), ("GBPCHF=X", "GBP/CHF", 2), ("CADJPY=X", "CAD/JPY", 2),
    ("AUDNZD=X", "AUD/NZD", 2), ("USDMXN=X", "USD/MXN", 2), ("USDCNH=X", "USD/CNH", 2), ("USDSGD=X", "USD/SGD", 3),
    ("USDHKD=X", "USD/HKD", 2), ("USDSEK=X", "USD/SEK", 3), ("USDNOK=X", "USD/NOK", 3), ("USDZAR=X", "USD/ZAR", 3),
    ("USDTRY=X", "USD/TRY", 4), ("USDINR=X", "USD/INR", 3), ("USDBRL=X", "USD/BRL", 3), ("USDKRW=X", "USD/KRW", 3),
]

INDICES: list[tuple[str, str]] = [
    ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq-100"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"), ("^VIX", "CBOE Volatility Index"), ("^TNX", "US 10Y Treasury Yield"), ("^FTSE", "FTSE 100"),
    ("^GDAXI", "DAX"), ("^FCHI", "CAC 40"), ("^STOXX50E", "Euro Stoxx 50"), ("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"),
]

STABLECOINS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDD", "PYUSD", "USDS", "FRAX", "BUSD", "GUSD", "USDP", "LUSD",
    "CRVUSD", "USD0", "USDX", "USDB", "SUSDE", "EURC", "EURS", "USDY", "RLUSD", "USD1", "USDG", "BFUSD", "USDF", "USYC",
    "SUSDS", "SYRUPUSDC", "USDTB", "USTB", "BUIDL", "OUSG", "XAUT", "PAXG", "USDAI", "USDO", "GHO", "DOLA", "AUSD", "USDZ",
}
DERIVATIVE_WORDS = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "wbtc", "steth", "binance-peg", "coinbase wrapped")


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        resp = requests.get(url, headers=UA, timeout=30, **kwargs)
        if resp.status_code == 429:
            time.sleep(2 ** (attempt + 1) * 5)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _wiki_constituents(url: str) -> pd.DataFrame:
    html = _get(url).text
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            continue
        cols = {re.sub(r"\[\d+\]", "", str(c)).strip().lower(): c for c in t.columns}
        sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("ticker symbol")
        name_col = cols.get("security") or cols.get("company") or cols.get("company name")
        if sym_col is None or name_col is None or len(t) < 90:
            continue
        sector_col = cols.get("gics sector") or cols.get("icb industry") or next(
            (v for k, v in cols.items() if "sector" in k or "industry" in k), None
        )
        out = pd.DataFrame({
            "symbol": t[sym_col].astype(str).str.strip().str.replace(".", "-", regex=False),
            "name": t[name_col].astype(str).str.strip(),
            "sector": t[sector_col].astype(str).str.strip() if sector_col is not None else "",
        })
        return out[out["symbol"].str.match(r"^[A-Z][A-Z0-9\-]*$")]
    raise RuntimeError(f"No constituent table found at {url}")


def build_equities(universe: AssetUniverse) -> None:
    tier_by_group = {"sp500": 2, "nasdaq100": 2, "sp400": 3, "sp600": 4}
    for group, url in WIKI_INDEXES.items():
        df = _wiki_constituents(url)
        print(f"  {group}: {len(df)} constituents")
        for row in df.itertuples(index=False):
            tier = 1 if row.symbol in MEGA_CAPS else tier_by_group[group]
            existing = universe.assets.get(row.symbol)
            if existing is not None:
                tier = min(tier, existing.liquidity_tier)
                groups = tuple(dict.fromkeys(existing.groups + (group,)))
                universe.assets[row.symbol] = Asset(
                    symbol=row.symbol, asset_class=AssetClass.EQUITY, name=existing.name, exchange="US",
                    groups=groups, sector=existing.sector or row.sector, liquidity_tier=tier,
                )
                continue
            universe.add(Asset(
                symbol=row.symbol, asset_class=AssetClass.EQUITY, name=row.name, exchange="US",
                groups=(group,), sector=row.sector, liquidity_tier=tier,
            ))


def build_crypto(universe: AssetUniverse, n: int) -> None:
    coins: list[dict] = []
    page = 1
    while len(coins) < n * 1.4 and page <= 4:
        resp = _get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
        )
        batch = resp.json()
        if not batch:
            break
        coins.extend(batch)
        page += 1
        time.sleep(3)
    added = 0
    seen: set[str] = set()
    for c in coins:
        sym = str(c.get("symbol", "")).upper()
        name = str(c.get("name", ""))
        if not sym.isalnum() or len(sym) > 10 or re.search(r"\d{3,}", sym) or sym in STABLECOINS or sym in seen:
            continue
        lname = name.lower()
        if any(w in lname for w in DERIVATIVE_WORDS) or sym.startswith(("W", "ST", "CB")) and sym[1:] in seen:
            continue
        price = c.get("current_price") or 0
        if ("usd" in lname or "dollar" in lname or "euro" in lname) and 0.97 <= price <= 1.03:
            continue
        hi, lo = c.get("high_24h") or 0, c.get("low_24h") or 0
        if lo > 0 and hi / lo < 1.004:  # pegged asset (fiat / gold / yield-bearing stable), not a volatile coin
            continue
        seen.add(sym)
        rank = c.get("market_cap_rank") or (added + 1)
        tier = 1 if added < 2 else 2 if added < 20 else 3 if added < 100 else 4 if added < 200 else 5
        universe.add(Asset(
            symbol=f"{sym}-USD", asset_class=AssetClass.CRYPTO, name=name, exchange="CRYPTO",
            groups=("crypto_top%d" % n,), liquidity_tier=tier, market_cap_rank=int(rank), shortable=False,
        ))
        added += 1
        if added >= n:
            break
    print(f"  crypto: {added} assets")


def build_static(universe: AssetUniverse) -> None:
    for sym, name, tier in ETFS:
        universe.add(Asset(symbol=sym, asset_class=AssetClass.ETF, name=name, exchange="US", groups=("etf",), liquidity_tier=tier))
    for sym, name, tier in FOREX:
        universe.add(Asset(symbol=sym, asset_class=AssetClass.FOREX, name=name, exchange="FX", groups=("fx",),
                           liquidity_tier=tier, quote_currency=name.split("/")[1]))
    for sym, name in INDICES:
        universe.add(Asset(symbol=sym, asset_class=AssetClass.INDEX, name=name, exchange="INDEX", groups=("index",),
                           liquidity_tier=1, tradable=False, shortable=False))
    print(f"  etf: {len(ETFS)}, fx: {len(FOREX)}, indices: {len(INDICES)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tradesim/assets/data/universe.csv"))
    parser.add_argument("--crypto", type=int, default=300, help="number of crypto assets (by market cap)")
    args = parser.parse_args()

    universe = AssetUniverse()
    print("Fetching equity index constituents from Wikipedia...")
    build_equities(universe)
    print("Adding ETFs / FX / indices...")
    build_static(universe)
    print("Fetching crypto market caps from CoinGecko...")
    build_crypto(universe, args.crypto)
    universe.to_csv(args.out)
    print(f"Wrote {len(universe)} assets to {args.out}: {universe.counts()}")


if __name__ == "__main__":
    main()
