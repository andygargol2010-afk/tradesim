"""Asset universe: metadata and trading rules for every tradable symbol.

Canonical symbol conventions (provider adapters translate from these):

* US equities / ETFs: plain ticker, ``AAPL``, ``SPY``, ``BRK-B`` (Yahoo style).
* Crypto: ``<BASE>-USD``, e.g. ``BTC-USD``. Priced in USD; the ccxt adapter
  maps to ``BTC/USDT`` / ``BTC/USD`` depending on the exchange.
* Forex: ``EURUSD=X`` (Yahoo style), quoted in the second currency.
* Indices: ``^GSPC`` - data only, not tradable.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Iterator

from tradesim.types import AssetClass, D

# Trading sessions. "XNYS" = NYSE/Nasdaq regular hours (09:30-16:00 America/New_York, Mon-Fri),
# "24/7" = crypto, "24/5" = forex (Sun 17:00 - Fri 17:00 New York).
SESSION_BY_CLASS: dict[AssetClass, str] = {
    AssetClass.EQUITY: "XNYS",
    AssetClass.ETF: "XNYS",
    AssetClass.CRYPTO: "24/7",
    AssetClass.FOREX: "24/5",
    AssetClass.INDEX: "XNYS",
}

# Smallest tradable increment. Alpaca supports fractional equities to 9 decimals,
# most crypto venues to 1e-8 of the base asset.
QTY_STEP_BY_CLASS: dict[AssetClass, Decimal] = {
    AssetClass.EQUITY: Decimal("0.000000001"),
    AssetClass.ETF: Decimal("0.000000001"),
    AssetClass.CRYPTO: Decimal("0.00000001"),
    AssetClass.FOREX: Decimal("0.01"),
    AssetClass.INDEX: Decimal("0.000000001"),
}

MIN_NOTIONAL_BY_CLASS: dict[AssetClass, Decimal] = {
    AssetClass.EQUITY: Decimal("1"),  # Alpaca: $1 minimum for fractional orders
    AssetClass.ETF: Decimal("1"),
    AssetClass.CRYPTO: Decimal("1"),
    AssetClass.FOREX: Decimal("1"),
    AssetClass.INDEX: Decimal("1"),
}


@dataclass(frozen=True)
class Asset:
    """Static description of a tradable instrument and its exchange rules."""

    symbol: str
    asset_class: AssetClass
    name: str = ""
    exchange: str = ""
    quote_currency: str = "USD"
    groups: tuple[str, ...] = ()  # e.g. ("sp500", "nasdaq100") or ("crypto_top300",)
    sector: str = ""
    liquidity_tier: int = 3  # 1 = mega-liquid (SPY, AAPL, BTC) ... 5 = illiquid micro-cap / long-tail token
    market_cap_rank: int | None = None
    tradable: bool = True
    fractionable: bool = True
    shortable: bool = True
    marginable: bool = True
    tick_size: Decimal | None = None  # None -> price-dependent default (see ``tick_for_price``)
    qty_step: Decimal | None = None
    min_qty: Decimal | None = None
    min_notional: Decimal | None = None
    session: str | None = None

    # ------------------------------------------------------------------ rules
    @property
    def effective_qty_step(self) -> Decimal:
        if self.qty_step is not None:
            return self.qty_step
        if not self.fractionable:
            return Decimal(1)
        return QTY_STEP_BY_CLASS[self.asset_class]

    @property
    def effective_min_qty(self) -> Decimal:
        return self.min_qty if self.min_qty is not None else self.effective_qty_step

    @property
    def effective_min_notional(self) -> Decimal:
        return self.min_notional if self.min_notional is not None else MIN_NOTIONAL_BY_CLASS[self.asset_class]

    @property
    def effective_session(self) -> str:
        return self.session or SESSION_BY_CLASS[self.asset_class]

    @property
    def is_24_7(self) -> bool:
        return self.effective_session == "24/7"

    def tick_for_price(self, price: float) -> Decimal:
        """Minimum price increment at ``price``.

        US equities follow Reg NMS Rule 612 ($0.01 at or above $1, $0.0001 below).
        Crypto / FX default to ~6 significant digits, similar to major venues.
        """
        if self.tick_size is not None:
            return self.tick_size
        if self.asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.INDEX):
            return Decimal("0.01") if price >= 1.0 else Decimal("0.0001")
        if self.asset_class is AssetClass.FOREX:
            return Decimal("0.00001") if price < 50 else Decimal("0.001")
        if price <= 0 or not math.isfinite(price):
            return Decimal("0.00000001")
        exponent = max(math.floor(math.log10(price)) - 5, -10)
        return Decimal(1).scaleb(exponent)

    # --------------------------------------------------------------- (de)ser
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asset_class"] = self.asset_class.value
        d["groups"] = list(self.groups)
        for k in ("tick_size", "qty_step", "min_qty", "min_notional"):
            d[k] = None if d[k] is None else str(d[k])
        return d


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "t")


def _parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value.strip() == "":
        return None
    return D(value.strip())


def asset_from_row(row: dict[str, str]) -> Asset:
    """Build an :class:`Asset` from a CSV row (missing columns use defaults)."""
    asset_class = AssetClass(row["asset_class"].strip().lower())
    rank = row.get("market_cap_rank") or ""
    return Asset(
        symbol=row["symbol"].strip(),
        asset_class=asset_class,
        name=(row.get("name") or "").strip(),
        exchange=(row.get("exchange") or "").strip(),
        quote_currency=(row.get("quote_currency") or "USD").strip() or "USD",
        groups=tuple(g for g in (row.get("groups") or "").split("|") if g),
        sector=(row.get("sector") or "").strip(),
        liquidity_tier=int(row.get("liquidity_tier") or 3),
        market_cap_rank=int(rank) if rank.strip() else None,
        tradable=_parse_bool(row.get("tradable"), asset_class is not AssetClass.INDEX),
        fractionable=_parse_bool(row.get("fractionable"), True),
        shortable=_parse_bool(row.get("shortable"), asset_class is not AssetClass.CRYPTO),
        marginable=_parse_bool(row.get("marginable"), True),
        tick_size=_parse_decimal(row.get("tick_size")),
        qty_step=_parse_decimal(row.get("qty_step")),
        min_qty=_parse_decimal(row.get("min_qty")),
        min_notional=_parse_decimal(row.get("min_notional")),
        session=(row.get("session") or "").strip() or None,
    )


CSV_COLUMNS = [
    "symbol", "name", "asset_class", "exchange", "quote_currency", "groups", "sector",
    "liquidity_tier", "market_cap_rank", "tradable", "fractionable", "shortable", "marginable",
    "tick_size", "qty_step", "min_qty", "min_notional", "session",
]


@dataclass
class AssetUniverse:
    """A searchable collection of :class:`Asset` keyed by canonical symbol."""

    assets: dict[str, Asset] = field(default_factory=dict)

    # ------------------------------------------------------------ factories
    @classmethod
    def from_csv(cls, path: str | Path) -> "AssetUniverse":
        with open(path, newline="", encoding="utf-8") as fh:
            return cls.from_rows(csv.DictReader(fh))

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str]]) -> "AssetUniverse":
        universe = cls()
        for row in rows:
            if not row.get("symbol"):
                continue
            universe.add(asset_from_row(row))
        return universe

    @classmethod
    def default(cls) -> "AssetUniverse":
        """Bundled snapshot: S&P 500 + Nasdaq-100 + S&P MidCap 400 + S&P SmallCap 600,
        ~150 ETFs, top ~300 crypto by market cap, FX majors/crosses and main indices.

        Regenerate with ``python scripts/build_universe.py``.
        """
        data = resources.files("tradesim.assets").joinpath("data/universe.csv")
        with data.open("r", encoding="utf-8", newline="") as fh:
            return cls.from_rows(csv.DictReader(fh))

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for asset in self:
                d = asset.to_dict()
                d["groups"] = "|".join(asset.groups)
                writer.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in CSV_COLUMNS})

    # ------------------------------------------------------------- mutation
    def add(self, asset: Asset, overwrite: bool = True) -> None:
        if asset.symbol in self.assets and not overwrite:
            existing = self.assets[asset.symbol]
            merged_groups = tuple(dict.fromkeys(existing.groups + asset.groups))
            self.assets[asset.symbol] = replace(existing, groups=merged_groups)
            return
        self.assets[asset.symbol] = asset

    def merge(self, other: "AssetUniverse") -> "AssetUniverse":
        for asset in other:
            self.add(asset, overwrite=False)
        return self

    # --------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self) -> Iterator[Asset]:
        return iter(self.assets.values())

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.assets

    def get(self, symbol: str) -> Asset:
        try:
            return self.assets[symbol]
        except KeyError:
            raise KeyError(f"Unknown symbol {symbol!r}. Use get_available_assets() to list symbols.") from None

    def symbols(self) -> list[str]:
        return list(self.assets)

    def filter(
        self,
        asset_class: AssetClass | str | Iterable[AssetClass | str] | None = None,
        group: str | None = None,
        tradable_only: bool = True,
        max_liquidity_tier: int | None = None,
        query: str | None = None,
    ) -> list[Asset]:
        classes: set[AssetClass] | None = None
        if asset_class is not None:
            items = [asset_class] if isinstance(asset_class, (str, AssetClass)) else list(asset_class)
            classes = {AssetClass(c) for c in items}
        q = query.lower() if query else None
        out: list[Asset] = []
        for a in self.assets.values():
            if tradable_only and not a.tradable:
                continue
            if classes is not None and a.asset_class not in classes:
                continue
            if group is not None and group not in a.groups:
                continue
            if max_liquidity_tier is not None and a.liquidity_tier > max_liquidity_tier:
                continue
            if q and q not in a.symbol.lower() and q not in a.name.lower():
                continue
            out.append(a)
        return out

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.assets.values():
            out[a.asset_class.value] = out.get(a.asset_class.value, 0) + 1
        return out
