"""Commission and regulatory-fee models with presets for common brokers / exchanges.

Fee levels change; the presets reflect published *entry-tier* retail schedules
as of 2025 and are deliberately on the conservative side. Verify against your
broker before relying on them, and override any field via ``FeeSchedule``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal

from tradesim.types import AssetClass, Liquidity, Side, ZERO, D

# US regulatory fees on equity *sales*. Rates are reset by the SEC/FINRA (often yearly):
# SEC Section 31: $27.80 per $1M of sale proceeds (FY2024-25 rate);
# FINRA Trading Activity Fee: $0.000166 per share sold, capped at $8.30 per trade.
SEC_FEE_RATE = Decimal("0.0000278")
FINRA_TAF_PER_SHARE = Decimal("0.000166")
FINRA_TAF_MAX = Decimal("8.30")


@dataclass(frozen=True)
class FeeSchedule:
    """Fee for one fill.

    ``fee = notional * pct(maker|taker) + quantity * per_share + fixed (first fill only)``,
    then raised to ``min_per_order`` (counting fees already paid on the order) and
    capped at ``max_pct_of_notional``. US regulatory fees are added on sells
    when ``us_regulatory_fees`` is set.
    """

    name: str = "custom"
    maker_pct: Decimal = ZERO  # fraction, e.g. 0.001 = 0.1% = 10 bps
    taker_pct: Decimal = ZERO
    fixed_per_order: Decimal = ZERO
    per_share: Decimal = ZERO
    min_per_order: Decimal = ZERO
    max_pct_of_notional: Decimal | None = None
    us_regulatory_fees: bool = False

    def compute(
        self,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        liquidity: Liquidity,
        fees_already_paid_on_order: Decimal = ZERO,
        first_fill: bool = True,
    ) -> Decimal:
        notional = quantity * price
        pct = self.maker_pct if liquidity is Liquidity.MAKER else self.taker_pct
        fee = notional * pct + quantity * self.per_share
        if first_fill:
            fee += self.fixed_per_order
        if self.min_per_order > 0:
            fee = max(fee, self.min_per_order - fees_already_paid_on_order)
        if self.max_pct_of_notional is not None:
            fee = min(fee, notional * self.max_pct_of_notional)
        if self.us_regulatory_fees and side is Side.SELL:
            fee += notional * SEC_FEE_RATE
            fee += min(quantity * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX)
        fee = max(fee, ZERO).quantize(Decimal("0.00000001"))
        return fee if fee != 0 else ZERO

    @classmethod
    def from_dict(cls, d: dict) -> "FeeSchedule":
        kw = {k: (D(v) if k not in ("name", "us_regulatory_fees") and v is not None else v) for k, v in d.items()}
        return cls(**kw)


# ----------------------------------------------------------------------------- presets
ZERO_FEES = FeeSchedule("zero")

SCHEDULES: dict[str, FeeSchedule] = {
    "zero": ZERO_FEES,
    # Alpaca: commission-free US equities (regulatory fees passed through); crypto tier 1 (< $100k 30d volume).
    "alpaca_equity": FeeSchedule("alpaca_equity", us_regulatory_fees=True),
    "alpaca_crypto": FeeSchedule("alpaca_crypto", maker_pct=D("0.0015"), taker_pct=D("0.0025")),
    # Interactive Brokers Pro, US stocks.
    "ibkr_fixed": FeeSchedule("ibkr_fixed", per_share=D("0.005"), min_per_order=D("1.00"),
                              max_pct_of_notional=D("0.01"), us_regulatory_fees=True),
    # Tiered (<= 300k shares/month) + approx. exchange & clearing fees (~$0.0005/share, removing liquidity).
    "ibkr_tiered": FeeSchedule("ibkr_tiered", per_share=D("0.0040"), min_per_order=D("0.35"),
                               max_pct_of_notional=D("0.01"), us_regulatory_fees=True),
    "ibkr_crypto": FeeSchedule("ibkr_crypto", maker_pct=D("0.0018"), taker_pct=D("0.0018"), min_per_order=D("1.75")),
    "ibkr_forex": FeeSchedule("ibkr_forex", maker_pct=D("0.00002"), taker_pct=D("0.00002"), min_per_order=D("2.00")),
    # Crypto exchanges, entry tier, no discounts.
    "binance_spot": FeeSchedule("binance_spot", maker_pct=D("0.001"), taker_pct=D("0.001")),
    "coinbase_advanced": FeeSchedule("coinbase_advanced", maker_pct=D("0.006"), taker_pct=D("0.012")),
    "kraken_pro": FeeSchedule("kraken_pro", maker_pct=D("0.0025"), taker_pct=D("0.004")),
    # Generic retail FX (cost mostly in the spread) and "percentage + fixed" example.
    "retail_forex": FeeSchedule("retail_forex"),
    "pct_plus_fixed": FeeSchedule("pct_plus_fixed", maker_pct=D("0.001"), taker_pct=D("0.001"), fixed_per_order=D("1")),
}


@dataclass
class FeeModel:
    """Maps each asset class to a :class:`FeeSchedule` (a broker's full fee sheet)."""

    name: str = "custom"
    by_class: dict[AssetClass, FeeSchedule] = field(default_factory=dict)
    default: FeeSchedule = ZERO_FEES

    def schedule_for(self, asset_class: AssetClass) -> FeeSchedule:
        return self.by_class.get(asset_class, self.default)

    def with_override(self, asset_class: AssetClass, schedule: FeeSchedule) -> "FeeModel":
        by_class = dict(self.by_class)
        by_class[asset_class] = schedule
        return replace(self, by_class=by_class)

    @classmethod
    def preset(cls, name: str) -> "FeeModel":
        s = SCHEDULES
        eq = (AssetClass.EQUITY, AssetClass.ETF)
        presets = {
            "zero": cls("zero"),
            "alpaca": cls("alpaca", {**{c: s["alpaca_equity"] for c in eq}, AssetClass.CRYPTO: s["alpaca_crypto"]},
                          default=s["retail_forex"]),
            "ibkr_fixed": cls("ibkr_fixed", {**{c: s["ibkr_fixed"] for c in eq}, AssetClass.CRYPTO: s["ibkr_crypto"],
                                             AssetClass.FOREX: s["ibkr_forex"]}),
            "ibkr_tiered": cls("ibkr_tiered", {**{c: s["ibkr_tiered"] for c in eq}, AssetClass.CRYPTO: s["ibkr_crypto"],
                                               AssetClass.FOREX: s["ibkr_forex"]}),
            # Exchanges that only list crypto: equities fall back to Alpaca-like pricing.
            "binance": cls("binance", {**{c: s["alpaca_equity"] for c in eq}, AssetClass.CRYPTO: s["binance_spot"]}),
            "coinbase": cls("coinbase", {**{c: s["alpaca_equity"] for c in eq}, AssetClass.CRYPTO: s["coinbase_advanced"]}),
            "kraken": cls("kraken", {**{c: s["alpaca_equity"] for c in eq}, AssetClass.CRYPTO: s["kraken_pro"]}),
            "pct_plus_fixed": cls("pct_plus_fixed", default=s["pct_plus_fixed"]),
        }
        if name not in presets:
            raise KeyError(f"Unknown fee preset {name!r}; choose from {sorted(presets)}")
        return presets[name]

    @staticmethod
    def presets() -> list[str]:
        return ["zero", "alpaca", "ibkr_fixed", "ibkr_tiered", "binance", "coinbase", "kraken", "pct_plus_fixed"]
