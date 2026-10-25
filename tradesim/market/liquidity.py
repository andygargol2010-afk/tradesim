"""Liquidity, spread and market-impact models.

The execution engine needs three things to price a fill realistically:

1. **Touch prices** (best bid / ask). Observed when the feed provides them
   (live crypto / Alpaca quotes), otherwise modelled as ``mid +/- half_spread``
   where the spread depends on the asset's liquidity tier, recent volatility
   and any scenario multiplier.
2. **Depth** beyond the touch, i.e. how price moves as we consume liquidity.
   Without a real L2 book we use the empirically robust *square-root law*
   (Almgren et al. 2005, Toth et al. 2011): executing ``Q`` units while the
   market trades ``V`` units moves price by ``Y * sigma * sqrt(Q / V)``. The
   book is continuous, so the volume-weighted fill price of the slice
   ``[q0, q1]`` has a closed form. Liquidity consumed by earlier orders in the
   same bar is remembered, so splitting an order inside one bar gains nothing.
3. **Capacity**: at most ``participation_rate * bar_volume`` can be filled per
   bar (you cannot be most of the tape without moving the market far more than
   any model predicts). The rest stays working, or is cancelled for IOC/FOK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradesim.data.base import OrderBookSnapshot
from tradesim.types import AssetClass, Side


@dataclass(frozen=True)
class LiquidityProfile:
    """Typical microstructure of an asset class / liquidity tier.

    Values are deliberately conservative medians (2023-2025, regular session):
    ``base_spread_bps`` is the full quoted spread at normal volatility;
    ``adv_usd`` typical daily dollar volume, used when the feed has no volume;
    ``annual_vol`` is used by the synthetic data generator.
    """

    base_spread_bps: float
    adv_usd: float
    annual_vol: float


LIQUIDITY_PROFILES: dict[AssetClass, dict[int, LiquidityProfile]] = {
    AssetClass.EQUITY: {
        1: LiquidityProfile(1.5, 5e9, 0.28),
        2: LiquidityProfile(4.0, 3e8, 0.30),
        3: LiquidityProfile(10.0, 6e7, 0.36),
        4: LiquidityProfile(25.0, 1.5e7, 0.45),
        5: LiquidityProfile(80.0, 1e6, 0.70),
    },
    AssetClass.ETF: {
        1: LiquidityProfile(1.0, 3e9, 0.17),
        2: LiquidityProfile(4.0, 1e8, 0.20),
        3: LiquidityProfile(12.0, 1e7, 0.25),
        4: LiquidityProfile(30.0, 2e6, 0.30),
        5: LiquidityProfile(60.0, 5e5, 0.35),
    },
    AssetClass.CRYPTO: {
        1: LiquidityProfile(1.0, 1.5e10, 0.55),
        2: LiquidityProfile(5.0, 4e8, 0.80),
        3: LiquidityProfile(15.0, 4e7, 1.00),
        4: LiquidityProfile(40.0, 6e6, 1.20),
        5: LiquidityProfile(100.0, 8e5, 1.50),
    },
    AssetClass.FOREX: {
        1: LiquidityProfile(0.6, 1e11, 0.08),
        2: LiquidityProfile(1.5, 1e10, 0.10),
        3: LiquidityProfile(5.0, 1e9, 0.13),
        4: LiquidityProfile(20.0, 1e8, 0.25),
        5: LiquidityProfile(50.0, 1e7, 0.35),
    },
    AssetClass.INDEX: {t: LiquidityProfile(1.0, 1e10, 0.18) for t in range(1, 6)},
}


def liquidity_profile(asset_class: AssetClass, tier: int) -> LiquidityProfile:
    tiers = LIQUIDITY_PROFILES[asset_class]
    return tiers[min(max(tier, 1), 5)]


@dataclass(frozen=True)
class SpreadModel:
    """Modelled bid/ask spread when the feed has none.

    ``spread_bps = base_spread_bps * (sigma / sigma_typical) ** vol_elasticity * spread_mult``,
    floored at one tick and capped at ``max_spread_bps``.
    Higher volatility -> wider spreads is one of the most stable stylised facts of
    market microstructure; the square-root elasticity is a reasonable middle ground.
    """

    vol_elasticity: float = 0.5
    max_spread_bps: float = 500.0
    spread_multiplier: float = 1.0  # global knob (e.g. 2.0 to stress-test)

    def spread_bps(
        self, profile: LiquidityProfile, sigma_bar: float, sigma_typical_bar: float, scenario_mult: float = 1.0
    ) -> float:
        ratio = 1.0
        if sigma_typical_bar > 0 and math.isfinite(sigma_bar) and sigma_bar > 0:
            ratio = min(max(sigma_bar / sigma_typical_bar, 0.25), 10.0) ** self.vol_elasticity
        bps = profile.base_spread_bps * ratio * scenario_mult * self.spread_multiplier
        return min(bps, self.max_spread_bps)


@dataclass
class SqrtImpactBook:
    """Continuous synthetic order book implied by the square-root impact law.

    Marginal price after consuming ``q`` units on one side:
    ``touch * (1 + Y * sigma * sqrt(q / V))`` for asks (``1 - ...`` for bids),
    where ``V`` is the typical volume per bar and ``sigma`` the per-bar volatility.
    """

    bid: float
    ask: float
    sigma_bar: float
    volume_bar: float
    impact_coefficient: float = 1.0

    def _k(self) -> float:
        if self.volume_bar <= 0 or not math.isfinite(self.volume_bar):
            return 0.0
        return self.impact_coefficient * max(self.sigma_bar, 0.0) / math.sqrt(self.volume_bar)

    def avg_price(self, side: Side, q0: float, q1: float) -> float:
        """Volume-weighted price for consuming liquidity from depth ``q0`` to ``q1`` (> q0)."""
        touch = self.ask if side is Side.BUY else self.bid
        if q1 <= q0:
            return touch
        k = self._k()
        # integral of sqrt(q) dq from q0 to q1 = 2/3 (q1^1.5 - q0^1.5)
        mean_sqrt = (2.0 / 3.0) * (q1 ** 1.5 - q0 ** 1.5) / (q1 - q0)
        move = k * mean_sqrt
        return touch * (1.0 + move) if side is Side.BUY else touch * max(1.0 - move, 1e-9)

    def marginal_price(self, side: Side, q: float) -> float:
        touch = self.ask if side is Side.BUY else self.bid
        move = self._k() * math.sqrt(max(q, 0.0))
        return touch * (1.0 + move) if side is Side.BUY else touch * max(1.0 - move, 1e-9)

    def max_qty_within(self, side: Side, limit_price: float, q0: float) -> float:
        """Largest depth ``q1 >= q0`` whose *marginal* price is still within ``limit_price``.

        Used for marketable limit orders: they sweep the book only up to their limit.
        """
        touch = self.ask if side is Side.BUY else self.bid
        k = self._k()
        if side is Side.BUY:
            if limit_price < touch:
                return q0
            if k == 0:
                return math.inf
            move = limit_price / touch - 1.0
        else:
            if limit_price > touch:
                return q0
            if k == 0:
                return math.inf
            move = 1.0 - limit_price / touch
        q = (move / k) ** 2
        return max(q, q0)


@dataclass
class L2Book:
    """A real (or replayed) discrete level-2 book, swept level by level."""

    snapshot: OrderBookSnapshot

    @property
    def bid(self) -> float:
        return self.snapshot.bids[0][0] if self.snapshot.bids else float("nan")

    @property
    def ask(self) -> float:
        return self.snapshot.asks[0][0] if self.snapshot.asks else float("nan")

    def _levels(self, side: Side) -> list[tuple[float, float]]:
        return self.snapshot.asks if side is Side.BUY else self.snapshot.bids

    def depth(self, side: Side) -> float:
        return sum(size for _, size in self._levels(side))

    def avg_price(self, side: Side, q0: float, q1: float) -> float:
        remaining_skip, remaining = q0, q1 - q0
        cost = filled = 0.0
        for price, size in self._levels(side):
            if remaining_skip >= size:
                remaining_skip -= size
                continue
            avail = size - remaining_skip
            remaining_skip = 0.0
            take = min(avail, remaining)
            cost += take * price
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled == 0:
            return self.ask if side is Side.BUY else self.bid
        return cost / filled

    def max_qty_within(self, side: Side, limit_price: float, q0: float) -> float:
        total = 0.0
        for price, size in self._levels(side):
            if (side is Side.BUY and price > limit_price) or (side is Side.SELL and price < limit_price):
                break
            total += size
        return max(total, q0)
