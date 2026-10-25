"""Deterministic synthetic market data with realistic stylised facts.

Used for offline development, unit tests, CI, and as a source of *unlimited
novel episodes* so an agent cannot memorise the single realised history.
It is **not** a substitute for real data when validating a strategy.

Daily process (per asset class a common market factor drives correlation):

* GARCH(1,1) volatility clustering, Student-t(4) fat-tailed shocks,
* Poisson jumps, beta to a shared asset-class factor,
* overnight gaps for exchange-traded assets (~25% of daily variance overnight),
  continuous 24/7 trading for crypto,
* volume proportional to the tier's typical dollar volume, rising with |return|.

Intraday bars are a Brownian bridge between each day's open and close with a
U-shaped volatility / volume profile for equities, so **every timeframe
agrees on each day's open and close**, and paths are prefix-stable: the same
``(seed, symbol, date)`` always yields the same prices regardless of the
requested range.
"""

from __future__ import annotations

import math
import zlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider
from tradesim.data.timeframes import Timeframe, normalize_bars, resample_bars
from tradesim.market.liquidity import liquidity_profile
from tradesim.types import AssetClass

ANCHOR = datetime(2015, 1, 1, tzinfo=timezone.utc)
NY = "America/New_York"

_KNOWN_START_PRICES = {
    "BTC-USD": 300.0, "ETH-USD": 1.0, "SPY": 205.0, "QQQ": 102.0, "AAPL": 27.0, "MSFT": 46.0,
    "EURUSD=X": 1.21, "USDJPY=X": 120.0, "GBPUSD=X": 1.56, "^GSPC": 2058.0, "^VIX": 19.0,
}


_FACTOR_GROUP = {
    AssetClass.EQUITY: "us", AssetClass.ETF: "us", AssetClass.INDEX: "us",
    AssetClass.CRYPTO: "crypto", AssetClass.FOREX: "fx",
}


def _symbol_key(symbol: str) -> int:
    return zlib.crc32(symbol.encode("utf-8"))


class SyntheticProvider(DataProvider):
    """Synthetic OHLCV for any :class:`Asset`.

    Args:
        seed: master seed; different seeds give independent market histories.
        drift_annual: optional override of the annual drift for every asset.
        vol_scale: multiplies every asset's volatility (quick stress knob).
    """

    name = "synthetic"

    def __init__(self, seed: int = 0, drift_annual: float | None = None, vol_scale: float = 1.0):
        self.seed = int(seed)
        self.drift_annual = drift_annual
        self.vol_scale = float(vol_scale)
        self._daily_cache: dict[tuple[str, int], pd.DataFrame] = {}

    def supports(self, asset: Asset) -> bool:
        return True

    # ------------------------------------------------------------------ calendars
    @staticmethod
    def _days(asset: Asset, end: datetime) -> pd.DatetimeIndex:
        end_day = pd.Timestamp(end).tz_convert("UTC").normalize()
        if asset.is_24_7:
            return pd.date_range(ANCHOR, end_day, freq="D", tz="UTC")
        if asset.effective_session == "24/5":
            return pd.bdate_range(ANCHOR, end_day, tz="UTC")
        return pd.bdate_range(ANCHOR, end_day, tz="UTC")  # weekdays; exchange holidays not modelled

    @lru_cache(maxsize=16)
    def _factor(self, asset_class: str, n: int) -> np.ndarray:
        """Shared standardised daily factor shocks for an asset class (prefix-stable)."""
        rng = np.random.default_rng([self.seed, _symbol_key("factor:" + asset_class)])
        return rng.standard_t(5, size=n) / math.sqrt(5 / 3)

    def _params(self, asset: Asset) -> dict[str, float]:
        prof = liquidity_profile(asset.asset_class, asset.liquidity_tier)
        rng = np.random.default_rng([self.seed, _symbol_key(asset.symbol), 0])
        vol = prof.annual_vol * float(np.exp(rng.normal(0, 0.2))) * self.vol_scale
        if asset.asset_class is AssetClass.CRYPTO:
            drift, beta, rho, jumps, start = 0.10, rng.uniform(0.7, 1.3), 0.7, 10.0, float(np.exp(rng.normal(0.0, 2.5)))
        elif asset.asset_class is AssetClass.FOREX:
            drift, beta, rho, jumps, start = 0.0, rng.uniform(-1, 1), 0.3, 1.0, float(np.exp(rng.normal(0.0, 1.0)))
        elif asset.asset_class is AssetClass.ETF:
            drift, beta, rho, jumps, start = 0.07, rng.uniform(0.8, 1.15), 0.85, 1.5, float(np.exp(rng.normal(math.log(80), 0.6)))
        else:
            drift, beta, rho, jumps, start = 0.08, rng.uniform(0.7, 1.3), 0.65, 4.0, float(np.exp(rng.normal(math.log(60), 0.9)))
        if self.drift_annual is not None:
            drift = self.drift_annual
        start = _KNOWN_START_PRICES.get(asset.symbol, min(max(start, 1e-5), 5e4))
        return {
            "vol": vol, "drift": drift, "beta": float(beta), "rho": rho, "jumps": jumps, "start": start,
            "adv_usd": prof.adv_usd, "overnight_share": 0.0 if asset.is_24_7 else 0.25,
        }

    # ---------------------------------------------------------------- daily path
    def _daily(self, asset: Asset, end: datetime) -> pd.DataFrame:
        days = self._days(asset, end)
        key = (asset.symbol, len(days))
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        n = len(days)
        p = self._params(asset)
        periods = 365.0 if asset.is_24_7 else 252.0
        sig_long = p["vol"] / math.sqrt(periods)
        mu = p["drift"] / periods  # median (log) growth rate, so high-vol assets don't decay to zero
        # One independent stream per component => prefix-stable paths (the value for a given
        # day never depends on how far into the future the series was generated).
        streams = np.random.SeedSequence([self.seed, _symbol_key(asset.symbol), 1]).spawn(7)
        g_eps, g_on, g_jump, g_jsize, g_vol, g_bridge, g_level = (np.random.default_rng(s) for s in streams)
        eps = g_eps.standard_t(6, size=n) / math.sqrt(1.5)  # unit-variance t(6)
        on_noise = g_on.standard_normal(n)
        jump_size = (g_jump.random(n) < p["jumps"] / periods) * g_jsize.normal(0.0, 3.0 * sig_long, size=n)
        vol_noise = g_vol.lognormal(0.0, 0.35, size=n)
        bridge_shocks = g_bridge.standard_normal((n, 48))
        volume_level = float(np.exp(g_level.normal(0, 0.5)))
        factor = self._factor(_FACTOR_GROUP[asset.asset_class], 20000)[:n]

        # GARCH(1,1): var_t = w + a*(r_{t-1}-mu)^2 + b*var_{t-1}, long-run var = sig_long^2.
        # With t(6) shocks the 4th moment is finite (b^2 + 2ab + 6a^2 < 1) => realistic, stable kurtosis.
        a, b = 0.06, 0.92
        w = sig_long ** 2 * (1 - a - b)
        var = sig_long ** 2
        rets = np.empty(n)
        sig = np.empty(n)
        loading = float(np.clip(p["rho"] * p["beta"], -0.95, 0.95))
        idio = math.sqrt(1 - loading ** 2)
        for t in range(n):
            s = math.sqrt(var)
            r = mu + s * (loading * factor[t] + idio * eps[t]) + jump_size[t]
            rets[t] = r
            sig[t] = s
            var = w + a * (r - mu) ** 2 + b * var

        log_close = math.log(p["start"]) + np.cumsum(rets)
        prev_close = np.concatenate([[math.log(p["start"])], log_close[:-1]])
        share = p["overnight_share"]
        r_on = share * rets + math.sqrt(share * (1 - share)) * sig * on_noise if share > 0 else np.zeros(n)
        log_open = prev_close + r_on
        o, c = np.exp(log_open), np.exp(log_close)

        # Day range from a 48-step Brownian bridge open -> close, so daily H/L have the right distribution.
        incr = bridge_shocks * (sig * math.sqrt(1 - share) / math.sqrt(48))[:, None]
        walk = np.concatenate([np.zeros((n, 1)), np.cumsum(incr, axis=1)], axis=1)
        frac = np.linspace(0.0, 1.0, 49)[None, :]
        path = log_open[:, None] + walk - frac * walk[:, -1:] + frac * (log_close - log_open)[:, None]
        hi, lo = np.exp(path.max(axis=1)), np.exp(path.min(axis=1))

        adv_units = p["adv_usd"] / np.maximum(c, 1e-12)
        activity = 1.0 + 2.0 * np.abs(rets - mu) / np.maximum(sig, 1e-12) / 3.0
        volume = adv_units * vol_noise * activity * volume_level
        df = pd.DataFrame(
            {"open": o, "high": np.maximum(hi, np.maximum(o, c)), "low": np.minimum(lo, np.minimum(o, c)),
             "close": c, "volume": volume},
            index=pd.DatetimeIndex(days, name="timestamp"),
        )
        self._daily_cache[key] = df
        return df

    def _bridge(self, asset: Asset, day: pd.Timestamp, x0: float, x1: float, sigma: float, steps: int,
                weights: np.ndarray | None = None) -> np.ndarray:
        """Brownian bridge in log-price from ``x0`` to ``x1`` with total std ``sigma``."""
        rng = np.random.default_rng([self.seed, _symbol_key(asset.symbol), 2, int(day.value // 86_400_000_000_000), steps])
        w = np.full(steps, 1.0 / steps) if weights is None else weights / weights.sum()
        incr = rng.standard_normal(steps) * np.sqrt(w) * sigma
        walk = np.concatenate([[0.0], np.cumsum(incr)])
        frac = np.concatenate([[0.0], np.cumsum(w)])
        return x0 + walk - frac * walk[-1] + frac * (x1 - x0)

    # ------------------------------------------------------------------ intraday
    def _session_bar_starts(self, asset: Asset, day: pd.Timestamp, tf: Timeframe) -> pd.DatetimeIndex:
        if asset.is_24_7 or asset.effective_session == "24/5":
            return pd.date_range(day, day + timedelta(days=1), freq=tf.pandas_freq, inclusive="left")
        local = pd.Timestamp(day.date()).tz_localize(NY)
        start = local + timedelta(hours=9, minutes=30)
        end = local + timedelta(hours=16)
        return pd.date_range(start, end, freq=tf.pandas_freq, inclusive="left").tz_convert("UTC")

    def _intraday(self, asset: Asset, daily: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        frames = []
        sub = 4  # sub-steps per bar to form realistic highs / lows
        for day, row in daily.iterrows():
            starts = self._session_bar_starts(asset, day, tf)
            nb = len(starts)
            if nb == 0:
                continue
            x = np.linspace(0, 1, nb * sub, endpoint=False) + 0.5 / (nb * sub)
            # U-shaped intraday activity for exchange-traded assets, flat for 24/7 markets.
            u = 1.0 + 10.0 * (x - 0.5) ** 2 if not asset.is_24_7 else np.ones_like(x)
            day_sigma = abs(math.log(row["high"] / row["low"])) / 1.6 if row["high"] > row["low"] else 1e-4
            path = self._bridge(asset, day, math.log(row["open"]), math.log(row["close"]), day_sigma, nb * sub, u)
            seg = path[: nb * sub + 1]
            idx = np.arange(nb) * sub
            o = np.exp(seg[idx])
            c = np.exp(seg[idx + sub])
            windows = np.lib.stride_tricks.sliding_window_view(seg, sub + 1)[idx]
            h, l = np.exp(windows.max(axis=1)), np.exp(windows.min(axis=1))
            vw = u.reshape(nb, sub).sum(axis=1)
            rng = np.random.default_rng([self.seed, _symbol_key(asset.symbol), 3, int(day.value // 86_400_000_000_000), nb])
            vol = row["volume"] * vw / vw.sum() * rng.lognormal(0, 0.3, nb)
            frames.append(pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": vol}, index=starts))
        if not frames:
            return normalize_bars(pd.DataFrame())
        return pd.concat(frames)

    # --------------------------------------------------------------------- API
    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        if start_ts < pd.Timestamp(ANCHOR):
            raise ValueError(f"Synthetic data starts at {ANCHOR.date()}")
        daily = self._daily(asset, end_ts.to_pydatetime())
        if tf is Timeframe.D1:
            out = daily
        elif tf is Timeframe.W1:
            out = resample_bars(daily, tf)
        else:
            day_slice = daily[(daily.index >= start_ts.normalize() - timedelta(days=1)) & (daily.index < end_ts)]
            base = self._intraday(asset, day_slice, Timeframe.M15 if tf is Timeframe.H4 else tf)
            out = resample_bars(base, tf) if tf is Timeframe.H4 else base
        out = out[(out.index >= start_ts) & (out.index < end_ts)]
        return normalize_bars(out)
