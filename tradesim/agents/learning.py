"""Agents that learn over time and carry their learned state between runs.

Both learn *inside* ``act`` from information available at decision time only, so they behave
identically in backtests (``evaluate_agent``), in daily paper trading and with a real broker.
Their learned state is plain JSON (``state_dict`` / ``load_state_dict``): copying the agent class
plus that file to another environment reproduces the agent exactly as it is now.

* :class:`OnlineLinearAgent` learns from data: recursive least squares with exponential forgetting
  predicts each asset's next-bar return in units of its own volatility from scale-free price
  features and technical-pattern features (breakouts, trend quality, candlesticks), and allocates
  to the best predictions. It warm-starts on history the first time it runs.
* :class:`WalkForwardRSIAgent` learns by simulation: every ``retune_days`` it backtests a grid of RSI
  settings over the recent past in the realistic simulator (costs, slippage, liquidity) and switches
  only when the best setting beats the current one by a margin (to avoid chasing noise).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tradesim.agents.base import Agent, ClosePosition, Observation, OrderIntent
from tradesim.agents.examples import SimpleRSIAgent
from tradesim.data.base import DataUnavailableError
from tradesim.indicators import rsi_last
from tradesim.patterns import PATTERN_FEATURES, pattern_features

log = logging.getLogger(__name__)

OWN_FEATURES = ["bias", "ret_1", "ret_5", "ret_20", "ret_60", "vol_ratio", "rsi_14", "dist_sma20", "dist_sma50",
                *PATTERN_FEATURES, "volume_z", "volume_trend"]
# market context (SPY; BTC-USD for crypto), strength vs the market and vs its sector / asset class, interactions
CONTEXT_FEATURES = ["mkt_ret_5", "mkt_ret_20", "mkt_vol_regime", "rel_mkt_20", "rel_group_20",
                    "ret_5_x_regime", "ret_20_x_mkt"]
FEATURES = OWN_FEATURES + CONTEXT_FEATURES
MIN_BARS = 61
_I_RET5, _I_RET20 = OWN_FEATURES.index("ret_5"), OWN_FEATURES.index("ret_20")


def price_features(closes: np.ndarray, opens: np.ndarray | None = None, highs: np.ndarray | None = None,
                   lows: np.ndarray | None = None, volumes: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """Scale-free features of one asset from daily bars (oldest first, last = most recent completed bar).

    Price features come from the closes; the technical-pattern features (:data:`PATTERN_FEATURES`) also use
    opens/highs/lows, and the volume features (today's volume z-score, 5/20-day volume trend) the volumes.

    Returns ``(x, sigma)`` over :data:`OWN_FEATURES` with ``sigma`` the 20-bar volatility of log returns, or
    ``None`` if there is not enough history. Every return-like feature is divided by ``sigma * sqrt(bars)`` so that
    one model can be shared by assets as different as an ETF and a small-cap token.
    """
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_BARS or not np.all(np.isfinite(c[-MIN_BARS:])) or np.any(c[-MIN_BARS:] <= 0):
        return None
    lr = np.diff(np.log(c[-MIN_BARS:]))
    sigma = float(np.std(lr[-20:], ddof=1))
    if not sigma > 0:
        return None
    x = np.array([
        1.0,
        lr[-1] / sigma,
        lr[-5:].sum() / (sigma * math.sqrt(5)),
        lr[-20:].sum() / (sigma * math.sqrt(20)),
        lr[-60:].sum() / (sigma * math.sqrt(60)),
        float(np.std(lr[-5:], ddof=1)) / sigma - 1.0,
        (rsi_last(c[-43:], 14) / 100.0 - 0.5) * 4.0,
        (c[-1] / c[-20:].mean() - 1.0) / (sigma * math.sqrt(20)),
        (c[-1] / c[-50:].mean() - 1.0) / (sigma * math.sqrt(50)),
    ])
    ohlc = [None if a is None else np.asarray(a, dtype=float)[-MIN_BARS:] for a in (opens, highs, lows)]
    vol = np.zeros(2)
    if volumes is not None:
        v = np.asarray(volumes, dtype=float)[-21:]
        if len(v) == 21 and np.all(np.isfinite(v)) and np.all(v > 0):
            lv = np.log(v)
            sd = float(np.std(lv[:-1], ddof=1))
            vol = np.array([(lv[-1] - lv[:-1].mean()) / sd if sd > 0 else 0.0, math.log(v[-5:].mean() / v[-20:].mean())])
    x = np.r_[x, pattern_features(c[-MIN_BARS:], *ohlc), vol]
    return np.clip(x, -5.0, 5.0), sigma


def market_features(closes: np.ndarray) -> np.ndarray | None:
    """``[ret_5, ret_20 (volatility units), volatility regime (20 vs 60 bars), raw 20-bar log return]`` of the
    market reference, or ``None`` without enough history."""
    c = np.asarray(closes, dtype=float)[-MIN_BARS:]
    if len(c) < MIN_BARS or not np.all(np.isfinite(c)) or np.any(c <= 0):
        return None
    lr = np.diff(np.log(c))
    s20, s60 = float(np.std(lr[-20:], ddof=1)), float(np.std(lr, ddof=1))
    if not (s20 > 0 and s60 > 0):
        return None
    return np.array([lr[-5:].sum() / (s20 * math.sqrt(5)), lr[-20:].sum() / (s20 * math.sqrt(20)), s20 / s60 - 1.0,
                     lr[-20:].sum()])


def context_features(own: np.ndarray, sigma: float, mkt: np.ndarray | None, group_ret20: float | None) -> np.ndarray:
    """Full feature vector: the asset's own features plus market context, relative strength and interactions."""
    m5, m20, regime = (mkt[0], mkt[1], mkt[2]) if mkt is not None else (0.0, 0.0, 0.0)
    rel_mkt = ((own[_I_RET20] * sigma * math.sqrt(20) - mkt[3]) / (sigma * math.sqrt(20))) if mkt is not None else 0.0
    rel_group = own[_I_RET20] - group_ret20 if group_ret20 is not None else 0.0
    x = np.r_[own, m5, m20, regime, rel_mkt, rel_group, own[_I_RET5] * regime,
              own[_I_RET20] * float(np.clip(m20 / 2, -1, 1))]
    return np.clip(x, -5.0, 5.0)


def _frame_features(df: pd.DataFrame | pd.Series, end: int | None = None) -> tuple[np.ndarray, float] | None:
    """:func:`price_features` of a closes Series or an OHLCV frame, using rows up to ``end`` (exclusive)."""
    sl = slice(max(0, (end if end is not None else len(df)) - MIN_BARS), end)
    if isinstance(df, pd.Series):
        return price_features(df.to_numpy(dtype=float)[sl])
    cols = [df[k].to_numpy(dtype=float)[sl] if k in df else None for k in ("close", "open", "high", "low", "volume")]
    return price_features(*cols)


def _closes(df: pd.DataFrame | pd.Series) -> pd.Series:
    return df if isinstance(df, pd.Series) else df["close"]


def _entry_exit(bars: pd.DataFrame | pd.Series, t: int, h: int) -> tuple[float, float, pd.Timestamp] | None:
    """Prices a decision made after bar ``t`` actually trades at: from the next bar's open to the open ``h`` bars
    later (the overnight gap after bar ``t`` is not tradable). Closes only: close of ``t`` to close of ``t + h``."""
    if isinstance(bars, pd.DataFrame) and "open" in bars:
        if t + 1 + h < len(bars):
            o = bars["open"].to_numpy(dtype=float)
            return float(o[t + 1]), float(o[t + 1 + h]), bars.index[t + 1 + h]
        return None
    c = _closes(bars)
    if t + h < len(c):
        return float(c.iloc[t]), float(c.iloc[t + h]), c.index[t + h]
    return None


def default_groups(symbols: Sequence[str], universe: Any = None) -> tuple[dict[str, str], dict[str, str]]:
    """``(group, market)`` per symbol: group = sector for stocks (asset class otherwise), market reference = SPY
    (BTC-USD for crypto)."""
    groups, markets = {}, {}
    for s in symbols:
        asset = None
        if universe is not None:
            try:
                asset = universe.get(s)
            except Exception:  # noqa: BLE001 - unknown symbol
                asset = None
        crypto = (asset.asset_class.value == "crypto") if asset is not None else s.endswith("-USD")
        groups[s] = "crypto" if crypto else (getattr(asset, "sector", "") or (asset.asset_class.value if asset else "equity"))
        markets[s] = "BTC-USD" if crypto else "SPY"
    return groups, markets


def build_panel(frames: dict[str, pd.DataFrame | pd.Series], groups: dict[str, str], markets: dict[str, str],
                horizon: int | None = None, last_only: bool = False, every_days: int = 1,
                cache: dict | None = None) -> list[dict[str, Any]]:
    """Feature vectors for every (symbol, bar) of ``frames`` (or only each symbol's last bar).

    With ``horizon`` each sample also gets its raw target ``y`` (open-to-open log return over ``horizon`` bars in
    volatility units, see :func:`_entry_exit`) and the date it becomes known (``realized``). Market and group context
    only use bars dated on or before the sample's bar: no look-ahead. ``every_days`` > 1 keeps one calendar day in
    that many (the same days for every asset), e.g. 7 for weekly, non-overlapping samples on large universes.
    ``cache`` (a dict kept by the caller) memoises each bar's own features between calls."""
    mkt: dict[str, tuple[pd.DatetimeIndex, list[np.ndarray | None]]] = {}
    for m in set(markets.values()):
        if m in frames:
            c = _closes(frames[m])
            arr = c.to_numpy(dtype=float)
            first = max(MIN_BARS - 1, len(arr) - 5) if last_only else MIN_BARS - 1
            mkt[m] = (c.index[first:], [market_features(arr[: t + 1]) for t in range(first, len(arr))])
    rows = []
    for sym, bars in frames.items():
        idx = _closes(bars).index
        cols = ([bars.to_numpy(dtype=float)] if isinstance(bars, pd.Series) else
                [bars[k].to_numpy(dtype=float) if k in bars else None for k in ("close", "open", "high", "low", "volume")])
        for t in ([len(idx) - 1] if last_only else range(MIN_BARS - 1, len(idx))):
            if every_days > 1 and idx[t].toordinal() % every_days:
                continue
            key = (sym, idx[t])
            f = cache.get(key) if cache is not None else None
            if f is None:
                lo = max(0, t + 1 - MIN_BARS)
                f = price_features(*[None if a is None else a[lo: t + 1] for a in cols])
                if cache is not None and f is not None and t < len(idx) - 1:  # the last bar may still change
                    cache[key] = f
            if f is not None:
                rows.append((sym, t, idx[t], f[0], f[1]))
    by_group: dict[tuple[Any, str], list[float]] = defaultdict(list)
    for sym, _, d, own, _ in rows:
        by_group[(d.date(), groups.get(sym, ""))].append(own[_I_RET20])
    out = []
    for sym, t, d, own, sigma in rows:
        mv = None
        if markets.get(sym) in mkt:
            dates, vecs = mkt[markets[sym]]
            j = int(dates.searchsorted(d, side="right")) - 1
            mv = vecs[j] if j >= 0 else None
        g = by_group[(d.date(), groups.get(sym, ""))]
        s: dict[str, Any] = {"sym": sym, "ts": d.isoformat(), "x": context_features(own, sigma, mv, float(np.mean(g))
                                                                                    if len(g) >= 3 else None),
                             "sigma": sigma}
        if horizon:
            ee = _entry_exit(frames[sym], t, horizon)
            if ee is not None and ee[0] > 0 and ee[1] > 0:
                s["y"] = float(np.clip(math.log(ee[1] / ee[0]) / (sigma * math.sqrt(horizon)), -6, 6))
                s["realized"] = ee[2].isoformat()
        out.append(s)
    return out


def demean_targets(samples: list[dict[str, Any]], min_group: int = 5) -> None:
    """``y_rel`` = target minus the mean of the samples created and realised on the same dates: the model learns
    which assets beat the rest (what it trades and how the IC scores it), not where the whole market goes."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        if s.get("y") is not None:
            groups[(s["ts"][:10], s["realized"][:10])].append(s)
    for g in groups.values():
        mu = float(np.mean([s["y"] for s in g])) if len(g) >= min_group else 0.0
        for s in g:
            s["y_rel"] = float(np.clip(s["y"] - mu, -6, 6))


class RecursiveLeastSquares:
    """Online linear regression with exponential forgetting (``forgetting`` = 1 means no forgetting).

    Each update costs O(d^2); effective memory is about ``1 / (1 - forgetting)`` samples.
    """

    def __init__(self, dim: int, forgetting: float = 0.998, ridge: float = 50.0):
        self.forgetting = forgetting
        self.ridge = ridge
        self.w = np.zeros(dim)
        self.P = np.eye(dim) / ridge
        self.n = 0

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    def update(self, x: np.ndarray, y: float) -> float:
        """Learn from one sample; returns the prediction made *before* the update (out-of-sample)."""
        pred = self.predict(x)
        Px = self.P @ x
        k = Px / (self.forgetting + x @ Px)
        self.w = self.w + k * (y - pred)
        self.P = (self.P - np.outer(k, Px)) / self.forgetting
        self.P = (self.P + self.P.T) / 2  # keep it symmetric against rounding drift
        trace_cap = 1e3 * len(x) / self.ridge
        if np.trace(self.P) > trace_cap:  # windup guard when features stop varying
            self.P *= trace_cap / np.trace(self.P)
        self.n += 1
        return pred

    def state_dict(self) -> dict[str, Any]:
        return {"w": self.w.tolist(), "P": self.P.tolist(), "n": self.n,
                "forgetting": self.forgetting, "ridge": self.ridge}

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "RecursiveLeastSquares":
        m = cls(len(d["w"]), d["forgetting"], d["ridge"])
        m.w, m.P, m.n = np.array(d["w"], dtype=float), np.array(d["P"], dtype=float), int(d["n"])
        return m


def rank_ic_by_day(rows: list[tuple[str, float, float]], every: int = 1, window: int | None = None) -> list[float]:
    """Spearman correlation between predictions and outcomes across assets, per sample date (one date in every
    ``every``: overlapping outcomes would inflate the t-statistic), newest ``window`` dates."""
    by_day: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for day, p, y in rows:
        by_day[day].append((p, y))
    days = sorted(by_day)
    if window:
        days = days[-window * every:]
    out = []
    for day in days[::-1][::every]:
        pts = np.array(by_day[day])
        if len(pts) >= 3 and pts[:, 0].std() > 0 and pts[:, 1].std() > 0:
            out.append(float(np.corrcoef(pd.Series(pts[:, 0]).rank(), pd.Series(pts[:, 1]).rank())[0, 1]))
    return out


def _ic_summary(ics: list[float]) -> dict[str, Any]:
    if len(ics) < 2:
        return {"rank_ic_mean": float(np.mean(ics)) if ics else None, "rank_ic_t": None, "days": len(ics)}
    arr = np.array(ics)
    sd = arr.std(ddof=1)
    return {"rank_ic_mean": float(arr.mean()), "rank_ic_t": float(arr.mean() / sd * math.sqrt(len(arr))) if sd > 0
            else None, "days": len(arr)}


class GradientBoostingModel:
    """Non-linear companion of the linear model (LightGBM), re-trained periodically on every resolved sample.

    Each fit first trains on the older 80 % of dates and scores the newest 20 % (with a purge gap of ``horizon``
    bars so no training outcome overlaps the test period): that out-of-sample rank IC weighs the model until its live
    predictions build their own track record. Then it is re-fitted on everything. Optional: without the ``lightgbm``
    package the agent simply uses the linear model."""

    PARAMS = {"objective": "huber", "alpha": 1.5, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
              "verbose": -1, "seed": 0, "deterministic": True, "num_threads": 1}

    def __init__(self, rounds: int = 200):
        self.rounds = rounds
        self.booster: Any = None
        self.holdout: dict[str, Any] = {}
        self.trained_at: str | None = None
        self.n_train = 0

    @staticmethod
    def available() -> bool:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return False
        return True

    def _train(self, X: np.ndarray, y: np.ndarray) -> Any:
        import lightgbm as lgb

        return lgb.train(self.PARAMS, lgb.Dataset(X, y, feature_name=FEATURES, free_raw_data=True),
                         num_boost_round=self.rounds)

    def fit(self, samples: list[dict[str, Any]], horizon: int, now: str | None = None) -> bool:
        data = [s for s in samples if s.get("y_rel") is not None]
        if len(data) < 2000 or not self.available():
            return False
        dates = sorted({s["ts"][:10] for s in data})
        cut = dates[int(len(dates) * 0.8)]
        train = [s for s in data if s["realized"][:10] < cut]  # purge: outcomes known before the test period
        test = [s for s in data if s["ts"][:10] >= cut]
        if len(train) >= 1000 and test:
            b = self._train(np.array([s["x"] for s in train]), np.array([s["y_rel"] for s in train]))
            preds = b.predict(np.array([s["x"] for s in test]))
            self.holdout = _ic_summary(rank_ic_by_day([(s["ts"][:10], float(p), s["y_rel"])
                                                       for s, p in zip(test, preds)], every=horizon))
        self.booster = self._train(np.array([s["x"] for s in data]), np.array([s["y_rel"] for s in data]))
        self.trained_at, self.n_train = now or dates[-1], len(data)
        return True

    def predict(self, x: np.ndarray) -> float | None:
        return float(self.booster.predict(x.reshape(1, -1))[0]) if self.booster is not None else None

    def importance(self, k: int = 6) -> list[str]:
        if self.booster is None:
            return []
        gain = self.booster.feature_importance("gain")
        return [FEATURES[i] for i in np.argsort(gain)[::-1][:k]]

    def state_dict(self) -> dict[str, Any] | None:
        if self.booster is None:
            return None
        return {"model": self.booster.model_to_string(), "holdout": self.holdout, "trained_at": self.trained_at,
                "n_train": self.n_train, "rounds": self.rounds}

    def load_state_dict(self, d: dict[str, Any] | None) -> None:
        if not d or not self.available():
            return
        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=d["model"])
        self.holdout, self.trained_at = d.get("holdout", {}), d.get("trained_at")
        self.n_train, self.rounds = d.get("n_train", 0), d.get("rounds", self.rounds)


class OnlineLinearAgent(Agent):
    """Learns which assets will beat the rest, keeps learning every day, and trades on its predictions.

    Each step it (1) learns from samples whose outcome is now known, (2) predicts every symbol and (3) holds an
    equal-weight long portfolio of the ``top_k`` best predictions above ``entry_threshold`` whose expected return
    beats ``cost_multiple`` x the round-trip cost (kept while above ``exit_threshold``).

    What it learns: the return from the open after the decision bar to the open ``horizon`` bars later (what it can
    actually trade: the overnight gap after the last completed bar is gone when it buys), in units of the asset's
    volatility, minus the average of all assets over the same dates (it picks assets, so the market's move is noise).
    Features: price, technical patterns, volume, market context (SPY; BTC-USD for crypto), strength against the
    market and its sector, and interactions with the market regime.

    Two models: recursive least squares with forgetting per *day* (``forgetting`` ** (1 / number of assets) per
    sample, so the memory does not shrink as the universe grows) and, when ``lightgbm`` is installed, gradient
    boosting re-trained every ``gbm_every_days``. Their predictions are averaged with weights = recent out-of-sample
    rank IC of each. With ``ic_gate`` it only trades while the combined rank IC is significant.

    Args:
        symbols: universe (default: the observation's symbols).
        top_k: maximum number of positions.
        entry_threshold / exit_threshold: predicted relative return, in volatility units, to open / keep a position.
        cost_multiple: expected return over the horizon must exceed this many round trips of fees + spread.
        max_weight: cap per position (fraction of equity); ``invested_fraction`` caps the total.
        rebalance_band: minimum weight change worth trading.
        min_hold_days: a position is not sold for a weak signal before this many days (costs!),
            only if its prediction falls below ``-entry_threshold``.
        forgetting: RLS forgetting per day (0.998 ~ the last 500 days matter most).
    """

    def __init__(self, symbols: Sequence[str] | None = None, top_k: int = 4, entry_threshold: float = 0.08,
                 exit_threshold: float = -0.02, max_weight: float = 0.25, invested_fraction: float = 0.95,
                 rebalance_band: float = 0.06, min_hold_days: int = 5, forgetting: float = 0.998,
                 ridge: float = 50.0, min_samples: int = 300, pretrain_bars: int = 750, track: int = 15000,
                 horizon: int = 1, ic_gate: bool = False, ic_window: int = 60, ic_min_t: float = 2.0,
                 ic_min_days: int = 20, cost_multiple: float = 1.5, gbm: bool = True, gbm_every_days: int = 7):
        self.symbols = list(symbols) if symbols else None
        self.top_k, self.entry_threshold, self.exit_threshold = top_k, entry_threshold, exit_threshold
        self.max_weight, self.invested_fraction, self.rebalance_band = max_weight, invested_fraction, rebalance_band
        self.min_samples, self.pretrain_bars, self.track = min_samples, pretrain_bars, track
        self.min_hold_days = min_hold_days
        self.horizon = max(1, int(horizon))
        self.ic_gate, self.ic_window, self.ic_min_t, self.ic_min_days = ic_gate, ic_window, ic_min_t, ic_min_days
        self.cost_multiple = cost_multiple
        self.day_forgetting = forgetting
        self.use_gbm, self.gbm_every_days = gbm, gbm_every_days
        self.entries: dict[str, str] = {}  # symbol -> when the agent opened it
        self.model = RecursiveLeastSquares(len(FEATURES), forgetting, ridge)
        self.gbm = GradientBoostingModel()
        self.gbm_attempt: str | None = None  # last (re)training attempt
        self._feature_cache: dict = {}  # in-memory only: features of past bars do not change
        self.pending: dict[str, list[dict[str, Any]]] = {}  # symbol -> samples waiting for their outcome
        # [combined prediction made at the time, realised, sample date, linear prediction, boosting prediction]
        self.history: list[list[Any]] = []
        self.pretrained_on = 0
        self.name = f"OnlineLinearAgent(top{top_k})"
        self._last: dict[str, Any] | None = None

    # ------------------------------------------------------------------ learning
    def _bars(self, obs: Observation, sym: str, n: int) -> pd.DataFrame:
        return obs.history(sym, limit=n)

    def _set_forgetting(self, n_assets: int) -> None:
        self.model.forgetting = self.day_forgetting ** (1.0 / max(n_assets, 1))  # lambda_sample = lambda_day^(1/N)

    def _weights(self) -> tuple[float, float]:
        """Ensemble weights (linear, boosting) = positive recent out-of-sample rank IC of each model."""
        if self.gbm.booster is None:
            return 1.0, 0.0
        lin = self.ic_stats(3)
        gb = self.ic_stats(4)
        wl = max(lin["rank_ic_mean"] or 0.0, 0.0)
        wg = max((gb["rank_ic_mean"] if gb["days"] >= 10 else self.gbm.holdout.get("rank_ic_mean")) or 0.0, 0.0)
        return (wl / (wl + wg), wg / (wl + wg)) if wl + wg > 0 else (1.0, 0.0)

    def _predict(self, x: np.ndarray) -> tuple[float, float, float | None]:
        lin = self.model.predict(x)
        gb = self.gbm.predict(x)
        wl, wg = self._weights()
        return (wl * lin + wg * gb if gb is not None else lin), lin, gb

    def _log(self, s: dict[str, Any], pred: float, lin: float, gb: float | None) -> None:
        self.history.append([round(pred, 4), round(s["y_rel"], 4), s["ts"][:10], round(lin, 4),
                             None if gb is None else round(gb, 4)])

    def _learn(self, bars: dict[str, pd.DataFrame | pd.Series]) -> int:
        """Learn from pending samples whose outcome is now known (all assets together: the target is relative to
        the average of the assets resolved on the same dates). Predictions logged are the ones made at the time,
        so the information coefficient is measured strictly out of sample."""
        resolved = []
        for sym, queue in self.pending.items():
            df = bars.get(sym)
            if df is None or not isinstance(queue, list):
                continue
            idx = _closes(df).index
            waiting = []
            for p in queue:
                ts = pd.Timestamp(p["ts"])
                pos = int(idx.get_indexer([ts])[0])
                if pos < 0:
                    if len(idx) and ts >= idx[0]:
                        waiting.append(p)
                    continue
                ee = _entry_exit(df, pos, self.horizon)
                if ee is None:
                    waiting.append(p)
                elif ee[0] > 0 and ee[1] > 0:
                    resolved.append({**p, "y": float(np.clip(math.log(ee[1] / ee[0])
                                                             / (p["sigma"] * math.sqrt(self.horizon)), -6, 6)),
                                     "realized": ee[2].isoformat()})
            self.pending[sym] = waiting
        demean_targets(resolved)
        for s in sorted(resolved, key=lambda s: s["ts"]):
            self.model.update(np.array(s["x"]), s["y_rel"])
            self._log(s, s["pred"], s.get("lin", s["pred"]), s.get("gbm"))
        del self.history[:-self.track]
        return len(resolved)

    def pretrain(self, frames: dict[str, pd.DataFrame | pd.Series], groups: dict[str, str] | None = None,
                 markets: dict[str, str] | None = None) -> int:
        """Walk forward through history (OHLCV frames or closes; all symbols interleaved in time order).

        Day by day: first learn the samples whose outcome is known that day, then predict the new ones, so no
        prediction ever uses information from its own future. Then fits the boosting model on everything.
        Returns the number of samples learned."""
        if groups is None or markets is None:
            groups, markets = default_groups(list(frames))
        self._set_forgetting(len(frames))
        samples = build_panel(frames, groups, markets, horizon=self.horizon, cache=self._feature_cache)
        demean_targets(samples)
        created: dict[str, list[int]] = defaultdict(list)
        realised: dict[str, list[int]] = defaultdict(list)
        for k, s in enumerate(samples):
            created[s["ts"][:10]].append(k)
            if s.get("y_rel") is not None:
                realised[s["realized"][:10]].append(k)
        preds: dict[int, float] = {}
        n = 0
        for day in sorted(set(created) | set(realised)):
            for k in realised.get(day, []):
                s = samples[k]
                self.model.update(s["x"], s["y_rel"])
                self._log(s, preds[k], preds[k], None)
                n += 1
            for k in created.get(day, []):
                preds[k] = self.model.predict(samples[k]["x"])
        del self.history[:-self.track]
        self.pretrained_on += n
        if self.use_gbm:
            self.gbm_attempt = max(created) if created else None
            self.gbm.fit(samples, self.horizon, now=self.gbm_attempt)
        return n

    def ic_stats(self, col: int = 0) -> dict[str, Any]:
        """Daily cross-sectional rank IC (Spearman between predictions and outcomes across assets) over the last
        ``ic_window`` non-overlapping sample dates: mean, t-statistic and number of days. ``col`` 0 = combined
        prediction, 3 = linear model, 4 = boosting. Pooling all pairs instead would mostly measure the market
        moving all assets together."""
        rows = [(r[2], r[col], r[1]) for r in self.history if len(r) > col and r[col] is not None]
        return _ic_summary(rank_ic_by_day(rows, every=self.horizon, window=self.ic_window))

    def has_edge(self) -> bool:
        st = self.ic_stats()
        return (st["days"] >= self.ic_min_days and (st["rank_ic_mean"] or 0) > 0
                and (st["rank_ic_t"] or 0) >= self.ic_min_t)

    @staticmethod
    def _round_trip_cost(obs: Observation, sym: str) -> float:
        """Fees both ways + the spread, as a fraction of the amount traded."""
        spread, fee = 0.001, 0.0
        try:
            spread = float(obs.quote(sym).get("spread_bps") or 0.0) / 1e4
        except Exception:  # noqa: BLE001 - no quote: keep the default
            pass
        try:
            sim = obs.sim
            fees = sim.config.fee_model()
            fee = float(fees.schedule_for(sim.universe.get(sym).asset_class).taker_pct)
        except Exception:  # noqa: BLE001
            pass
        return spread + 2 * fee

    # ------------------------------------------------------------------ acting
    def _frames(self, obs: Observation, symbols: list[str], n: int) -> dict[str, pd.DataFrame]:
        frames = {}
        for s in symbols:
            try:
                frames[s] = self._bars(obs, s, n)
            except (DataUnavailableError, KeyError) as exc:  # learn / predict on what is available
                log.warning("%s: no history for %s (%s)", self.name, s, exc)
        return frames

    def _gbm_due(self, now: datetime, n_symbols: int) -> bool:
        if not (self.use_gbm and self.gbm.available() and self.pretrain_bars > MIN_BARS):
            return False
        if n_symbols * (self.pretrain_bars - MIN_BARS - self.horizon) < 2000:
            return False  # not enough history for a boosting model
        last = self.gbm_attempt or self.gbm.trained_at
        return last is None or (now.date() - datetime.fromisoformat(last[:10]).date()).days >= self.gbm_every_days

    def act(self, obs: Observation) -> list[Any]:
        symbols = list(self.symbols or obs.symbols)
        groups, markets = default_groups(symbols, getattr(obs._sim, "universe", None))
        context = [m for m in set(markets.values()) if m not in symbols]
        self._set_forgetting(len(symbols))
        if self.model.n == 0 and self.pretrain_bars > MIN_BARS:
            n = self.pretrain(self._frames(obs, symbols + context, self.pretrain_bars), groups, markets)
            log.info("%s pretrained on %d samples", self.name, n)
        elif self._gbm_due(obs.timestamp, len(symbols)):
            self.gbm_attempt = obs.timestamp.isoformat()
            samples = build_panel(self._frames(obs, symbols + context, self.pretrain_bars), groups, markets,
                                  horizon=self.horizon, cache=self._feature_cache)
            demean_targets(samples)
            self.gbm.fit(samples, self.horizon, now=obs.timestamp.isoformat())
        lookback = max(MIN_BARS + 8 + self.horizon, 80)
        bars = self._frames(obs, symbols + context, lookback)
        self._learn(bars)
        preds: dict[str, float] = {}
        sigmas: dict[str, float] = {}
        for s in build_panel(bars, groups, markets, last_only=True):
            if s["sym"] not in symbols:
                continue
            pred, lin, gb = self._predict(s["x"])
            preds[s["sym"]], sigmas[s["sym"]] = pred, s["sigma"]
            queue = self.pending.setdefault(s["sym"], [])
            if not any(p["ts"] == s["ts"] for p in queue):
                queue.append({"ts": s["ts"], "x": [round(float(v), 5) for v in s["x"]], "sigma": s["sigma"],
                              "pred": pred, "lin": lin, "gbm": gb})
            del queue[:-(self.horizon + 3)]
        wl, wg = self._weights()
        self._last = {"predictions": {s: round(v, 4) for s, v in sorted(preds.items(), key=lambda kv: -kv[1])},
                      "samples": self.model.n, "ensemble": {"linear": round(wl, 3), "gbm": round(wg, 3)}}
        if self.model.n < self.min_samples or obs.open_orders:
            self._last["signals"] = ["learning: not trading yet" if self.model.n < self.min_samples
                                     else "open orders pending: waiting"]
            return []
        if self.ic_gate and not self.has_edge():
            st = self.ic_stats()
            self._last["signals"] = [f"no statistical edge (rank IC {st['rank_ic_mean'] or 0:+.3f}, "
                                     f"t {st['rank_ic_t'] or 0:.2f}): cash"]
            return [ClosePosition(s) for s, q in obs.positions.items() if q > 0]
        return self._rebalance(obs, preds, sigmas)

    def _held_days(self, obs: Observation, sym: str) -> float:
        opened = self.entries.get(sym)
        return (obs.timestamp - datetime.fromisoformat(opened)).total_seconds() / 86400 if opened else math.inf

    def _pays_costs(self, obs: Observation, sym: str, pred: float, sigma: float) -> bool:
        """Expected relative return over the horizon beats ``cost_multiple`` round trips of costs."""
        return pred * sigma * math.sqrt(self.horizon) > self.cost_multiple * self._round_trip_cost(obs, sym)

    def _rebalance(self, obs: Observation, preds: dict[str, float], sigmas: dict[str, float]) -> list[Any]:
        for sym in list(self.entries):
            if obs.position(sym) <= 0:
                del self.entries[sym]
        young = {s for s in preds if obs.position(s) > 0 and self._held_days(obs, s) < self.min_hold_days
                 and preds[s] > -self.entry_threshold}
        keep = {s for s, p in preds.items() if obs.position(s) > 0 and p > self.exit_threshold} | young
        fresh = [s for s, p in sorted(preds.items(), key=lambda kv: -kv[1])
                 if p > self.entry_threshold and s not in keep and self._pays_costs(obs, s, p, sigmas[s])]
        chosen = sorted(keep, key=lambda s: -preds[s]) + fresh
        chosen = chosen[: self.top_k]
        target = min(self.max_weight, self.invested_fraction / max(len(chosen), 1))
        actions: list[Any] = []
        signals: list[str] = []
        for sym in list(obs.positions):
            if sym in preds and sym not in chosen and obs.position(sym) > 0:
                actions.append(ClosePosition(sym))
                signals.append(f"sell {sym}: prediction {preds[sym]:+.3f}σ")
        for sym in chosen:
            cur = obs.weight(sym)
            diff = target - cur
            price = obs.prices.get(sym, float("nan"))
            if abs(diff) < self.rebalance_band or not price > 0:
                continue
            tag = f"pred {preds[sym]:+.3f}σ"
            if diff > 0:
                actions.append(OrderIntent.buy(sym, notional=round(diff * obs.equity, 2), tag=tag))
                signals.append(f"buy {sym} to {target:.0%}: {tag}")
                self.entries.setdefault(sym, obs.timestamp.isoformat())
            else:
                qty = min(float(obs.position(sym)), -diff * obs.equity / price)
                actions.append(OrderIntent.sell(sym, quantity=qty, tag=f"trim, {tag}"))
                signals.append(f"trim {sym} to {target:.0%}: {tag}")
        self._last["signals"] = signals or ["hold"]
        return actions

    def explain(self) -> dict[str, Any] | None:
        return self._last

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        from tradesim.agents import diary as d

        last = getattr(self, "_last", None) or {}
        preds = last.get("predictions") or {}
        sig = (last.get("signals") or [""])[0]
        if sig.startswith("learning"):
            por = f"Todavía estoy aprendiendo ({self.model.n} ejemplos de {self.min_samples}); no opero hasta tener bastantes."
        elif sig.startswith("no statistical edge"):
            por = "Mis predicciones recientes no muestran ventaja estadística, así que me quedo en efectivo."
        elif sig.startswith("open orders"):
            por = "Tenía órdenes pendientes y esperé."
        else:
            top = ", ".join(f"{s} ({v:+.2f}σ)" for s, v in list(preds.items())[:3])
            ens = last.get("ensemble") or {}
            por = (f"Mis modelos (lineal {ens.get('linear', 1):.0%}, árboles {ens.get('gbm', 0):.0%}, pesados por su "
                   f"acierto reciente) predicen qué activos batirán a la media en {self.horizon} días; compro los "
                   f"{self.top_k} mejores si la ventaja esperada paga {self.cost_multiple:g} veces los costes. "
                   f"Hoy los más prometedores: {top}.")
        pos = sum(v > 0 for v in preds.values())
        mean = sum(preds.values()) / len(preds) if preds else 0.0
        segun = (f"Con el mercado {d.tag(situation)}, {pos} de {len(preds)} activos tienen predicción positiva "
                 f"(media {mean:+.2f}σ): " + ("veo más oportunidades que riesgos." if mean > 0 else
                                              "soy prudente y solo entro en los mejores."))
        return d.entry(d.did(orders), por, segun)

    def learning_summary(self) -> dict[str, Any] | None:
        st = self.ic_stats()
        h = np.array([r[:2] for r in self.history[-5000:]], dtype=float) if self.history else np.zeros((0, 2))
        hit = float(np.mean(np.sign(h[:, 0]) == np.sign(h[:, 1])) * 100) if len(h) >= 30 else None
        lin, gb = self.ic_stats(3), self.ic_stats(4)
        wl, wg = self._weights()
        return {
            "method": "Online linear (RLS, forgetting per day) + gradient boosting, weighted by out-of-sample rank IC",
            "target": "open-to-open return over the horizon, relative to the average asset (volatility units)",
            "horizon_bars": self.horizon,
            "forgetting_per_day": self.day_forgetting,
            "models": {"linear": {"weight": round(wl, 3), "rank_ic_mean": lin["rank_ic_mean"], "rank_ic_t": lin["rank_ic_t"]},
                       "gbm": {"weight": round(wg, 3), "rank_ic_mean": gb["rank_ic_mean"], "rank_ic_t": gb["rank_ic_t"],
                               "live_days": gb["days"], "holdout": self.gbm.holdout, "trained_at": self.gbm.trained_at,
                               "top_features": self.gbm.importance()} if self.gbm.booster is not None else None},
            "samples_learned": self.model.n,
            "pretrained_samples": self.pretrained_on,
            "trading_enabled": self.model.n >= self.min_samples and (not self.ic_gate or self.has_edge()),
            "ic_gate": self.ic_gate,
            "rank_ic_mean": st["rank_ic_mean"], "rank_ic_t": st["rank_ic_t"], "rank_ic_days": st["days"],
            "direction_hit_rate_pct": hit,
            "weights": dict(zip(FEATURES, (round(float(w), 4) for w in self.model.w))),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"version": 2, "model": self.model.state_dict(), "pending": self.pending, "history": self.history,
                "pretrained_on": self.pretrained_on, "entries": self.entries, "gbm": self.gbm.state_dict(),
                "gbm_attempt": self.gbm_attempt}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.entries = state.get("entries", {})
        if len(state["model"]["w"]) != len(FEATURES) or state.get("version", 1) < 2:
            # The features or the target changed: relearn from history on the next act().
            log.warning("%s: saved model is from an older version (%d features, expected %d); it will be retrained",
                        self.name, len(state["model"]["w"]), len(FEATURES))
            return
        self.model = RecursiveLeastSquares.from_state(state["model"])
        self.pending = state.get("pending", {})
        self.history = state.get("history", [])
        self.pretrained_on = state.get("pretrained_on", 0)
        if self.use_gbm:
            self.gbm.load_state_dict(state.get("gbm"))
        self.gbm_attempt = state.get("gbm_attempt")

    def reset(self) -> None:
        self.entries = {}


class WalkForwardRSIAgent(Agent):
    """RSI mean reversion whose parameters are re-tuned periodically by backtesting recent history.

    Every ``retune_days`` it simulates each setting in ``grid`` over the last ``lookback_bars`` bars in
    a ``TradingSimulator`` (same costs model as the account), scores it by the *median* Sharpe ratio of
    ``windows`` consecutive sub-periods (quarters by default: a setting must work in most of them, not ride one
    lucky stretch) and adopts the best one only if it beats the current setting by ``min_improvement``. Trading is
    delegated to :class:`SimpleRSIAgent` with the adopted setting.
    """

    def __init__(self, symbols: Sequence[str] | None = None,
                 grid: Sequence[tuple[int, float, float]] | None = None, retune_days: int = 7,
                 lookback_bars: int = 252, min_improvement: float = 0.15, position_pct: float = 10.0,
                 max_positions: int = 8, initial: tuple[int, float, float] = (14, 30.0, 70.0), windows: int = 4):
        self.symbols = list(symbols) if symbols else None
        self.grid = [tuple(g) for g in (grid or product((7, 14, 21), (25.0, 30.0), (65.0, 75.0)))]
        self.retune_days, self.lookback_bars, self.min_improvement = retune_days, lookback_bars, min_improvement
        self.position_pct, self.max_positions = position_pct, max_positions
        self.windows = max(1, int(windows))
        self.params: tuple[int, float, float] = tuple(initial)  # type: ignore[assignment]
        self.last_tune: str | None = None
        self.tunes: list[dict[str, Any]] = []
        self.name = "WalkForwardRSIAgent"
        self._inner = self._make_inner()
        self._tuned_now = False

    def _make_inner(self) -> SimpleRSIAgent:
        p, lo, hi = self.params
        return SimpleRSIAgent(self.symbols, period=int(p), oversold=lo, overbought=hi,
                              position_pct=self.position_pct, max_positions=self.max_positions)

    def _due(self, now: datetime) -> bool:
        if self.last_tune is None:
            return True
        return now.date() - datetime.fromisoformat(self.last_tune).date() >= timedelta(days=self.retune_days)

    def _score_grid(self, obs: Observation) -> dict[tuple[int, float, float], float]:
        from tradesim.data.memory import InMemoryProvider
        from tradesim.practice import run_episode
        from tradesim.simulator import SimulatorConfig, TradingSimulator

        sim = obs.sim
        symbols = self.symbols or obs.symbols
        frames = {}
        for s in symbols:
            try:
                frames[s] = obs.history(s, self.lookback_bars + 80)
            except DataUnavailableError:
                continue
        frames = {s: f for s, f in frames.items() if len(f) > 80}
        if not frames:
            return {}
        start = min(f.index[70] for f in frames.values())
        end = max(f.index[-1] for f in frames.values()) + pd.Timedelta(days=1)
        fees = getattr(getattr(sim, "config", None), "fees", "alpaca")
        env = TradingSimulator(getattr(sim, "universe", None), InMemoryProvider(frames),
                               SimulatorConfig(initial_cash=10_000, fees=fees, warmup_bars=70),
                               symbols=list(frames), start=start, end=end)
        scores: dict[tuple[int, float, float], float] = {}
        for params in dict.fromkeys(self.grid + [self.params]):
            p, lo, hi = params
            agent = SimpleRSIAgent(list(frames), period=int(p), oversold=lo, overbought=hi,
                                   position_pct=self.position_pct, max_positions=self.max_positions)
            scores[params] = self.median_sharpe(run_episode(agent, env, seed=0).equity, self.windows)
        env.close()
        return scores

    @staticmethod
    def median_sharpe(equity: pd.Series, windows: int) -> float:
        """Median of the annualised Sharpe ratios of ``windows`` consecutive sub-periods (-9 if undefined)."""
        daily = equity.resample("1D").last().dropna().pct_change().dropna()
        out = []
        for chunk in np.array_split(daily.to_numpy(dtype=float), windows):
            sd = chunk.std(ddof=1) if len(chunk) > 5 else 0.0
            out.append(chunk.mean() / sd * math.sqrt(252) if sd > 0 else (0.0 if len(chunk) > 5 else -9.0))
        val = float(np.median(out)) if out else -9.0
        return val if math.isfinite(val) else -9.0

    def _tune(self, obs: Observation) -> None:
        scores = self._score_grid(obs)
        self.last_tune = obs.timestamp.isoformat()
        if not scores:
            return
        best = max(scores, key=scores.get)
        current = scores.get(self.params, -9.0)
        switched = best != self.params and scores[best] > current + self.min_improvement
        record = {"date": obs.timestamp.isoformat(), "previous": list(self.params), "current_sharpe": round(current, 3),
                  "best": list(best), "best_sharpe": round(scores[best], 3), "switched": switched,
                  "top": [[list(k), round(v, 3)] for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:5]]}
        if switched:
            self.params = best
            self._inner = self._make_inner()
        self.tunes.append(record)
        del self.tunes[:-30]

    def act(self, obs: Observation) -> Any:
        self._tuned_now = False
        if self._due(obs.timestamp):
            self._tune(obs)
            self._tuned_now = True
        return self._inner.act(obs)

    def explain(self) -> dict[str, Any] | None:
        out = dict(self._inner.explain() or {})
        out["params"] = {"period": self.params[0], "oversold": self.params[1], "overbought": self.params[2]}
        if self._tuned_now and self.tunes:
            out["retuned"] = self.tunes[-1]
        return out

    def diary(self, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str] | None:
        note = self._inner.diary(situation, orders) or {}
        if self._tuned_now and self.tunes:
            t = self.tunes[-1]
            note["por_que"] = (f"Hoy reajusté mis parámetros simulando el pasado reciente: "
                               + (f"cambio a RSI({t['best'][0]}) {t['best'][1]:g}/{t['best'][2]:g} (Sharpe "
                                  f"{t['best_sharpe']:.2f} frente a {t['current_sharpe']:.2f})." if t["switched"]
                                  else "mantengo los actuales, ninguno era claramente mejor.")
                               + " " + note.get("por_que", ""))
        return note

    def learning_summary(self) -> dict[str, Any] | None:
        return {"method": "Walk-forward re-tuning in the simulator (median quarterly Sharpe after costs)",
                "params": {"period": self.params[0], "oversold": self.params[1], "overbought": self.params[2]},
                "last_tune": self.last_tune, "retune_every_days": self.retune_days,
                "recent_tunes": self.tunes[-5:]}

    def state_dict(self) -> dict[str, Any]:
        return {"params": list(self.params), "last_tune": self.last_tune, "tunes": self.tunes}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.params = tuple(state["params"])  # type: ignore[assignment]
        self.last_tune = state.get("last_tune")
        self.tunes = state.get("tunes", [])
        self._inner = self._make_inner()
