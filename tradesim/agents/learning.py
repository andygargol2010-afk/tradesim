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

FEATURES = ["bias", "ret_1", "ret_5", "ret_20", "ret_60", "vol_ratio", "rsi_14", "dist_sma20", "dist_sma50",
            *PATTERN_FEATURES]
MIN_BARS = 61


def price_features(closes: np.ndarray, opens: np.ndarray | None = None, highs: np.ndarray | None = None,
                   lows: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """Scale-free features from daily bars (oldest first, last = most recent completed bar).

    Price features come from the closes; the technical-pattern features (:data:`PATTERN_FEATURES`: 20-bar
    breakout, trend quality, candlestick signal) also use opens/highs/lows when given.

    Returns ``(x, sigma)`` with ``sigma`` the 20-bar volatility of log returns, or ``None`` if there is
    not enough history. Every return-like feature is divided by ``sigma * sqrt(horizon)`` so that one
    model can be shared by assets as different as an ETF and a small-cap token.
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
    x = np.r_[x, pattern_features(c[-MIN_BARS:], *ohlc)]
    return np.clip(x, -5.0, 5.0), sigma


def _frame_features(df: pd.DataFrame | pd.Series, end: int | None = None) -> tuple[np.ndarray, float] | None:
    """:func:`price_features` of a closes Series or an OHLC frame, using rows up to ``end`` (exclusive)."""
    sl = slice(max(0, (end if end is not None else len(df)) - MIN_BARS), end)
    if isinstance(df, pd.Series):
        return price_features(df.to_numpy(dtype=float)[sl])
    cols = [df[k].to_numpy(dtype=float)[sl] if k in df else None for k in ("close", "open", "high", "low")]
    return price_features(*cols)


def _closes(df: pd.DataFrame | pd.Series) -> pd.Series:
    return df if isinstance(df, pd.Series) else df["close"]


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


class OnlineLinearAgent(Agent):
    """Learns which price patterns precede gains, keeps learning every day, and trades on its predictions.

    Each step it (1) learns from bars completed since its last look (target = next-bar log return
    divided by the asset's volatility), (2) predicts the next bar for every symbol, and (3) holds an
    equal-weight long portfolio of the ``top_k`` symbols whose prediction exceeds ``entry_threshold``
    (kept while above ``exit_threshold``; the gap avoids churning). Nothing is traded until the model
    has seen ``min_samples`` samples; the first call warm-starts on up to ``pretrain_bars`` of history.

    Args:
        symbols: universe (default: the observation's symbols).
        top_k: maximum number of positions.
        entry_threshold / exit_threshold: predicted return, in volatility units, to open / keep a position.
        max_weight: cap per position (fraction of equity); ``invested_fraction`` caps the total.
        rebalance_band: minimum weight change worth trading.
        min_hold_days: a position is not sold for a weak signal before this many days (costs!),
            only if its prediction falls below ``-entry_threshold``.
        forgetting: RLS forgetting factor (0.998 ~ the last 500 samples matter most).
    """

    def __init__(self, symbols: Sequence[str] | None = None, top_k: int = 4, entry_threshold: float = 0.08,
                 exit_threshold: float = -0.02, max_weight: float = 0.25, invested_fraction: float = 0.95,
                 rebalance_band: float = 0.06, min_hold_days: int = 5, forgetting: float = 0.998,
                 ridge: float = 50.0, min_samples: int = 300, pretrain_bars: int = 750, track: int = 4000,
                 horizon: int = 1, ic_gate: bool = False, ic_window: int = 60, ic_min_t: float = 2.0,
                 ic_min_days: int = 20):
        self.symbols = list(symbols) if symbols else None
        self.top_k, self.entry_threshold, self.exit_threshold = top_k, entry_threshold, exit_threshold
        self.max_weight, self.invested_fraction, self.rebalance_band = max_weight, invested_fraction, rebalance_band
        self.min_samples, self.pretrain_bars, self.track = min_samples, pretrain_bars, track
        self.min_hold_days = min_hold_days
        self.horizon = max(1, int(horizon))
        self.ic_gate, self.ic_window, self.ic_min_t, self.ic_min_days = ic_gate, ic_window, ic_min_t, ic_min_days
        self.entries: dict[str, str] = {}  # symbol -> when the agent opened it
        self.model = RecursiveLeastSquares(len(FEATURES), forgetting, ridge)
        self.pending: dict[str, list[dict[str, Any]]] = {}  # symbol -> samples waiting for their outcome
        self.history: list[list[Any]] = []  # [prediction made at the time, realised, sample date] (out of sample)
        self.pretrained_on = 0
        self.name = f"OnlineLinearAgent(top{top_k})"
        self._last: dict[str, Any] | None = None

    # ------------------------------------------------------------------ learning
    def _bars(self, obs: Observation, sym: str, n: int) -> pd.DataFrame:
        return obs.history(sym, limit=n)

    def _target(self, close_then: float, close_later: float, sigma: float) -> float:
        """Return over ``horizon`` bars in units of the asset's volatility (clipped)."""
        return float(np.clip(math.log(close_later / close_then) / (sigma * math.sqrt(self.horizon)), -6, 6))

    def _learn_symbol(self, sym: str, bars: pd.DataFrame | pd.Series) -> None:
        """Learn from samples whose ``horizon`` bars have now completed, then add today's sample.

        The prediction logged for each sample is the one the model made when the sample was created, so
        the information coefficient is measured strictly out of sample."""
        closes = _closes(bars)
        if len(closes) < MIN_BARS:
            return
        queue = self.pending.get(sym) or []
        if isinstance(queue, dict):  # state saved by an older version (one sample per symbol)
            queue = [queue]
        waiting = []
        for p in queue:
            later = closes[closes.index > pd.Timestamp(p["ts"])]
            if len(later) >= self.horizon:
                y = self._target(p["close"], float(later.iloc[self.horizon - 1]), p["sigma"])
                x = np.array(p["x"])
                pred = p.get("pred", self.model.predict(x))
                self.model.update(x, y)
                self.history.append([float(pred), y, p["ts"][:10]])
            else:
                waiting.append(p)
        del self.history[:-self.track]
        last_ts = closes.index[-1].isoformat()
        if not any(p["ts"] == last_ts for p in waiting):
            feats = _frame_features(bars)
            if feats is not None:
                waiting.append({"ts": last_ts, "x": feats[0].tolist(), "sigma": feats[1],
                                "close": float(closes.iloc[-1]), "pred": self.model.predict(feats[0])})
        self.pending[sym] = waiting[-(self.horizon + 2):]

    def pretrain(self, frames: dict[str, pd.DataFrame | pd.Series]) -> int:
        """Walk forward through history (OHLC frames or closes; all symbols interleaved in time order).

        Day by day: first learn the samples whose outcome is known that day, then predict the new ones,
        so no prediction ever uses information from its own future. Returns the number of samples learned."""
        h = self.horizon
        created: dict[pd.Timestamp, list[tuple[np.ndarray, float, str]]] = {}
        realised: dict[pd.Timestamp, list[tuple[int, float]]] = {}
        xs: list[np.ndarray] = []
        for sym, bars in frames.items():
            closes = _closes(bars)
            c = closes.to_numpy(dtype=float)
            for t in range(MIN_BARS - 1, len(c) - h):
                feats = _frame_features(bars, t + 1)
                if feats is None:
                    continue
                k = len(xs)
                xs.append(feats[0])
                created.setdefault(closes.index[t], []).append((k, closes.index[t].isoformat()[:10]))
                realised.setdefault(closes.index[t + h], []).append((k, self._target(c[t], c[t + h], feats[1])))
        preds: dict[int, tuple[float, str]] = {}
        n = 0
        for day in sorted(set(created) | set(realised)):
            for k, y in realised.get(day, []):
                pred, ts = preds.pop(k)
                self.model.update(xs[k], y)
                self.history.append([pred, y, ts])
                n += 1
            for k, ts in created.get(day, []):
                preds[k] = (self.model.predict(xs[k]), ts)
        del self.history[:-self.track]
        self.pretrained_on += n
        return n

    def ic_stats(self) -> dict[str, Any]:
        """Daily cross-sectional rank IC (Spearman between predictions and outcomes across assets) over the
        last ``ic_window`` non-overlapping sample dates: mean, t-statistic and number of days. Pooling all pairs instead
        would mostly measure the market moving all assets together."""
        by_day: dict[str, list[tuple[float, float]]] = {}
        for row in self.history:
            if len(row) >= 3:
                by_day.setdefault(row[2], []).append((row[0], row[1]))
        ics = []
        # one day in every ``horizon``: outcomes of consecutive days overlap and would inflate the t-statistic
        days = sorted(by_day)[-self.ic_window * self.horizon:][::-1][::self.horizon]
        for day in days:
            pts = np.array(by_day[day])
            if len(pts) >= 3 and pts[:, 0].std() > 0 and pts[:, 1].std() > 0:
                r0 = pd.Series(pts[:, 0]).rank().to_numpy()
                r1 = pd.Series(pts[:, 1]).rank().to_numpy()
                ics.append(float(np.corrcoef(r0, r1)[0, 1]))
        if len(ics) < 2:
            return {"rank_ic_mean": None, "rank_ic_t": None, "days": len(ics)}
        arr = np.array(ics)
        sd = arr.std(ddof=1)
        return {"rank_ic_mean": float(arr.mean()), "rank_ic_t": float(arr.mean() / sd * math.sqrt(len(arr))) if sd > 0
                else None, "days": len(arr)}

    def has_edge(self) -> bool:
        st = self.ic_stats()
        return (st["days"] >= self.ic_min_days and (st["rank_ic_mean"] or 0) > 0
                and (st["rank_ic_t"] or 0) >= self.ic_min_t)

    # ------------------------------------------------------------------ acting
    def act(self, obs: Observation) -> list[Any]:
        symbols = self.symbols or obs.symbols
        lookback = max(MIN_BARS + 5 + self.horizon, 80)
        if self.model.n == 0 and self.pretrain_bars > MIN_BARS:
            frames = {}
            for s in symbols:
                try:
                    frames[s] = self._bars(obs, s, self.pretrain_bars)
                except DataUnavailableError as exc:  # pretrain on what is available
                    log.warning("%s: no pretraining history for %s (%s)", self.name, s, exc)
            n = self.pretrain(frames)
            log.info("%s pretrained on %d samples", self.name, n)
            for s, bars in frames.items():  # continue online from the last pretraining bar
                self._learn_symbol(s, bars.tail(lookback))
        preds: dict[str, float] = {}
        for sym in symbols:
            try:
                bars = self._bars(obs, sym, lookback)
            except DataUnavailableError as exc:
                log.warning("%s: skipping %s this step (%s)", self.name, sym, exc)
                continue
            self._learn_symbol(sym, bars)
            last = bars["close"].index[-1].isoformat() if len(bars) else None
            cur = next((p for p in self.pending.get(sym, []) if p["ts"] == last), None)
            if cur is not None:
                preds[sym] = self.model.predict(np.array(cur["x"]))
        self._last = {"predictions": {s: round(v, 4) for s, v in sorted(preds.items(), key=lambda kv: -kv[1])},
                      "samples": self.model.n}
        if self.model.n < self.min_samples or obs.open_orders:
            self._last["signals"] = ["learning: not trading yet" if self.model.n < self.min_samples
                                     else "open orders pending: waiting"]
            return []
        if self.ic_gate and not self.has_edge():
            st = self.ic_stats()
            self._last["signals"] = [f"no statistical edge (rank IC {st['rank_ic_mean'] or 0:+.3f}, "
                                     f"t {st['rank_ic_t'] or 0:.2f}): cash"]
            return [ClosePosition(s) for s, q in obs.positions.items() if q > 0]
        return self._rebalance(obs, preds)

    def _held_days(self, obs: Observation, sym: str) -> float:
        opened = self.entries.get(sym)
        return (obs.timestamp - datetime.fromisoformat(opened)).total_seconds() / 86400 if opened else math.inf

    def _rebalance(self, obs: Observation, preds: dict[str, float]) -> list[Any]:
        for sym in list(self.entries):
            if obs.position(sym) <= 0:
                del self.entries[sym]
        young = {s for s in preds if obs.position(s) > 0 and self._held_days(obs, s) < self.min_hold_days
                 and preds[s] > -self.entry_threshold}
        keep = {s for s, p in preds.items() if obs.position(s) > 0 and p > self.exit_threshold} | young
        fresh = [s for s, p in sorted(preds.items(), key=lambda kv: -kv[1])
                 if p > self.entry_threshold and s not in keep]
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

    # ------------------------------------------------------------------ state
    def explain(self) -> dict[str, Any] | None:
        return self._last

    def learning_summary(self) -> dict[str, Any] | None:
        st = self.ic_stats()
        h = np.array([r[:2] for r in self.history[-2000:]], dtype=float) if self.history else np.zeros((0, 2))
        hit = float(np.mean(np.sign(h[:, 0]) == np.sign(h[:, 1])) * 100) if len(h) >= 30 else None
        return {
            "method": "Recursive least squares with forgetting (online)",
            "horizon_bars": self.horizon,
            "samples_learned": self.model.n,
            "pretrained_samples": self.pretrained_on,
            "trading_enabled": self.model.n >= self.min_samples and (not self.ic_gate or self.has_edge()),
            "ic_gate": self.ic_gate,
            "rank_ic_mean": st["rank_ic_mean"], "rank_ic_t": st["rank_ic_t"], "rank_ic_days": st["days"],
            "direction_hit_rate_pct": hit,
            "weights": dict(zip(FEATURES, (round(float(w), 4) for w in self.model.w))),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "pending": self.pending, "history": self.history,
                "pretrained_on": self.pretrained_on, "entries": self.entries}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.entries = state.get("entries", {})
        if len(state["model"]["w"]) != len(FEATURES):
            # The feature set changed (e.g. pattern features added): relearn from history on the next act().
            log.warning("%s: saved model has %d features, expected %d; it will be retrained",
                        self.name, len(state["model"]["w"]), len(FEATURES))
            return
        self.model = RecursiveLeastSquares.from_state(state["model"])
        self.pending = state.get("pending", {})
        self.history = state.get("history", [])
        self.pretrained_on = state.get("pretrained_on", 0)

    def reset(self) -> None:
        self.entries = {}


class WalkForwardRSIAgent(Agent):
    """RSI mean reversion whose parameters are re-tuned periodically by backtesting recent history.

    Every ``retune_days`` it simulates each setting in ``grid`` over the last ``lookback_bars`` bars in
    a ``TradingSimulator`` (same costs model as the account), scores it by Sharpe ratio, and adopts the
    best one only if it beats the current setting by ``min_improvement``. Trading is delegated to
    :class:`SimpleRSIAgent` with the adopted setting.
    """

    def __init__(self, symbols: Sequence[str] | None = None,
                 grid: Sequence[tuple[int, float, float]] | None = None, retune_days: int = 7,
                 lookback_bars: int = 252, min_improvement: float = 0.15, position_pct: float = 10.0,
                 max_positions: int = 8, initial: tuple[int, float, float] = (14, 30.0, 70.0)):
        self.symbols = list(symbols) if symbols else None
        self.grid = [tuple(g) for g in (grid or product((7, 14, 21), (25.0, 30.0), (65.0, 75.0)))]
        self.retune_days, self.lookback_bars, self.min_improvement = retune_days, lookback_bars, min_improvement
        self.position_pct, self.max_positions = position_pct, max_positions
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
            sharpe = run_episode(agent, env, seed=0).metrics.get("sharpe")
            scores[params] = float(sharpe) if isinstance(sharpe, (int, float)) and math.isfinite(sharpe) else -9.0
        env.close()
        return scores

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

    def learning_summary(self) -> dict[str, Any] | None:
        return {"method": "Walk-forward re-tuning in the simulator (Sharpe after costs)",
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
