"""Honest validation: purged cross-validation of the learner's models, walk-forward backtests after costs and the
gate a change must pass before it goes live.

* :func:`purged_kfold` splits samples into contiguous date blocks; training drops every sample whose outcome window
  overlaps the test block (purge) or starts right after it (embargo), so no test information leaks into training.
* :func:`cv_models` scores the linear model, the boosting model and their average by daily cross-sectional rank IC
  on the held-out blocks.
* :func:`backtest` runs an agent through the realistic simulator (fees, spreads, slippage, next-bar execution) and
  compares it with equal-weight buy & hold on the same bars.
* :func:`gate` accepts a candidate only if it beats the current version out of sample (Sharpe after costs and, when
  measured, rank IC) and also beats buy & hold.

Survivorship bias: ``universe.csv`` lists today's index members only. Companies that were delisted or dropped from
the indexes are missing, so any test on past data (and the learner's pre-training) looks better than reality.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from tradesim.agents.learning import GradientBoostingModel, _ic_summary, rank_ic_by_day

SURVIVORSHIP_WARNING = ("universe.csv solo contiene los integrantes actuales de los índices: faltan las empresas que "
                        "quebraron o salieron, así que cualquier resultado sobre el pasado es optimista "
                        "(sesgo de supervivencia).")


def purged_kfold(samples: list[dict[str, Any]], k: int = 5, embargo_days: int = 7
                 ) -> Iterator[tuple[list[int], list[int]]]:
    """``(train, test)`` indexes over samples with ``ts`` (created) and ``realized`` (outcome known) dates."""
    dates = sorted({s["ts"][:10] for s in samples if s.get("y_rel") is not None})
    if len(dates) < k:
        return
    for block in np.array_split(np.array(dates), k):
        lo, hi = str(block[0]), str(block[-1])
        emb = (date.fromisoformat(hi) + timedelta(days=embargo_days)).isoformat()
        test = [i for i, s in enumerate(samples) if s.get("y_rel") is not None and lo <= s["ts"][:10] <= hi]
        train = [i for i, s in enumerate(samples) if s.get("y_rel") is not None
                 and (s["realized"][:10] < lo or s["ts"][:10] > emb)]
        if train and test:
            yield train, test


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 50.0) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def cv_models(samples: list[dict[str, Any]], horizon: int, k: int = 5, embargo_days: int = 7) -> dict[str, Any]:
    """Out-of-sample daily rank IC of each model over purged folds (mean, t-statistic, days)."""
    X = np.array([s["x"] for s in samples], dtype=float)
    y = np.array([s.get("y_rel") if s.get("y_rel") is not None else np.nan for s in samples], dtype=float)
    days = [s["ts"][:10] for s in samples]
    ics: dict[str, list[float]] = {"linear": [], "gbm": [], "ensemble": []}
    gbm = GradientBoostingModel()
    folds = 0
    for train, test in purged_kfold(samples, k, embargo_days):
        folds += 1
        lin = X[test] @ _ridge(X[train], y[train])
        preds = {"linear": lin}
        if gbm.available() and len(train) >= 1000:
            g = gbm._train(X[train], y[train]).predict(X[test])
            preds["gbm"] = g
            preds["ensemble"] = (lin / (lin.std() or 1) + g / (g.std() or 1)) / 2
        for name, p in preds.items():
            ics[name] += rank_ic_by_day([(days[i], float(v), float(y[i])) for i, v in zip(test, p)], every=1)
    out = {name: _ic_summary(v) for name, v in ics.items() if v}
    out.update(folds=folds, samples=int(np.isfinite(y).sum()), horizon=horizon, embargo_days=embargo_days)
    return out


def _sharpe(equity: pd.Series) -> float | None:
    r = equity.resample("1D").last().dropna().pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(252)) if len(r) > 5 and r.std() > 0 else None


def backtest(make_agent: Callable[[], Any], frames: dict[str, pd.DataFrame], universe: Any, start: Any, end: Any,
             fees: str = "alpaca", initial_cash: float = 100.0, max_position_pct: float | None = 30.0,
             auto_stop_atr: float | None = None) -> dict[str, Any]:
    """Walk-forward run of one agent from ``start`` to ``end`` (it only ever sees past bars; ``frames`` may begin
    earlier, as history for warm-up and pre-training) and the same-period equal-weight buy & hold."""
    from tradesim.data.memory import InMemoryProvider
    from tradesim.practice import run_episode
    from tradesim.simulator import SimulatorConfig, TradingSimulator

    first = min(f.index[0] for f in frames.values())
    env = TradingSimulator(universe, InMemoryProvider(frames),
                           SimulatorConfig(initial_cash=initial_cash, fees=fees, max_position_pct=max_position_pct,
                                           auto_stop_atr=auto_stop_atr),
                           symbols=list(frames), start=first.to_pydatetime(), end=pd.Timestamp(end).to_pydatetime())
    agent = make_agent()
    try:
        env.reset()
        cal = pd.DatetimeIndex(env._calendar.astype("datetime64[ns]")).tz_localize("UTC")
        env.reset(start_offset=int((cal < pd.Timestamp(start)).sum()))
        res = run_episode(agent, env, seed=0, reset=False)
    finally:
        env.close()
    m = res.metrics
    out = {"final_equity": round(m["final_equity"], 2), "return_pct": round(m["total_return_pct"], 2),
           "max_drawdown_pct": round(m["max_drawdown_pct"], 2), "sharpe": _sharpe(res.equity),
           "fills": m["num_fills"], "fees": round(float(m.get("total_fees", 0) or 0), 3),
           "start": str(m["start"])[:10], "end": str(m["end"])[:10]}
    if res.benchmark is not None and len(res.benchmark) > 1:
        out["buy_hold_return_pct"] = round((float(res.benchmark.iloc[-1]) / float(res.benchmark.iloc[0]) - 1) * 100, 2)
        out["buy_hold_sharpe"] = _sharpe(res.benchmark)
    if hasattr(agent, "ic_stats"):
        out["rank_ic"] = agent.ic_stats()  # its own out-of-sample predictions during the run
    return out


def gate(candidate: dict[str, Any], baseline: dict[str, Any] | None, min_sharpe_gain: float = 0.0) -> dict[str, Any]:
    """Promote only if the candidate beats the current version and buy & hold out of sample, after costs."""
    checks = []
    cs = candidate.get("sharpe")
    bh = candidate.get("buy_hold_sharpe")
    checks.append({"check": "Sharpe tras costes > buy & hold", "ok": cs is not None and bh is not None and cs > bh,
                   "values": [cs, bh]})
    if baseline is not None:
        bs = baseline.get("sharpe")
        checks.append({"check": "Sharpe tras costes > versión actual",
                       "ok": cs is not None and (bs is None or cs > bs + min_sharpe_gain), "values": [cs, bs]})
        ci, bi = (candidate.get("rank_ic") or {}).get("rank_ic_mean"), (baseline.get("rank_ic") or {}).get("rank_ic_mean")
        if ci is not None or bi is not None:
            checks.append({"check": "IC de rango fuera de muestra > versión actual",
                           "ok": ci is not None and (bi is None or ci > bi), "values": [ci, bi]})
    return {"pass": all(c["ok"] for c in checks), "checks": checks}
