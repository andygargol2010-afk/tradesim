"""Performance metrics from an equity curve and closed trades."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tradesim.types import ClosedTrade


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in bars) spent below a previous peak."""
    underwater = (equity < equity.cummax()).to_numpy()
    longest = cur = 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return longest


def compute_metrics(
    equity: pd.Series,
    closed_trades: Sequence[ClosedTrade] = (),
    periods_per_year: float = 252.0,
    risk_free_annual: float = 0.0,
    total_fees: float = 0.0,
    turnover_notional: float = 0.0,
    exposure: pd.Series | None = None,
) -> dict[str, Any]:
    """Standard performance statistics.

    Equity-curve metrics (return, drawdown, Sharpe, Sortino...) use one point per bar;
    trade metrics (win rate, profit factor, average / best / worst trade) use closed round
    trips, net of their share of entry and exit fees. Trade keys are always present and are
    ``None`` when no trade was closed.

    Args:
        equity: equity values indexed by timestamp (one point per bar).
        closed_trades: round trips for trade statistics.
        periods_per_year: bars per year for annualisation (see ``Timeframe.bars_per_year``).
        risk_free_annual: annual risk-free rate for Sharpe / Sortino.
        exposure: optional gross exposure series (same index) for time-in-market.
    """
    out: dict[str, Any] = {}
    equity = equity.astype(float).dropna()
    if len(equity) < 2:
        final = float(equity.iloc[-1]) if len(equity) else None
        return {"bars": len(equity), "final_equity": final, "total_return_pct": 0.0, "num_trades": len(closed_trades),
                "note": "not enough data"}
    rets = equity.pct_change().dropna()
    rf = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = rets - rf
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(rets) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = rets.std(ddof=1) * math.sqrt(periods_per_year)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * math.sqrt(periods_per_year)
    dd = drawdown_series(equity)
    mdd = -dd.min()

    out.update({
        "bars": int(len(equity)),
        "start_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "end_equity": float(equity.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "cagr_pct": float(cagr * 100) if math.isfinite(cagr) else None,
        "volatility_annual_pct": float(vol * 100),
        "sharpe": float(excess.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year)) if rets.std(ddof=1) > 0 else 0.0,
        "sortino": float(excess.mean() * periods_per_year / downside) if downside > 0 else None,
        "max_drawdown_pct": float(mdd * 100),
        "max_drawdown_duration_bars": max_drawdown_duration(equity),
        "calmar": float(cagr / mdd) if mdd > 0 and math.isfinite(cagr) else None,
        "best_bar_pct": float(rets.max() * 100),
        "worst_bar_pct": float(rets.min() * 100),
        "skew": float(rets.skew()) if len(rets) > 2 else None,
        "var_95_pct": float(np.percentile(rets, 5) * 100),
        "cvar_95_pct": float(rets[rets <= np.percentile(rets, 5)].mean() * 100),
        "total_fees": float(total_fees),
    })
    if turnover_notional and equity.mean() > 0:
        out["turnover_annual"] = float(turnover_notional / equity.mean() / years) if years > 0 else None
    if exposure is not None and len(exposure):
        out["time_in_market_pct"] = float((exposure.astype(float) > 0).mean() * 100)

    pnls = np.array([float(t.pnl) for t in closed_trades])
    out["num_trades"] = int(len(pnls))
    out.update({k: None for k in ("win_rate_pct", "profit_factor", "avg_trade_pnl", "best_trade", "worst_trade",
                                  "avg_win", "avg_loss", "payoff_ratio", "expectancy", "largest_win",
                                  "largest_loss", "avg_trade_return_pct")})
    if len(pnls):
        wins, losses = pnls[pnls > 0], pnls[pnls < 0]
        gross_win, gross_loss = wins.sum(), -losses.sum()
        out.update({
            "win_rate_pct": float(len(wins) / len(pnls) * 100),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else None),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
            "expectancy": float(pnls.mean()),
            "avg_trade_pnl": float(pnls.mean()),
            "avg_trade_return_pct": float(np.mean([t.return_pct for t in closed_trades]) * 100),
            "best_trade": float(pnls.max()),
            "worst_trade": float(pnls.min()),
            "largest_win": float(pnls.max()),
            "largest_loss": float(pnls.min()),
        })
        holding = [(t.closed_at - t.opened_at).total_seconds() / 3600 for t in closed_trades if t.opened_at]
        if holding:
            out["avg_holding_hours"] = float(np.mean(holding))
    return out
