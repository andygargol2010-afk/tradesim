"""Human-readable summaries and charts (matplotlib, imported lazily: ``pip install matplotlib``)."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from tradesim.practice import EvaluationReport
    from tradesim.simulator import BaseSimulator

_W = 62


def _num(v: Any, fmt: str = "{:,.2f}", na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "inf" if isinstance(v, float) and v == math.inf else na
    return fmt.format(v)


def _money(v: Any) -> str:
    return _num(v, "${:,.2f}")


def _pct(v: Any, signed: bool = True) -> str:
    return _num(v, "{:+.2f} %" if signed else "{:.2f} %")


def format_summary(m: dict[str, Any], title: str = "Episode summary") -> str:
    """Format a ``get_metrics()`` dict as an aligned text block."""
    if not m:
        return f"{title}: no data yet"
    rows: list[tuple[str, str]] = []
    if m.get("start"):
        rows.append(("Period", f"{m['start'][:10]} -> {m['end'][:10]}  ({m.get('bars', 0)} bars)"))
    rows += [
        ("Initial equity", _money(m.get("start_equity"))),
        ("Final equity", _money(m.get("final_equity"))),
        ("Total return", _pct(m.get("total_return_pct"))
         + (f"   (buy & hold {_pct(m['benchmark_return_pct'])})" if m.get("benchmark_return_pct") is not None else "")),
        ("CAGR", _pct(m.get("cagr_pct"))),
        ("Max drawdown", _pct(-(m.get("max_drawdown_pct") or 0.0))),
        ("Volatility (annual)", _pct(m.get("volatility_annual_pct"), signed=False)),
        ("Sharpe ratio", _num(m.get("sharpe"), "{:.3f}")),
        ("Sortino ratio", _num(m.get("sortino"), "{:.3f}")),
        ("Calmar ratio", _num(m.get("calmar"), "{:.3f}")),
        ("", ""),
        ("Closed trades", f"{m.get('num_trades', 0)}   (fills: {m.get('num_fills', 'n/a')}, "
                          f"open positions: {m.get('open_positions', 'n/a')})"),
        ("Win rate", _pct(m.get("win_rate_pct"), signed=False)),
        ("Profit factor", _num(m.get("profit_factor"), "{:.3f}")),
        ("Average trade PnL", _money(m.get("avg_trade_pnl"))),
        ("Best trade", _money(m.get("best_trade"))),
        ("Worst trade", _money(m.get("worst_trade"))),
        ("", ""),
        ("Fees paid", _money(m.get("total_fees"))),
        ("Slippage cost", _money(m.get("total_slippage_cost"))),
        ("Time in market", _pct(m.get("time_in_market_pct"), signed=False)),
    ]
    line = "=" * _W
    body = "\n".join(f" {k:<22}{v}" if k else "" for k, v in rows)
    return f"{line}\n {title}\n{line}\n{body}\n{line}"


def format_portfolio(pf: dict[str, Any]) -> str:
    """Format a ``get_portfolio()`` dict: cash, equity and a table of open positions."""
    head = (f"Portfolio @ {pf.get('timestamp', '')[:16]}  equity ${float(pf['equity']):,.2f}  "
            f"cash ${float(pf['cash']):,.2f}  unrealized ${float(pf['unrealized_pnl']):,.2f}  "
            f"realized ${float(pf['realized_pnl']):,.2f}")
    if not pf["positions"]:
        return head + "\n  (no open positions)"
    lines = [head, f"  {'symbol':<10}{'quantity':>20}{'avg price':>14}{'last':>14}{'value':>13}{'uPnL':>12}"
                   f"{'uPnL %':>9}{'weight':>8}"]
    for p in pf["positions"]:
        lines.append(f"  {p['symbol']:<10}{p['quantity']:>20}{float(p['avg_price']):>14,.6g}"
                     f"{float(p['mark_price']):>14,.6g}{float(p['market_value']):>13,.2f}"
                     f"{float(p['unrealized_pnl']):>12,.2f}{p['unrealized_pnl_pct'] * 100:>8.2f}%"
                     f"{(p.get('weight') or 0) * 100:>7.1f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------- charts
def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from exc
    return plt


def _finish(fig, path: str | Path | None, show: bool | None):
    plt = _plt()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if show is None:
        import matplotlib

        show = path is None and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")
    if show:
        plt.show()
    return fig


def equity_frame(sim: "BaseSimulator") -> pd.DataFrame:
    """Equity curve of the current episode as a DataFrame (equity, cash, exposure, drawdown %)."""
    curve = sim.portfolio.equity_curve
    df = pd.DataFrame({
        "equity": [float(p.equity) for p in curve],
        "cash": [float(p.cash) for p in curve],
        "gross_exposure": [float(p.gross_exposure) for p in curve],
    }, index=pd.DatetimeIndex([p.timestamp for p in curve], name="timestamp"))
    df["drawdown_pct"] = (df["equity"] / df["equity"].cummax() - 1) * 100
    return df


def benchmark_curve(sim: "BaseSimulator", index: pd.DatetimeIndex) -> pd.Series | None:
    """Equal-weight buy & hold of the episode's symbols (no costs), aligned to ``index``."""
    if not hasattr(sim, "price_series") or not getattr(sim, "symbols", None):
        return None
    tf = timedelta(seconds=sim.timeframe.seconds)
    rel = []
    for sym in sim.symbols:
        px = sim.price_series(sym)
        if len(px) < 2:
            continue
        px.index = px.index + tf  # bar close time, as the equity curve
        aligned = px.reindex(px.index.union(index)).ffill().reindex(index)
        first = aligned.first_valid_index()
        if first is None:
            continue
        rel.append(aligned / aligned.loc[first])
    if not rel:
        return None
    start = float(sim.portfolio.initial_cash)
    return pd.concat(rel, axis=1).mean(axis=1).fillna(1.0) * start


def plot_results(sim: "BaseSimulator", symbol: str | None = None, path: str | Path | None = None,
                 show: bool | None = None, benchmark: bool = True):
    """Equity curve (+ buy & hold benchmark and trade markers), drawdown, and price with buys/sells.

    Args:
        sim: a simulator after (or during) an episode.
        symbol: symbol for the price panel; default = the most traded one (panel omitted if no trades).
        path: save the figure here (PNG, SVG, PDF...).
        show: open a window; default only if no ``path`` and the backend is interactive.
        benchmark: overlay equal-weight buy & hold of the episode's symbols.
    """
    plt = _plt()
    eq = equity_frame(sim)
    if len(eq) < 2:
        raise ValueError("Nothing to plot yet: run at least one step")
    fills = sim.get_trade_history("fills", as_dataframe=True)
    if symbol is None and len(fills):
        symbol = fills["symbol"].value_counts().idxmax()
    has_price = symbol is not None and hasattr(sim, "price_series")
    n = 3 if has_price else 2
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n + 1), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3] + ([2.2] if has_price else [])})
    ax_eq, ax_dd = axes[0], axes[1]
    m = sim.get_metrics()

    ax_eq.plot(eq.index, eq["equity"], color="#1f5fbf", lw=1.6, label="Agent equity")
    if benchmark:
        bench = benchmark_curve(sim, eq.index)
        if bench is not None:
            ax_eq.plot(bench.index, bench.values, color="#888888", lw=1.1, ls="--", label="Buy & hold (equal weight)")
    ax_eq.axhline(float(sim.portfolio.initial_cash), color="#bbbbbb", lw=0.8)
    if len(fills):
        tf = timedelta(seconds=sim.timeframe.seconds)
        at = pd.DatetimeIndex(fills["timestamp"]) + tf  # executed during the bar -> equity point at its close
        pos = np.clip(eq.index.searchsorted(at, side="right") - 1, 0, len(eq) - 1)  # several fills per bar OK
        y = eq["equity"].to_numpy()[pos]
        buys = (fills["side"] == "buy").to_numpy()
        ax_eq.scatter(at[buys], y[buys], marker="^", color="#1a9e3f", s=28, zorder=3, label="Buy fill")
        ax_eq.scatter(at[~buys], y[~buys], marker="v", color="#d62728", s=28, zorder=3, label="Sell fill")
    ax_eq.set_ylabel("Equity (USD)")
    ax_eq.legend(loc="upper left", fontsize=8, ncol=4)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"Return {_pct(m.get('total_return_pct'))} | Sharpe {_num(m.get('sharpe'), '{:.2f}')} | "
        f"Sortino {_num(m.get('sortino'), '{:.2f}')} | Max DD {_pct(-(m.get('max_drawdown_pct') or 0))} | "
        f"Trades {m.get('num_trades', 0)} | Win rate {_num(m.get('win_rate_pct'), '{:.1f}%')}", fontsize=10)

    ax_dd.fill_between(eq.index, eq["drawdown_pct"], 0, color="#d62728", alpha=0.35, lw=0)
    ax_dd.plot(eq.index, eq["drawdown_pct"], color="#d62728", lw=0.8)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.grid(alpha=0.25)

    if has_price:
        ax_px = axes[2]
        px = sim.price_series(symbol)
        px.index = px.index + timedelta(seconds=sim.timeframe.seconds)
        ax_px.plot(px.index, px.values, color="#444444", lw=1.0, label=f"{symbol} close")
        f = fills[fills["symbol"] == symbol]
        if len(f):
            t = pd.DatetimeIndex(f["timestamp"]) + timedelta(seconds=sim.timeframe.seconds)
            b = (f["side"] == "buy").to_numpy()
            ax_px.scatter(t[b], f["price"].to_numpy()[b], marker="^", color="#1a9e3f", s=40, zorder=3, label="Buy")
            ax_px.scatter(t[~b], f["price"].to_numpy()[~b], marker="v", color="#d62728", s=40, zorder=3, label="Sell")
        ax_px.set_ylabel(f"{symbol} price")
        ax_px.legend(loc="upper left", fontsize=8, ncol=3)
        ax_px.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_evaluation(report: "EvaluationReport", path: str | Path | None = None, show: bool | None = None):
    """All episodes' equity curves (normalised), the distribution of returns and per-episode Sharpe."""
    plt = _plt()
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
    ax_curves = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    df = report.episodes

    best, worst = report.summary["best_episode"], report.summary["worst_episode"]
    for ep in report.results:
        curve = ep.equity / ep.equity.iloc[0] * 100 - 100
        x = np.arange(len(curve))
        if ep.episode in (best, worst):
            color = "#1a9e3f" if ep.episode == best else "#d62728"
            ax_curves.plot(x, curve.values, color=color, lw=1.8,
                           label=f"{'best' if ep.episode == best else 'worst'} (ep {ep.episode})")
        else:
            ax_curves.plot(x, curve.values, color="#1f5fbf", lw=0.8, alpha=0.35)
    ax_curves.axhline(0, color="#999999", lw=0.8)
    ax_curves.set_xlabel("bars into episode")
    ax_curves.set_ylabel("return %")
    ax_curves.set_title(f"{report.agent_name}: {len(df)} episodes | mean return "
                        f"{_pct(report.summary['total_return_pct']['mean'])} | median "
                        f"{_pct(report.summary['total_return_pct']['median'])} | mean Sharpe "
                        f"{_num(report.summary['sharpe']['mean'], '{:.2f}')} | positive episodes "
                        f"{report.summary['pct_positive_episodes']:.0f}%", fontsize=10)
    ax_curves.legend(fontsize=8)
    ax_curves.grid(alpha=0.25)

    agent_rets = df["total_return_pct"].astype(float).dropna()
    bench_rets = (df["benchmark_return_pct"].astype(float).dropna() if "benchmark_return_pct" in df
                  else pd.Series(dtype=float))
    bins = np.histogram_bin_edges(pd.concat([agent_rets, bench_rets]), bins=min(20, max(5, len(df))))
    ax_hist.hist(agent_rets, bins=bins, color="#1f5fbf", alpha=0.75, label="agent")
    if len(bench_rets):
        ax_hist.hist(bench_rets, bins=bins, color="#888888", alpha=0.45, label="buy & hold")
    ax_hist.axvline(0, color="#333333", lw=0.8)
    ax_hist.set_xlabel("episode return %")
    ax_hist.set_ylabel("episodes")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.25)

    sharpe = df["sharpe"].fillna(0.0)
    ax_bar.bar(df["episode"], sharpe, color=np.where(sharpe >= 0, "#1a9e3f", "#d62728"))
    ax_bar.set_xlabel("episode")
    ax_bar.set_ylabel("Sharpe")
    ax_bar.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return _finish(fig, path, show)
