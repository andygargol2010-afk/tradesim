"""Generate the data for the results dashboard (``dashboard/data.json``).

Runs every agent on exactly the same seeded episodes (same start dates, same
execution noise) with the regular practice loop, and exports per-episode metrics,
equity curves, the buy & hold benchmark, fills, closed trades and the prices of
the traded symbols. The dashboard itself (``dashboard/index.html``) is static and
reads this file, so it can be served by Vercel or any static host.

    python scripts/build_dashboard.py                        # synthetic data (offline)
    python scripts/build_dashboard.py --data yfinance        # real prices (Yahoo, cached)
    python scripts/build_dashboard.py --episodes 20 --length 252 --symbols SPY QQQ BTC-USD

Preview locally:  python -m http.server -d dashboard 8000
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesim  # noqa: E402
from tradesim import (  # noqa: E402
    Agent, CachedProvider, MovingAverageCrossAgent, RandomAgent, RoutingProvider, SimpleRSIAgent, SimulatorConfig,
    SyntheticProvider, TradingSimulator, run_episode,
)
from tradesim.practice import KEY_METRICS, EpisodeResult, summarize_episodes  # noqa: E402

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "GLD", "BTC-USD", "ETH-USD"]
EPISODE_METRICS = KEY_METRICS + ["cagr_pct", "volatility_annual_pct", "calmar", "num_fills", "time_in_market_pct",
                                 "total_slippage_cost", "rejected_orders", "open_positions"]
MAX_PRICE_SYMBOLS = 6  # price series exported per episode (most traded symbols)


def num(x: Any, sig: int = 6) -> Any:
    """JSON-safe number: rounded to ``sig`` significant digits, inf -> "inf", nan -> null."""
    if x is None:
        return None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(f):
        return None
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return float(f"{f:.{sig}g}")


def stamp(ts: pd.Timestamp, intraday: bool) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M" if intraday else "%Y-%m-%d")


def build_provider(kind: str):
    if kind == "synthetic":
        return SyntheticProvider(seed=1)
    from tradesim.data.yfinance_provider import YFinanceProvider

    if kind == "yfinance":
        return CachedProvider(YFinanceProvider())
    if kind == "mixed":
        from tradesim.data.ccxt_provider import CCXTProvider

        return CachedProvider(RoutingProvider([CCXTProvider("coinbase"), YFinanceProvider()]))
    raise SystemExit(f"unknown --data {kind}")


def export_episode(res: EpisodeResult, env: TradingSimulator, intraday: bool) -> dict[str, Any]:
    tf = timedelta(seconds=env.timeframe.seconds)
    eq = res.equity
    out: dict[str, Any] = {
        "episode": res.episode,
        "seed": res.seed,
        "start": str(res.metrics.get("start", ""))[:16],
        "end": str(res.metrics.get("end", ""))[:16],
        "metrics": {k: num(res.metrics.get(k)) for k in EPISODE_METRICS},
        "t": [stamp(t, intraday) for t in eq.index],
        "equity": [num(v, 8) for v in eq.to_numpy()],
        "benchmark": ([num(v, 8) for v in res.benchmark.to_numpy()] if res.benchmark is not None else None),
        "fills": [],
        "trades": [],
        "prices": {},
    }
    if len(res.fills):
        f = res.fills.sort_values("timestamp")
        out["fills"] = [{"t": stamp(row.timestamp + tf, intraday), "symbol": row.symbol, "side": row.side,
                         "quantity": row.quantity, "price": num(row.price, 8), "fee": num(row.fee, 4),
                         "liquidity": row.liquidity} for row in f.itertuples()]
        top = f["symbol"].value_counts().index[:MAX_PRICE_SYMBOLS]
        for sym in top:
            px = env.price_series(sym)
            out["prices"][sym] = {"t": [stamp(t + tf, intraday) for t in px.index],
                                  "close": [num(v, 7) for v in px.to_numpy()]}
    if len(res.trades):
        out["trades"] = [{"symbol": r.symbol, "direction": r.direction, "quantity": num(r.quantity, 8),
                          "entry_price": num(r.entry_price, 8), "exit_price": num(r.exit_price, 8),
                          "pnl": num(r.pnl, 6), "return_pct": num(r.return_pct * 100, 4),
                          "opened": stamp(r.opened_at, intraday) if pd.notna(r.opened_at) else None,
                          "closed": stamp(r.closed_at, intraday)} for r in res.trades.itertuples()]
    return out


def mean_path(curves: list[np.ndarray]) -> dict[str, list[Any]]:
    """Mean / median return % by bar offset across episodes (truncated to the shortest episode)."""
    n = min(len(c) for c in curves)
    mat = np.vstack([(c[:n] / c[0] - 1) * 100 for c in curves])
    return {"mean": [num(v, 5) for v in mat.mean(axis=0)], "median": [num(v, 5) for v in np.median(mat, axis=0)],
            "p10": [num(v, 5) for v in np.percentile(mat, 10, axis=0)],
            "p90": [num(v, 5) for v in np.percentile(mat, 90, axis=0)]}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "yfinance", "mixed"])
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--length", type=int, default=126, help="bars per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--fees", default="alpaca")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "data.json"))
    args = ap.parse_args()

    config = SimulatorConfig(initial_cash=args.cash, timeframe=args.timeframe, fees=args.fees)
    env = TradingSimulator(provider=build_provider(args.data), config=config, symbols=args.symbols,
                           start=args.start, end=args.end)
    env.reset()
    intraday = env.timeframe.seconds < 86_400
    agents: list[Agent] = [SimpleRSIAgent(), MovingAverageCrossAgent(), RandomAgent(seed=1)]

    exported: list[dict[str, Any]] = []
    bench_curves: list[np.ndarray] = []
    for agent in agents:
        results: list[EpisodeResult] = []
        episodes: list[dict[str, Any]] = []
        for i in range(args.episodes):
            res = run_episode(agent, env, seed=args.seed + i, episode_length=args.length, random_start=True, episode=i)
            results.append(res)
            episodes.append(export_episode(res, env, intraday))  # prices must be read before the next reset
            if agent is agents[0] and res.benchmark is not None:
                bench_curves.append(res.benchmark.to_numpy())
        report = summarize_episodes(str(agent), results)
        s = report.summary
        exported.append({
            "name": str(agent),
            "class": type(agent).__name__,
            "summary": {k: ({kk: num(vv) for kk, vv in v.items()} if isinstance(v, dict) else num(v))
                        for k, v in s.items()},
            "path": mean_path([r.equity.to_numpy() for r in results]),
            "episodes": episodes,
        })
        print(f"{str(agent):34s} mean return {s['total_return_pct']['mean']:+7.2f}%  "
              f"sharpe {s['sharpe']['mean']:+.2f}  beat B&H {s['pct_beat_benchmark']:.0f}%")

    ex = config.execution
    data = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tradesim_version": tradesim.__version__,
            "commit": git_commit(),
            "data_source": args.data,
            "synthetic": args.data == "synthetic",
            "symbols": env.symbols,
            "window_start": args.start,
            "window_end": args.end,
            "timeframe": env.timeframe.value,
            "initial_cash": args.cash,
            "fees": args.fees,
            "n_episodes": args.episodes,
            "episode_length": args.length,
            "seed": args.seed,
            "execution": {"slippage_model": ex.slippage_model, "participation_rate": ex.participation_rate,
                          "impact_coefficient": ex.impact_coefficient},
        },
        "benchmark_path": mean_path(bench_curves) if bench_curves else None,
        "agents": exported,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    env.close()


if __name__ == "__main__":
    main()
