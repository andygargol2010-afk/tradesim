"""Practice / evaluation loops: run an agent for one or many episodes and summarise the results.

Works with both environment flavours:

* :class:`~tradesim.simulator.TradingSimulator` — the agent gets an :class:`~tradesim.agents.base.Observation`
  and returns orders / target weights (rule-based and LLM agents);
* :class:`~tradesim.env.PaperTradingEnv` — the agent gets the Gymnasium observation vector and returns an
  element of ``env.action_space`` (reinforcement learning).

Example::

    env = TradingSimulator(provider=SyntheticProvider(seed=1), symbols=["SPY", "AAPL", "BTC-USD"],
                           start="2021-01-01", end="2024-12-31")
    report = evaluate_agent(SimpleRSIAgent(), env, n_episodes=10, episode_length=126)
    report.plot("runs/eval.png")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tradesim.agents.base import Agent
from tradesim.reporting import _money, _num, _pct, benchmark_curve, equity_frame, format_summary
from tradesim.simulator import BaseSimulator, TradingSimulator

#: Metrics aggregated across episodes by :func:`evaluate_agent`.
KEY_METRICS = [
    "final_equity", "total_return_pct", "benchmark_return_pct", "excess_return_pct", "max_drawdown_pct",
    "sharpe", "sortino", "win_rate_pct", "profit_factor", "num_trades", "avg_trade_pnl", "best_trade",
    "worst_trade", "total_fees",
]


@dataclass
class EpisodeResult:
    """Outcome of one episode.

    Attributes:
        metrics: everything from ``get_metrics()`` plus ``benchmark_return_pct`` (equal-weight buy & hold
            of the episode's symbols, no costs), ``excess_return_pct``, ``total_reward`` (Gym), ``steps``,
            ``rejected_orders``, ``seed`` and ``wall_time_s``.
        equity: equity curve (indexed by time).
        fills: every execution of the episode (DataFrame).
        trades: closed round trips of the episode (DataFrame).
        benchmark: equal-weight buy & hold equity on the same timestamps (``None`` if unavailable).
    """

    episode: int
    seed: int | None
    metrics: dict[str, Any]
    equity: pd.Series
    fills: pd.DataFrame
    trades: pd.DataFrame
    benchmark: pd.Series | None = None

    def summary(self) -> str:
        return format_summary(self.metrics, title=f"Episode {self.episode} (seed {self.seed})")

    def print_summary(self) -> None:
        print(self.summary())


@dataclass
class EvaluationReport:
    """Results of :func:`evaluate_agent`: per-episode table + statistics across episodes."""

    agent_name: str
    results: list[EpisodeResult]
    episodes: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:  # report["sharpe"]["mean"]
        return self.summary[key]

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent_name, "summary": self.summary,
                "episodes": self.episodes.to_dict(orient="records")}

    def __str__(self) -> str:
        s = self.summary
        n = len(self.episodes)
        line = "=" * 90
        rows = [line, f" Evaluation of {self.agent_name}: {n} episodes", line,
                f" {'metric':<24}{'n':>4}{'mean':>12}{'median':>12}{'std':>11}{'worst':>13}{'best':>13}"]
        fmts = {"final_equity": "{:,.2f}", "num_trades": "{:.1f}", "sharpe": "{:.3f}", "sortino": "{:.3f}",
                "profit_factor": "{:.3f}"}
        for k in KEY_METRICS:
            st = s.get(k)
            if not st or st["count"] == 0:
                continue
            f = fmts.get(k, "{:,.2f}")
            # "worst" / "best" from the agent's point of view (drawdown: higher is worse)
            lo, hi = (st["max"], st["min"]) if k == "max_drawdown_pct" else (st["min"], st["max"])
            rows.append(f" {k:<24}{st['count']:>4}{_num(st['mean'], f):>12}{_num(st['median'], f):>12}{_num(st['std'], f):>11}"
                        f"{_num(lo, f):>13}{_num(hi, f):>13}")
        rows += [line,
                 " n = episodes with a finite value (no closed trade -> no win rate; no losing trade -> profit factor inf)",
                 f" Best episode:  #{s['best_episode']}  return {_pct(s['best_return_pct'])}   "
                 f"Worst episode: #{s['worst_episode']}  return {_pct(s['worst_return_pct'])}",
                 f" Positive episodes: {s['pct_positive_episodes']:.0f}%   "
                 + (f"Beat buy & hold: {s['pct_beat_benchmark']:.0f}%   " if s.get("pct_beat_benchmark") is not None else "")
                 + f"Mean final equity: {_money(s['final_equity']['mean'])}",
                 line]
        return "\n".join(rows)

    def print(self) -> None:
        print(str(self))

    def plot(self, path: str | Path | None = None, show: bool | None = None):
        """Equity curves of all episodes, distribution of returns and per-episode Sharpe."""
        from tradesim.reporting import plot_evaluation

        return plot_evaluation(self, path=path, show=show)


# ------------------------------------------------------------------------------ helpers
def _is_simulator(env: Any) -> bool:
    return isinstance(env, BaseSimulator)


def _episode_sim(env: Any) -> BaseSimulator | None:
    return env if _is_simulator(env) else getattr(env, "sim", None)


def _collect(episode: int, seed: int | None, sim: BaseSimulator | None, extra: dict[str, Any]) -> EpisodeResult:
    if sim is None:  # foreign Gym env: only what the loop itself measured
        return EpisodeResult(episode, seed, dict(extra), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame())
    metrics = sim.get_metrics()
    eq = equity_frame(sim)
    bench = benchmark_curve(sim, eq.index) if len(eq) else None
    if bench is not None and len(bench) and bench.iloc[0] > 0:
        metrics["benchmark_return_pct"] = float(bench.iloc[-1] / bench.iloc[0] * 100 - 100)
        metrics["excess_return_pct"] = metrics.get("total_return_pct", 0.0) - metrics["benchmark_return_pct"]
    metrics.update(extra)
    return EpisodeResult(episode, seed, metrics, eq["equity"],
                         sim.get_trade_history("fills", as_dataframe=True),
                         sim.get_trade_history("closed", as_dataframe=True), bench)


def _render_line(sim: BaseSimulator, step: int, action_results: list[dict[str, Any]], fills: list[dict[str, Any]]) -> str:
    pf = sim.portfolio
    acts = ", ".join(f"{r['side']} {r['quantity']} {r['symbol']}" + (" [REJECTED]" if r["status"] == "rejected" else "")
                     for r in action_results[:4])
    if len(action_results) > 4:
        acts += f" (+{len(action_results) - 4})"
    return (f"{sim.now:%Y-%m-%d %H:%M} step {step:>4} | equity {float(pf.equity):>12,.2f} | cash "
            f"{float(pf.cash):>11,.2f} | pos {len(pf.open_positions()):>2} | fills {len(fills):>2}"
            + (f" | orders: {acts}" if acts else ""))


# ------------------------------------------------------------------------------- loops
def run_episode(agent: Agent, env: Any, render: bool = False, seed: int | None = None,
                episode_length: int | None = None, random_start: bool | None = None,
                max_steps: int | None = None, render_every: int = 1, reset: bool = True,
                episode: int = 0) -> EpisodeResult:
    """Run one episode: ``observe -> agent.act -> execute -> step`` until the episode ends.

    Args:
        agent: anything with ``act(observation) -> action`` (``reset()`` is called if present).
        env: a ``TradingSimulator`` (``LivePaperTrader`` is not episodic) or a ``PaperTradingEnv``.
        render: print one line per step (``render_every`` to thin it out).
        seed: episode seed (start date if random, execution noise, scenario draws).
        episode_length: bars per episode (``TradingSimulator``; default: until the end of the window).
        random_start: draw the start bar at random (default: ``True`` when ``episode_length`` is given).
        max_steps: hard cap on steps.
        reset: reset the ``TradingSimulator`` first (``False`` continues an episode you set up yourself).
            Gymnasium environments are always reset.

    Returns:
        :class:`EpisodeResult` with metrics, equity curve, fills and closed trades.
    """
    t0 = time.perf_counter()
    if hasattr(agent, "reset"):
        agent.reset()
    steps = rejected = 0
    total_reward = 0.0

    if _is_simulator(env):
        sim: TradingSimulator = env
        if reset:
            rs = (episode_length is not None) if random_start is None else random_start
            sim.reset(seed=seed, episode_length=episode_length, random_start=rs)
        while not sim.done and (max_steps is None or steps < max_steps):
            obs = sim.observe()
            results = sim.apply_action(agent.act(obs))
            rejected += sum(1 for r in results if r.get("status") == "rejected")
            fills = sim.step()
            steps += 1
            if render and steps % render_every == 0:
                print(_render_line(sim, steps, results, fills))
        return _collect(episode, seed, sim, {"steps": steps, "rejected_orders": rejected, "seed": seed,
                                             "wall_time_s": time.perf_counter() - t0})

    # Gymnasium-style environment (RL); episodes are configured on the env itself.
    obs, info = env.reset(seed=seed)
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        total_reward += float(reward)
        rejected += len(info.get("rejected_orders", []) or [])
        steps += 1
        if render and steps % render_every == 0:
            print(env.render() if hasattr(env, "render") else f"step {steps} reward {reward:.5f}")
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return _collect(episode, seed, _episode_sim(env), {"steps": steps, "rejected_orders": rejected, "seed": seed,
                                                       "total_reward": total_reward,
                                                       "wall_time_s": time.perf_counter() - t0})


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.array([v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)],
                   dtype=float)
    if not len(arr):
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {"count": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "min": float(arr.min()), "max": float(arr.max())}


def summarize_episodes(agent_name: str, results: list[EpisodeResult]) -> EvaluationReport:
    """Build an :class:`EvaluationReport` from episode results."""
    rows = [{"episode": r.episode, "seed": r.seed, "start": r.metrics.get("start"), "end": r.metrics.get("end"),
             "steps": r.metrics.get("steps"), **{k: r.metrics.get(k) for k in KEY_METRICS},
             "rejected_orders": r.metrics.get("rejected_orders"), "total_reward": r.metrics.get("total_reward")}
            for r in results]
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {k: _stats(df[k].tolist()) for k in KEY_METRICS if k in df}
    rets = df["total_return_pct"].astype(float)
    best_i, worst_i = int(rets.idxmax()), int(rets.idxmin())
    summary.update({
        "n_episodes": len(df),
        "best_episode": int(df.loc[best_i, "episode"]), "best_return_pct": float(rets[best_i]),
        "worst_episode": int(df.loc[worst_i, "episode"]), "worst_return_pct": float(rets[worst_i]),
        "pct_positive_episodes": float((rets > 0).mean() * 100),
        "pct_beat_benchmark": (float((df["excess_return_pct"].astype(float) > 0).mean() * 100)
                               if df["excess_return_pct"].notna().any() else None),
    })
    return EvaluationReport(agent_name, results, df, summary)


def evaluate_agent(agent: Agent, env: Any, n_episodes: int = 20, episode_length: int | None = None,
                   seed: int = 0, random_start: bool = True, render: bool = False,
                   verbose: bool = True) -> EvaluationReport:
    """Run ``n_episodes`` seeded episodes and summarise them (mean, median, std, best, worst...).

    With a ``TradingSimulator``, each episode starts at a random (seeded) bar of the data window so
    the agent is tested on different market periods; ``episode_length`` defaults to half the window.
    Episode ``i`` uses seed ``seed + i``, so two agents evaluated with the same arguments face exactly
    the same episodes (see :func:`compare_agents`).

    Returns:
        :class:`EvaluationReport`: ``report.episodes`` (one row per episode), ``report.summary``
        (statistics), ``print(report)`` for a table and ``report.plot()`` for charts.
    """
    if _is_simulator(env):
        if getattr(env, "data", None) is None or not len(getattr(env, "_calendar", [])):
            env.reset()  # load data once to know the window length
        if episode_length is None and random_start:
            episode_length = max((env.num_bars - 1) // 2, 1)
    name = str(agent)
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        res = run_episode(agent, env, render=render, seed=seed + i, episode_length=episode_length,
                          random_start=random_start if _is_simulator(env) else None, episode=i)
        results.append(res)
        if verbose:
            m = res.metrics
            print(f"[{name}] episode {i + 1:>3}/{n_episodes}  {str(m.get('start', ''))[:10]} -> "
                  f"{str(m.get('end', ''))[:10]}  return {_pct(m.get('total_return_pct')):>10}  "
                  f"buy&hold {_pct(m.get('benchmark_return_pct')):>10}  sharpe {_num(m.get('sharpe'), '{:6.2f}')}  "
                  f"maxDD {_num(m.get('max_drawdown_pct'), '{:5.2f}%')}  trades {m.get('num_trades', 0):>3}")
    report = summarize_episodes(name, results)
    if verbose:
        print(report)
    return report


def compare_agents(agents: Sequence[Agent], env: Any, n_episodes: int = 20, episode_length: int | None = None,
                   seed: int = 0, verbose: bool = False) -> pd.DataFrame:
    """Evaluate several agents on identical episodes; one row per agent (mean / median of key metrics)."""
    rows = []
    for agent in agents:
        rep = evaluate_agent(agent, env, n_episodes, episode_length, seed, verbose=verbose)
        s = rep.summary
        rows.append({"agent": rep.agent_name,
                     "mean_return_%": s["total_return_pct"]["mean"], "median_return_%": s["total_return_pct"]["median"],
                     "mean_excess_vs_bh_%": (s.get("excess_return_pct") or {}).get("mean"),
                     "mean_sharpe": s["sharpe"]["mean"], "mean_max_dd_%": s["max_drawdown_pct"]["mean"],
                     "mean_win_rate_%": s["win_rate_pct"]["mean"], "mean_trades": s["num_trades"]["mean"],
                     "mean_fees": s["total_fees"]["mean"], "positive_episodes_%": s["pct_positive_episodes"]})
    return pd.DataFrame(rows).set_index("agent")
