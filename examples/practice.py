"""Complete practice workflow: environment -> agent -> episodes -> metrics -> charts.

    python examples/practice.py                     # synthetic data, runs offline in seconds
    python examples/practice.py --episodes 20 --show

Charts are written to runs/practice/ (``--show`` also opens them in a window).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import (  # noqa: E402
    BuyAndHoldAgent, MovingAverageCrossAgent, RandomAgent, SimpleRSIAgent, SimulatorConfig, SyntheticProvider,
    TradingSimulator, compare_agents, evaluate_agent, run_episode,
)

OUT = Path("runs/practice")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--length", type=int, default=126, help="bars per episode (126 trading days ~ 6 months)")
    ap.add_argument("--show", action="store_true", help="open the charts in a window")
    args = ap.parse_args()
    show = True if args.show else False

    # 1. Environment: 10 assets (stocks, ETFs, crypto) over 4 years of daily bars.
    #    Swap the provider for CachedProvider(YFinanceProvider()) to practise on real prices.
    env = TradingSimulator(
        provider=SyntheticProvider(seed=1),
        config=SimulatorConfig(initial_cash=10_000, timeframe="1d", fees="alpaca"),
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XOM", "GLD", "BTC-USD", "ETH-USD"],
        start="2021-01-01", end="2024-12-31",
    )

    # 2. Agent: only implements act(observation) -> action.
    agent = SimpleRSIAgent(period=14, oversold=30, overbought=70, position_pct=10)

    # 3a. One episode, rendered, then its summary, portfolio and chart.
    print("\n### One episode")
    result = run_episode(agent, env, seed=42, episode_length=args.length, render=True, render_every=21)
    env.print_summary()
    env.print_portfolio()
    print("\nLast closed trades:")
    print(env.get_trade_history("closed", as_dataframe=True).tail(5).to_string(index=False))
    env.plot_results(path=OUT / "episode.png", show=show)
    print(f"\nEpisode chart: {OUT / 'episode.png'}")

    # 3b. Many episodes at random (seeded) start dates -> statistics across episodes.
    print(f"\n### {args.episodes} episodes")
    report = evaluate_agent(agent, env, n_episodes=args.episodes, episode_length=args.length, seed=0)
    report.plot(OUT / "evaluation.png", show=show)
    print(f"Evaluation chart: {OUT / 'evaluation.png'}")
    report.episodes.to_csv(OUT / "episodes.csv", index=False)

    # 4. Compare against baselines on exactly the same episodes.
    print("\n### Comparison on identical episodes")
    table = compare_agents([agent, MovingAverageCrossAgent(), RandomAgent(seed=1), BuyAndHoldAgent()], env,
                           n_episodes=args.episodes, episode_length=args.length, seed=0)
    print(table.round(2).to_string())
    env.close()
    assert result.metrics["final_equity"] > 0


if __name__ == "__main__":
    main()
