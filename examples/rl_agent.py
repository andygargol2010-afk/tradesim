"""Train / evaluate agents in the Gymnasium environment.

Runs three baselines on identical, seeded episodes (random, buy-and-hold,
momentum), then - if ``stable-baselines3`` is installed - trains PPO and
evaluates it on held-out episodes from a later period (out-of-sample).

    python examples/rl_agent.py                     # baselines only (no extra deps)
    pip install stable-baselines3 && python examples/rl_agent.py --ppo-steps 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import SimulatorConfig, SyntheticProvider  # noqa: E402
from tradesim.env import PaperTradingEnv  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD"]
Policy = Callable[[np.ndarray, PaperTradingEnv], np.ndarray]


def random_policy(obs, env):
    return env.action_space.sample()


def buy_and_hold(obs, env):
    return np.full(len(env.symbols), 1.0 / len(env.symbols), dtype=np.float32)


def momentum(obs, env):
    """Equal-weight the assets whose last-`window` cumulative return is positive (reads the observation)."""
    n, w = len(env.symbols), env.window
    bars = obs[: n * w * 4].reshape(n, w, 4)
    trend = bars[:, :, 0].sum(axis=1)
    mask = trend > 0
    out = np.zeros(n, dtype=np.float32)
    if mask.any():
        out[mask] = 1.0 / mask.sum()
    return out


def evaluate(env: PaperTradingEnv, policy: Policy, seeds: list[int]) -> dict[str, float]:
    rets, dds, fees = [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs, env))
            done = term or trunc
        rets.append(info["equity"] / float(env.sim.portfolio.initial_cash) - 1)
        dds.append(info["max_drawdown"])
        fees.append(info["total_fees"])
    return {"mean_return_%": 100 * np.mean(rets), "std_return_%": 100 * np.std(rets),
            "mean_max_dd_%": 100 * np.mean(dds), "mean_fees_$": float(np.mean(fees))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ppo-steps", type=int, default=0, help="train PPO for this many steps (needs stable-baselines3)")
    args = ap.parse_args()

    provider = SyntheticProvider(seed=11)  # swap for CachedProvider(YFinanceProvider()) with real data
    config = SimulatorConfig(initial_cash=100_000, fees="alpaca", seed=0)
    common = dict(provider=provider, config=config, episode_length=126, window=30, reward="log_return",
                  drawdown_penalty=0.5)
    train_env = PaperTradingEnv(SYMBOLS, "2018-01-01", "2022-12-31", **common)
    test_env = PaperTradingEnv(SYMBOLS, "2023-01-01", "2024-12-31",
                               **{**common, "config": SimulatorConfig(initial_cash=100_000, fees="alpaca")})
    print(f"obs dim {train_env.observation_space.shape[0]}, action {train_env.action_space}")

    seeds = list(range(1000, 1000 + args.episodes))
    print(f"\nOut-of-sample evaluation, {args.episodes} episodes x {common['episode_length']} days:")
    for name, pol in [("random", random_policy), ("buy_and_hold", buy_and_hold), ("momentum", momentum)]:
        res = evaluate(test_env, pol, seeds)
        print(f"  {name:13s} " + "  ".join(f"{k} {v:9.2f}" for k, v in res.items()))

    if args.ppo_steps:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise SystemExit("pip install stable-baselines3 to train PPO")
        model = PPO("MlpPolicy", train_env, verbose=1, seed=0, n_steps=1024, batch_size=256)
        model.learn(total_timesteps=args.ppo_steps)
        res = evaluate(test_env, lambda o, e: model.predict(o, deterministic=True)[0], seeds)
        print(f"  {'ppo':13s} " + "  ".join(f"{k} {v:9.2f}" for k, v in res.items()))
        model.save("runs/ppo_tradesim")


if __name__ == "__main__":
    main()
