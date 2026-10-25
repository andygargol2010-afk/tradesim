"""An LLM trading agent (Claude) driving the simulator through function calling.

``ClaudeAgent`` implements the standard ``Agent.act(observation)`` interface, so
``run_episode`` / ``evaluate_agent`` drive it exactly like a rule-based agent.
At every decision point it gets a compact briefing (date, account state, its own
notes from the previous decision) and can call the tools in
``tradesim.agents.llm_tools`` as many times as it wants. When it answers without
a tool call, its text becomes its notes for the next decision. A fresh
conversation per decision keeps token use flat over long backtests; the notes
carry the agent's memory.

    pip install anthropic
    export ANTHROPIC_API_KEY=...            # or `ant auth login`
    python examples/llm_agent.py --bars 40 --every 5

    python examples/llm_agent.py --dry-run  # no API calls: a scripted agent exercises the same tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator, run_episode  # noqa: E402
from tradesim.agents.base import HOLD, Agent, Observation  # noqa: E402
from tradesim.agents.llm_tools import TOOLS, dispatch  # noqa: E402

MODEL = "claude-opus-5-5"
WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD", "TLT", "BTC-USD", "ETH-USD"]

SYSTEM = f"""You manage a paper-trading account in a high-fidelity market simulator. Orders pay realistic spreads,
market impact and broker fees, fill on the next market update (never at prices you have already seen), and are
limited by available liquidity. Fractional quantities are allowed. Short selling and leverage are disabled.

Objective: maximise risk-adjusted return (Sharpe) while keeping drawdowns under 15%. Trading costs are real, so
avoid churning. Your watchlist is {", ".join(WATCHLIST)}; you may research other symbols with get_available_assets.

At each decision: inspect what you need with the tools, place or cancel orders, then reply with a short note
(under 120 words) of your current view, positions and plan. That note is all you will remember next time."""


def briefing(obs: Observation, notes: str) -> str:
    pf = obs.portfolio()
    positions = ", ".join(f"{p['symbol']} {p['quantity']} ({p['weight']:.1%}, uPnL {p['unrealized_pnl']})"
                          for p in pf["positions"]) or "none"
    return (f"Decision time: {obs.timestamp:%Y-%m-%d %H:%M} UTC. Bars remaining in episode: {obs.steps_remaining}.\n"
            f"Equity {pf['equity']} | cash {pf['cash']} | drawdown {pf['current_drawdown_pct']:.2f}% "
            f"(max {pf['max_drawdown_pct']:.2f}%) | fees paid {pf['total_fees']}\n"
            f"Positions: {positions}\nOpen orders: {pf['open_orders']}\n"
            f"Your notes from the previous decision: {notes or '(first decision)'}")


class ClaudeAgent(Agent):
    """LLM agent: every ``every`` bars, Claude inspects the market with tools and trades through them.

    ``act`` places orders itself via ``llm_tools.dispatch`` and returns ``HOLD``, so it plugs into
    ``run_episode`` / ``evaluate_agent`` like any other agent.
    """

    def __init__(self, client: Any, every: int = 5, max_tool_rounds: int = 15, verbose: bool = True):
        self.client = client
        self.every = every
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose
        self.name = f"ClaudeAgent({MODEL})"
        self.notes = ""

    def reset(self) -> None:
        self.notes = ""

    def act(self, obs: Observation) -> Any:
        if obs.step % self.every:
            return HOLD
        import anthropic

        messages: list[dict[str, Any]] = [{"role": "user", "content": briefing(obs, self.notes)}]
        for _ in range(self.max_tool_rounds):
            try:
                response = self.client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=TOOLS,
                    messages=messages,
                    output_config={"effort": "medium"},
                    cache_control={"type": "ephemeral"},  # system + tools are identical every call
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",  # re-run on a fallback model if a safety classifier declines
                )
            except anthropic.RateLimitError:
                print("  rate limited; holding this decision")
                return HOLD
            except anthropic.APIStatusError as exc:
                print(f"  API error {exc.status_code}: {exc.message}; holding this decision")
                return HOLD
            if response.stop_reason == "refusal":
                print("  request declined; holding")
                return HOLD
            if response.stop_reason != "tool_use":
                self.notes = "\n".join(b.text for b in response.content if b.type == "text").strip() or self.notes
                if self.verbose:
                    print(f"  [{obs.timestamp:%Y-%m-%d}] notes: {self.notes[:300]}")
                return HOLD
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = dispatch(obs.sim, block.name, block.input)
                if self.verbose:
                    print(f"  -> {block.name}({json.dumps(block.input)[:90]}) = {output[:110]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})  # all results in one message
        return HOLD


class ScriptedToolAgent(Agent):
    """Offline stand-in for the LLM: uses the same tools, with a simple close-vs-SMA20 rule."""

    name = "ScriptedToolAgent"

    def __init__(self, every: int = 5):
        self.every = every

    def act(self, obs: Observation) -> Any:
        if obs.step % self.every:
            return HOLD
        for sym in ["SPY", "GLD", "BTC-USD"]:
            bars = json.loads(dispatch(obs.sim, "get_historical_data",
                                       {"symbol": sym, "limit": 1, "indicators": True}))["bars"]
            if not bars or bars[-1]["sma_20"] is None:
                continue
            above = bars[-1]["close"] > bars[-1]["sma_20"]
            held = obs.position(sym) > 0
            call = None
            if above and not held:
                call = ("place_order", {"symbol": sym, "side": "buy", "percent_of_equity": 25, "tag": "close > SMA20"})
            elif not above and held:
                call = ("close_position", {"symbol": sym})
            if call:
                print(f"  [{obs.timestamp:%Y-%m-%d}] -> {call[0]}({json.dumps(call[1])}) = "
                      f"{dispatch(obs.sim, *call)[:100]}")
        return HOLD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=40, help="episode length in bars")
    ap.add_argument("--every", type=int, default=5, help="bars between decisions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = TradingSimulator(provider=SyntheticProvider(seed=5),
                           config=SimulatorConfig(initial_cash=25_000, fees="alpaca", log_dir="runs/llm_agent"),
                           symbols=WATCHLIST, start="2024-01-01", end="2024-12-31")
    if args.dry_run:
        agent: Agent = ScriptedToolAgent(args.every)
    else:
        import anthropic

        agent = ClaudeAgent(anthropic.Anthropic(), every=args.every)

    result = run_episode(agent, env, seed=0, episode_length=args.bars)
    result.print_summary()
    env.print_portfolio()
    env.plot_results(path="runs/llm_agent/results.png")
    print("Chart saved to runs/llm_agent/results.png")
    env.close()


if __name__ == "__main__":
    main()
