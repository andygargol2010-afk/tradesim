"""An LLM trading agent (Claude) driving the simulator through function calling.

At every decision point the agent gets a compact briefing (date, account state,
its own notes from the previous decision) and can call the tools in
``tradesim.agents.llm_tools`` as many times as it wants. When it answers
without a tool call, its text becomes its notes for the next decision and the
simulator advances. A fresh conversation per decision keeps token use flat
over long backtests; the notes carry the agent's memory.

    pip install anthropic
    export ANTHROPIC_API_KEY=...            # or `ant auth login`
    python examples/llm_agent.py --decisions 10 --every 5

    python examples/llm_agent.py --dry-run  # no API calls: a scripted agent exercises the same tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import SimulatorConfig, SyntheticProvider, TradingSimulator  # noqa: E402
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


def briefing(sim: TradingSimulator, notes: str) -> str:
    pf = sim.get_portfolio()
    positions = ", ".join(f"{p['symbol']} {p['quantity']} ({p['weight']:.1%}, uPnL {p['unrealized_pnl']})"
                          for p in pf["positions"]) or "none"
    return (f"Decision time: {sim.now:%Y-%m-%d %H:%M} UTC. Steps remaining in episode: {sim.steps_remaining}.\n"
            f"Equity {pf['equity']} | cash {pf['cash']} | drawdown {pf['current_drawdown_pct']:.2f}% "
            f"(max {pf['max_drawdown_pct']:.2f}%) | fees paid {pf['total_fees']}\n"
            f"Positions: {positions}\nOpen orders: {pf['open_orders']}\n"
            f"Your notes from the previous decision: {notes or '(first decision)'}")


def claude_decision(client: Any, sim: TradingSimulator, notes: str, max_tool_rounds: int = 15) -> str:
    """One decision: manual tool-use loop until Claude answers without calling a tool."""
    import anthropic

    messages: list[dict[str, Any]] = [{"role": "user", "content": briefing(sim, notes)}]
    for _ in range(max_tool_rounds):
        try:
            response = client.beta.messages.create(
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
            print("  rate limited; skipping this decision")
            return notes
        except anthropic.APIStatusError as exc:
            print(f"  API error {exc.status_code}: {exc.message}; skipping this decision")
            return notes
        if response.stop_reason == "refusal":
            print("  request declined; keeping previous notes")
            return notes
        if response.stop_reason != "tool_use":
            return "\n".join(b.text for b in response.content if b.type == "text").strip() or notes
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = dispatch(sim, block.name, block.input)
            print(f"  -> {block.name}({json.dumps(block.input)[:90]}) = {output[:110]}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})  # all results in one message
    return notes


def scripted_decision(sim: TradingSimulator, notes: str) -> str:
    """Offline stand-in for the LLM: same tools, simple trend rule."""
    calls: list[tuple[str, dict[str, Any]]] = [("get_portfolio", {})]
    for sym in ["SPY", "GLD", "BTC-USD"]:
        bars = json.loads(dispatch(sim, "get_historical_data", {"symbol": sym, "limit": 1, "indicators": True}))["bars"]
        if not bars:
            continue
        last = bars[-1]
        above = last["sma_20"] is not None and last["close"] > last["sma_20"]
        held = float(sim.portfolio.quantity(sym))
        if above and held == 0:
            calls.append(("place_order", {"symbol": sym, "side": "buy", "percent_of_equity": 25, "tag": "close > SMA20"}))
        elif not above and held > 0:
            calls.append(("close_position", {"symbol": sym}))
    for name, args in calls:
        print(f"  -> {name}({json.dumps(args)}) = {dispatch(sim, name, args)[:110]}")
    return "trend-following on SPY, GLD, BTC-USD vs SMA20"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=8)
    ap.add_argument("--every", type=int, default=5, help="bars between decisions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sim = TradingSimulator(provider=SyntheticProvider(seed=5),
                           config=SimulatorConfig(initial_cash=25_000, fees="alpaca", log_dir="runs/llm_agent"))
    sim.reset(WATCHLIST, "2024-01-01", "2024-12-31")
    client = None
    if not args.dry_run:
        import anthropic

        client = anthropic.Anthropic()

    notes = ""
    for i in range(args.decisions):
        if sim.done:
            break
        print(f"\n[{i + 1}/{args.decisions}] {sim.now:%Y-%m-%d}")
        notes = scripted_decision(sim, notes) if client is None else claude_decision(client, sim, notes)
        print(f"  notes: {notes[:300]}")
        sim.step(args.every)

    print("\nFinal portfolio:", json.dumps({k: v for k, v in sim.get_portfolio().items() if k != "positions"}, indent=1))
    m = sim.get_metrics()
    print("Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()
                       if k in ("total_return_pct", "sharpe", "max_drawdown_pct", "num_trades", "total_fees")})
    sim.close()


if __name__ == "__main__":
    main()
