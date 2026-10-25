"""Live paper trading on real-time crypto quotes and order books (ccxt, no API key needed).

    python examples/live_paper.py --exchange kraken --iterations 5 --interval 15

Behind a corporate proxy, export REQUESTS_CA_BUNDLE / HTTPS_PROXY first.
For US stocks use AlpacaProvider (real bid/ask; needs free API keys) or YFinanceProvider.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import SimulatorConfig  # noqa: E402
from tradesim.data.ccxt_provider import CCXTProvider  # noqa: E402
from tradesim.live import LivePaperTrader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="kraken")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--state", default="runs/live/state.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    provider = CCXTProvider(args.exchange)
    config = SimulatorConfig(initial_cash=10_000, timeframe="1h", fees=args.exchange if args.exchange in
                             ("kraken", "binance", "coinbase") else "alpaca", log_dir="runs/live")
    state = Path(args.state)
    trader = (LivePaperTrader.load_state(state, provider) if state.exists()
              else LivePaperTrader(provider, config=config))
    print(f"Resumed from {state}" if state.exists() else "New live paper account")

    def agent(t: LivePaperTrader) -> None:
        """Toy agent: hold 30% BTC / 20% ETH, with a resting ETH bid 1% under the market."""
        if t.portfolio.quantity("BTC-USD") == 0 and not t.get_open_orders("BTC-USD"):
            print("  BTC buy ->", t.place_order("BTC-USD", "buy", percent_of_equity=30)["status"])
        if not t.get_open_orders("ETH-USD") and t.portfolio.quantity("ETH-USD") == 0:
            bid = t.get_price("ETH-USD")["bid"]
            r = t.place_order("ETH-USD", "buy", percent_of_equity=20, order_type="limit", limit_price=bid * 0.99)
            print(f"  ETH limit @ {r['limit_price']} ->", r["status"])
        q = t.get_price("BTC-USD")
        pf = t.get_portfolio()
        print(f"{t.now:%H:%M:%S} BTC {q['bid']:.2f}/{q['ask']:.2f} ({q['spread_bps']:.2f} bps) "
              f"equity {pf['equity']} cash {pf['cash']} fees {pf['total_fees']}")

    trader.run(agent, interval_seconds=args.interval, max_iterations=args.iterations)
    for f in trader.get_fills():
        print(f"fill {f['side']} {f['quantity']} {f['symbol']} @ {f['price']} fee {f['fee']} ({f['liquidity']})")
    trader.save_state(state)
    print(f"State saved to {state}; run again to continue the same account.")
    trader.close()


if __name__ == "__main__":
    main()
