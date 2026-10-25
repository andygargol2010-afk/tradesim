"""End-to-end demo: 1000+ assets, fractional orders, all order types, portfolio & metrics.

    python examples/demo.py                       # offline, synthetic market data
    python examples/demo.py --data yfinance       # real Yahoo data (stocks + crypto)
    python examples/demo.py --data mixed          # crypto from Coinbase via ccxt, the rest from Yahoo
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradesim import (  # noqa: E402
    AssetUniverse, CachedProvider, RoutingProvider, SimulatorConfig, SyntheticProvider, TradingSimulator,
)

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT", "NVDA", "SPY", "QQQ", "GLD"]


def build_provider(kind: str):
    if kind == "synthetic":
        return SyntheticProvider(seed=42)
    if kind == "yfinance":
        from tradesim.data.yfinance_provider import YFinanceProvider

        return CachedProvider(YFinanceProvider())
    if kind == "mixed":
        from tradesim.data.ccxt_provider import CCXTProvider
        from tradesim.data.yfinance_provider import YFinanceProvider

        return CachedProvider(RoutingProvider([CCXTProvider("coinbase"), YFinanceProvider()]))
    raise SystemExit(f"unknown --data {kind}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show_order(r: dict) -> None:
    extra = f" -> {r['reject_reason']}" if r["status"] == "rejected" else ""
    px = r["limit_price"] or r["stop_price"] or "mkt"
    print(f"  {r['id']} {r['side']:4s} {r['quantity']:>16s} {r['symbol']:8s} {r['order_type']:10s} @ {px:<10} "
          f"[{r['status']}]{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "yfinance", "mixed"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--log-dir", default="runs/demo")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # ------------------------------------------------------------------ universe
    section("1. Asset universe")
    universe = AssetUniverse.default()
    print(f"Loaded {len(universe)} assets: {universe.counts()}")
    print("Most liquid crypto:", [a.symbol for a in universe.filter("crypto", max_liquidity_tier=2)][:10])
    print("Nasdaq-100 sample: ", [a.symbol for a in universe.filter(group="nasdaq100")][:10])

    # ---------------------------------------------------------------- simulator
    config = SimulatorConfig(initial_cash=50_000, timeframe="1d", fees="alpaca", seed=7, log_dir=args.log_dir)
    sim = TradingSimulator(universe, build_provider(args.data), config)
    long_tail = [a.symbol for a in universe.filter("crypto") if a.liquidity_tier == 5][:1]
    small_cap = [a.symbol for a in universe.filter(group="sp600")][:1]
    sim.reset(SYMBOLS + long_tail + small_cap, args.start, args.end)
    print(f"\nEpisode: {len(sim.symbols)} symbols, {sim.num_bars} bars, starting {sim.now:%Y-%m-%d}, "
          f"data source: {args.data}")

    section("2. Quotes (bid/ask modelled from liquidity tier + volatility when the feed has none)")
    for sym in sim.symbols:
        q = sim.get_price(sym)
        print(f"  {sym:8s} bid {q['bid']:>12.6g}  ask {q['ask']:>12.6g}  spread {q['spread_bps']:7.2f} bps")

    # ---------------------------------------------------------- fractional buys
    section("3. Fractional orders")
    btc = sim.get_price("BTC-USD")["ask"]
    btc_qty = 0.003 if 0.003 * btc >= 1 else round(5 / btc, 8)  # synthetic BTC can be priced low
    for r in [
        sim.place_order("BTC-USD", "buy", btc_qty, tag="fractional BTC"),
        sim.place_order("AAPL", "buy", 1.75, tag="1.75 shares"),
        sim.place_order("ETH-USD", "buy", notional=2_500, tag="$2,500 of ETH"),
        sim.place_order("SPY", "buy", percent_of_equity=20, tag="20% of equity"),
        sim.place_order("NVDA", "buy", 3.3333, order_type="limit", limit_price=sim.get_price("NVDA")["bid"] * 0.98),
        sim.place_order("QQQ", "buy", notional=10_000, order_type="stop", stop_price=sim.get_price("QQQ")["ask"] * 1.01),
        sim.place_order("BTC-USD", "buy", 0.0000001),  # below the $1 minimum -> rejected
        sim.place_order("MSFT", "sell", 1),  # nothing to sell, shorting disabled -> rejected
        sim.place_order("GLD", "buy", notional=1_000_000),  # more than the account can afford -> rejected
    ]:
        show_order(r)

    sim.step()
    print("\nAfter one bar:")
    for f in sim.get_fills():
        print(f"  fill {f['side']:4s} {f['quantity']:>14s} {f['symbol']:8s} @ {f['price']:<12s} fee {f['fee']:<12s} "
              f"{f['liquidity']}")

    # ------------------------------------------------------ liquidity demo
    section("4. Liquidity: a large order in an illiquid asset fills partially over several bars")
    if long_tail:
        # Separate hourly simulation: at most 10% of each bar's volume can be filled, with
        # square-root market impact, so a big order in a long-tail token takes several bars.
        sym = long_tail[0]
        side_sim = TradingSimulator(universe, sim.provider, SimulatorConfig(initial_cash=100_000, timeframe="1h"))
        side_sim.reset([sym], args.start, "2024-01-10")
        r = side_sim.place_order(sym, "buy", notional=15_000, tag="illiquid")
        show_order(r)
        for _ in range(6):
            side_sim.step()
            o = side_sim.get_order(r["id"])
            print(f"  {side_sim.now:%Y-%m-%d %H:%M} filled {o['filled_quantity']:>14s} / {o['quantity']}  "
                  f"avg {o['avg_fill_price']}  status {o['status']}")
            if o["status"] == "filled":
                break
        mid = side_sim.get_price(sym)["mid"]
        print(f"  current mid {mid:.6g}: the average price paid includes spread + impact of walking the book")
        side_sim.close()

    # ------------------------------------------------- a simple strategy loop
    section("5. Run a momentum rebalance every 5 bars with stop-losses")

    def agent(s: TradingSimulator) -> None:
        if s.step_index % 5:
            return
        scores = {}
        for sym in ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD", "SOL-USD"]:
            if sym not in s.symbols:
                continue
            hist = s.get_historical_data(sym, limit=21)
            if len(hist) == 21:
                scores[sym] = hist["close"].iloc[-1] / hist["close"].iloc[0] - 1
        top = sorted(scores, key=scores.get, reverse=True)[:4]
        s.target_weights({sym: 0.22 for sym in top})
        for sym in top:  # protective stop 8% below the market
            stop = s.get_price(sym)["bid"] * 0.92
            held = s.portfolio.quantity(sym)
            if held > 0:
                s.place_order(sym, "sell", held, order_type="stop", stop_price=stop, tag="stop-loss")

    metrics = sim.run(agent)

    # ---------------------------------------------------------------- results
    section("6. Portfolio")
    pf = sim.get_portfolio()
    for k in ("equity", "cash", "realized_pnl", "unrealized_pnl", "total_fees", "total_slippage_cost",
              "max_drawdown_pct", "num_fills", "num_closed_trades"):
        print(f"  {k:22s} {pf[k]}")
    print("\n  Open positions:")
    for p in pf["positions"]:
        print(f"    {p['symbol']:8s} qty {p['quantity']:>18s} avg {p['avg_price']:>14s} value {p['market_value']:>11s} "
              f"uPnL {p['unrealized_pnl']:>9s} weight {p['weight']:.1%}")

    section("7. Performance metrics")
    for k in ("total_return_pct", "cagr_pct", "volatility_annual_pct", "sharpe", "sortino", "max_drawdown_pct",
              "calmar", "num_trades", "win_rate_pct", "profit_factor", "expectancy", "total_fees",
              "total_slippage_cost", "turnover_annual"):
        v = metrics.get(k)
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")

    section("8. Persistence")
    state = Path(args.log_dir) / "state.json"
    sim.save_state(state)
    restored = TradingSimulator.load_state(state, sim.provider, universe)
    same = restored.portfolio.equity == sim.portfolio.equity
    print(f"  saved {state} and restored it: equity matches = {same}")
    print(f"  event log: {Path(args.log_dir) / 'events.jsonl'}   trade db: {Path(args.log_dir) / 'trades.sqlite'}")
    sim.close()
    restored.close()
    assert isinstance(sim.portfolio.equity, Decimal)


if __name__ == "__main__":
    main()
