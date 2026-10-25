"""Daily live operation: several agents, each with its own account, on real-time prices.

A run (typically from a scheduler such as GitHub Actions) does, for every enabled account:

1. load the account (orders, positions, equity history) and the agent's learned state from disk;
2. ``sync``: execute orders still pending whose market is now open, mark positions to market and
   record an equity point;
0. in ``decide`` and ``research`` modes, refresh the market research first (news analysis, macro, event
   calendar, technical patterns; see :mod:`tradesim.research`);
3. in ``decide`` mode, once per day and only when every market of the universe is open: build an
   :class:`~tradesim.agents.base.Observation` from live prices, call ``agent.act``, apply risk limits,
   send the orders and execute them immediately against the current quotes / order books;
4. save the account, the agent's learned state and a journal entry (orders, fills, reasoning).

Finally it appends a record to ``runs.jsonl`` and regenerates the dashboard data. All accounts in a run
see exactly the same prices. A failing account never stops the others.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tradesim.assets.universe import AssetUniverse
from tradesim.data.base import DataProvider, DataUnavailableError, RoutingProvider
from tradesim.live import LivePaperTrader, is_market_open
from tradesim.persistence import dumps, read_json, write_json
from tradesim.simulator import SimulatorConfig
from tradesim.trading.config import AccountSpec, LiveConfig
from tradesim.trading.provider import RunCacheProvider, default_live_provider

log = logging.getLogger(__name__)


def _flatten(action: Any) -> list[Any]:
    if action is None:
        return []
    if isinstance(action, (list, tuple)):
        return [x for a in action for x in _flatten(a)]
    return [action]


def _is_exit(action: Any) -> bool:
    """Sells and closes run first so that their proceeds can fund the buys of the same decision."""
    from tradesim.agents.base import ClosePosition, OrderIntent

    if isinstance(action, ClosePosition):
        return True
    if isinstance(action, OrderIntent):
        return action.side.lower() == "sell"
    if isinstance(action, dict):
        return "close" in action or str(action.get("side", "")).lower() == "sell"
    return False


def _count_orders(action: Any) -> int:
    if action is None:
        return 0
    if isinstance(action, (list, tuple)):
        return sum(_count_orders(a) for a in action)
    weights = getattr(action, "weights", None) or (action.get("weights") if isinstance(action, dict) else None)
    return len(weights) if weights else 1


def _research_digest(r: dict[str, Any] | None) -> dict[str, Any] | None:
    if r is None:
        return None
    keep = ("status", "message", "llm", "macro", "calendar", "news", "technicals", "duration_s", "calendar_research")
    out = {k: r[k] for k in keep if k in r}
    if r.get("llm_usage"):
        out["llm_cost_usd"] = r["llm_usage"].get("cost_usd_estimate")
    if r.get("errors"):
        out["errors"] = len(r["errors"])
    return out


def _buy_notional(a: Any, trader) -> float | None:
    from tradesim.agents.base import OrderIntent

    if not isinstance(a, OrderIntent) or a.side.lower() != "buy":
        return None
    if a.notional is not None:
        return float(a.notional)
    if a.percent_of_equity is not None:
        return float(a.percent_of_equity) / 100 * float(trader.portfolio.equity)
    if a.quantity is not None:
        return float(a.quantity) * float(trader.get_price(a.symbol)["ask"])
    return None


def _fund_from_cash(entries: list[Any], trader, entry: dict[str, Any]) -> list[Any]:
    """Partial sessions: buys may only use free cash (positions in closed markets cannot be sold to fund them)."""
    from dataclasses import replace

    wanted = {i: n for i, a in enumerate(entries) if (n := _buy_notional(a, trader))}
    total = sum(wanted.values())
    cash = float(trader.portfolio.cash) * 0.99  # small buffer for fees and spread
    if total <= cash or total <= 0:
        return entries
    k = max(cash, 0.0) / total
    entry["funding"] = f"buys scaled to {k:.0%} of the request: free cash {cash:.2f} < {total:.2f}"
    out = []
    for i, a in enumerate(entries):
        if i in wanted:
            n = round(wanted[i] * k, 2)
            if n >= 1.0:
                out.append(replace(a, notional=n, quantity=None, percent_of_equity=None))
        else:
            out.append(a)
    return out


class LiveRunner:
    """Runs the accounts of a :class:`LiveConfig` once (``run("decide")`` or ``run("mark")``).

    Args:
        config: accounts, symbols, data sources, risk limits and file locations.
        provider: market data (default: :func:`default_live_provider`).
        clock: current UTC time (inject for tests and replays).
        llm_client: Anthropic client for research and LLM agents (``"auto"``: from ``ANTHROPIC_API_KEY``).
    """

    def __init__(self, config: LiveConfig, provider: DataProvider | None = None,
                 universe: AssetUniverse | None = None, clock: Callable[[], datetime] | None = None,
                 llm_client: Any = "auto"):
        self.config = config
        self.universe = universe or AssetUniverse.default()
        self.base_provider = provider or default_live_provider(config.crypto_exchange, config.stocks_data)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.llm_client: Any = llm_client
        unknown = [s for s in config.symbols if s not in self.universe]
        if unknown:
            raise ValueError(f"Unknown symbols in config: {unknown}")

    # ------------------------------------------------------------------ files
    def account_dir(self, spec: AccountSpec) -> Path:
        return Path(self.config.state_dir) / spec.id

    def _sim_config(self, spec: AccountSpec) -> SimulatorConfig:
        return SimulatorConfig(initial_cash=spec.initial_cash, timeframe=self.config.timeframe, fees=spec.fees,
                               max_position_pct=self.config.risk.max_position_pct,
                               auto_stop_atr=(spec.stop_loss_atr or self.config.risk.stop_loss_atr) if spec.auto_stop else None,
                               seed=0)

    def load_trader(self, spec: AccountSpec, provider: DataProvider):
        path = self.account_dir(spec) / "account.json"
        if spec.broker == "paper":
            kw = dict(clock=self.clock, quote_ttl=600.0, symbols=list(self.config.symbols))
            if path.exists():
                return LivePaperTrader.load_state(path, provider, self.universe, self._sim_config(spec), **kw)
            return LivePaperTrader(provider, self.universe, self._sim_config(spec), **kw)
        if spec.broker in ("alpaca-paper", "alpaca-live"):
            from tradesim.trading.alpaca import AlpacaTrader

            trader = AlpacaTrader(paper=spec.broker == "alpaca-paper", universe=self.universe,
                                  config=self._sim_config(spec), data_provider=provider, clock=self.clock,
                                  symbols=list(self.config.symbols))
            if path.exists():
                trader.load_ledger(path)
            return trader
        raise ValueError(f"Unknown broker {spec.broker!r} for account {spec.id}")

    def load_agent(self, spec: AccountSpec):
        agent = spec.build_agent()
        path = self.account_dir(spec) / "agent.json"
        if path.exists():
            agent.load_state_dict(read_json(path).get("state", {}))
        return agent

    def _save_agent(self, spec: AccountSpec, agent) -> None:
        write_json(self.account_dir(spec) / "agent.json", {
            "agent": spec.agent, "params": spec.params, "saved_at": self.clock().isoformat(),
            "state": agent.state_dict(),
        })

    def _meta(self, spec: AccountSpec) -> dict[str, Any]:
        path = self.account_dir(spec) / "meta.json"
        return read_json(path) if path.exists() else {"created_at": self.clock().isoformat()}

    def _journal(self, spec: AccountSpec, entry: dict[str, Any]) -> None:
        path = self.account_dir(spec) / "journal.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(dumps(entry) + "\n")

    # ------------------------------------------------------------------ run
    def decision_window_open(self, now: datetime) -> tuple[bool, str]:
        """Decisions need every market of the universe open (mixed sessions make rebalances inconsistent)."""
        closed = [s for s in self.config.symbols if not is_market_open(self.universe.get(s), now)]
        if closed:
            return False, f"market closed for {len(closed)} symbols (e.g. {closed[0]}): no decisions today"
        return True, ""

    def price_snapshot(self, provider: RunCacheProvider) -> dict[str, Any]:
        out: dict[str, Any] = {}
        now = self.clock()
        for sym in self.config.symbols:
            asset = self.universe.get(sym)
            source = None
            if isinstance(self.base_provider, RoutingProvider):
                try:
                    source = self.base_provider._pick(asset).name
                except DataUnavailableError:
                    source = None
            try:
                q = provider.get_quote(asset)
                out[sym] = {"last": q.last, "bid": q.bid, "ask": q.ask, "spread_bps": round(q.spread_bps, 3),
                            "modelled_spread": q.modelled, "time": q.timestamp.isoformat(), "source": source,
                            "market_open": is_market_open(asset, now)}
            except DataUnavailableError as exc:
                out[sym] = {"error": str(exc)[:300], "source": source, "market_open": is_market_open(asset, now)}
        return out

    def research_store(self):
        rc = self.config.research_config()
        if not rc.enabled:
            return None
        from tradesim.research.refresh import ResearchStore

        return ResearchStore(rc.dir)

    def holdings(self) -> set[str]:
        """Symbols held by any account (their news gets priority in the deep analysis)."""
        out: set[str] = set()
        for spec in self.config.accounts:
            path = self.account_dir(spec) / "account.json"
            if path.exists():
                try:
                    pf = read_json(path).get("portfolio", {})
                    out |= {p.get("symbol") for p in pf.get("positions", []) if p.get("symbol")}
                except (OSError, ValueError):
                    continue
        return out

    def refresh_research(self, provider: DataProvider) -> dict[str, Any]:
        store = self.research_store()
        if store is None:
            return {"status": "disabled"}
        from tradesim.research.refresh import refresh_research

        try:
            return refresh_research(self.config.research_config(), list(self.config.symbols), self.universe, store,
                                    provider=provider, now=self.clock(), client=self.llm_client,
                                    holdings=self.holdings())
        except Exception as exc:  # research must never block trading
            log.exception("research refresh failed")
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}

    def run(self, mode: str = "decide", export: bool = True) -> dict[str, Any]:
        """One pass. ``mode``: ``"decide"`` (research + sync + daily decision), ``"mark"`` (sync only) or
        ``"research"`` (refresh news / macro / calendar / patterns, then sync like ``mark``). Every sync
        executes pending orders and the protective stop-loss / take-profit orders the market reached."""
        if mode not in ("decide", "mark", "research"):
            raise ValueError("mode must be 'decide', 'mark' or 'research'")
        t0 = time.perf_counter()
        now = self.clock()
        provider = RunCacheProvider(self.base_provider)
        research = self.refresh_research(provider) if mode in ("decide", "research") else None
        # research mode also syncs the accounts like "mark": resting stop-loss / take-profit orders are
        # checked every hour and the equity curve gets an hourly point.
        prices = self.price_snapshot(provider)
        quoted = [s for s, p in prices.items() if "error" not in p]
        window_ok, window_msg = self.decision_window_open(now)
        open_now = [s for s in self.config.symbols if s in quoted and is_market_open(self.universe.get(s), now)]
        partial = open_now if (not window_ok and self.config.weekend_crypto and open_now) else None
        record: dict[str, Any] = {"started_at": now.isoformat(), "mode": mode, "accounts": {}, "prices": prices}
        if research is not None:
            record["research"] = _research_digest(research)
        for spec in self.config.accounts:
            if not spec.enabled:
                continue
            try:
                record["accounts"][spec.id] = self._run_account(spec, provider, mode, now, quoted, window_ok, window_msg,
                                                                   partial)
            except Exception as exc:  # one broken account must not stop the others
                log.exception("account %s failed", spec.id)
                record["accounts"][spec.id] = {"status": "error", "message": f"{type(exc).__name__}: {exc}",
                                               "trace": traceback.format_exc(limit=5)}
                self._journal(spec, {"time": now.isoformat(), "mode": mode, "status": "error",
                                     "message": f"{type(exc).__name__}: {exc}"})
        record["data_errors"] = provider.errors
        record["duration_s"] = round(time.perf_counter() - t0, 2)
        runs = Path(self.config.runs_log)
        runs.parent.mkdir(parents=True, exist_ok=True)
        with open(runs, "a", encoding="utf-8") as fh:
            fh.write(dumps(record) + "\n")
        if export:
            from tradesim.trading.export import export_dashboard

            export_dashboard(self.config, self.universe, now=self.clock())
            if self.config.chronicle:
                from tradesim.trading.chronicle import maybe_write

                try:  # yesterday's chronicle, once; never blocks the trading run
                    entry = maybe_write(self.config, self.clock())
                    if entry:
                        record["chronicle"] = {"date": entry["date"], "engine": entry["engine"]}
                except Exception as exc:  # noqa: BLE001
                    log.warning("chronicle failed: %s", exc)
                    record["chronicle"] = {"error": str(exc)[:300]}
        return record

    def _run_account(self, spec: AccountSpec, provider: RunCacheProvider, mode: str, now: datetime,
                     quoted: list[str], window_ok: bool, window_msg: str,
                     partial: list[str] | None = None) -> dict[str, Any]:
        trader = self.load_trader(spec, provider)
        meta = self._meta(spec)
        meta.setdefault("created_at", now.isoformat())
        trader.symbols = [s for s in self.config.symbols if s in quoted]
        synced = trader.sync()  # queued orders whose market opened + mark to market
        status, message, entry = "synced", "", None
        today = now.date().isoformat()
        if mode == "decide":
            pf = trader.portfolio
            drawdown = float((pf.peak_equity - pf.equity) / pf.peak_equity * 100) if pf.peak_equity > 0 else 0.0
            if meta.get("last_decision_date") == today:
                status, message = "skipped", "already decided today"
            elif not window_ok and partial is None:
                status, message = "skipped", window_msg
            elif drawdown >= self.config.risk.halt_drawdown_pct:
                status, message = "halted", f"drawdown {drawdown:.1f}% >= halt limit {self.config.risk.halt_drawdown_pct}%"
            elif not trader.symbols:
                status, message = "skipped", "no prices available"
            else:
                entry = self._decide(spec, trader, now, restrict=None if window_ok else partial)
                status = entry["status"]
                message = entry.get("message", "")
                meta["last_decision_date"] = today
            if status in ("skipped", "halted"):
                entry = {"time": now.isoformat(), "mode": mode, "status": status, "message": message,
                         "equity": float(pf.equity)}
            if entry is not None:
                self._journal(spec, entry)
        meta["last_run"] = now.isoformat()
        meta["last_status"] = status
        if hasattr(trader, "save_state"):
            trader.save_state(self.account_dir(spec) / "account.json")
        write_json(self.account_dir(spec) / "meta.json", meta)
        trader.close()
        return {"status": status, "message": message, "equity": float(trader.portfolio.equity),
                "fills_on_sync": len(synced)}

    def _restrict(self, items: list[Any], open_syms: set[str], trader, entry: dict[str, Any]) -> list[Any]:
        """Keep the part of a decision that can execute now; the rest waits for the next full session."""
        from tradesim.agents.base import CancelOrders, ClosePosition, OrderIntent, SetExits, TargetWeights

        out: list[Any] = []
        deferred: set[str] = set()
        for a in items:
            weights = a.weights if isinstance(a, TargetWeights) else a.get("weights") if isinstance(a, dict) else None
            if weights is not None:  # rebalance: translate the open-market part into orders
                eq = float(trader.portfolio.equity)
                deferred |= {s for s in weights if s not in open_syms}
                held = {p.symbol for p in trader.portfolio.open_positions()}
                for sym in sorted((set(weights) | held) & open_syms):
                    px = float(trader.get_price(sym)["mid"])
                    if not px > 0:
                        continue
                    delta = float(weights.get(sym, 0.0)) * eq - float(trader.portfolio.quantity(sym)) * px
                    if delta >= 1.0:
                        out.append(OrderIntent.buy(sym, notional=round(delta, 2), tag="rebalance (solo cripto)"))
                    elif delta <= -1.0:
                        qty = min(float(trader.portfolio.quantity(sym)), -delta / px)
                        out.append(OrderIntent.sell(sym, quantity=qty, tag="rebalance (solo cripto)"))
                continue
            sym = getattr(a, "symbol", None) if not isinstance(a, dict) else (a.get("symbol") or a.get("close"))
            if isinstance(a, CancelOrders) and sym is None:
                continue  # would also cancel the protective orders of positions in closed markets
            if isinstance(a, SetExits) or sym in open_syms:
                out.append(a)
            elif isinstance(a, (OrderIntent, ClosePosition)) or isinstance(a, dict):
                deferred.add(sym)
        entry["deferred_to_next_session"] = sorted(s for s in deferred if s)
        entry["message"] = (f"partial session: only {len(open_syms)} open markets traded; "
                            f"{len(entry['deferred_to_next_session'])} decisions wait for the next full session")
        return out

    def _decide(self, spec: AccountSpec, trader, now: datetime, restrict: list[str] | None = None) -> dict[str, Any]:
        """One decision. With ``restrict`` (the symbols whose market is open) the agent still decides on the whole
        universe, but only the part on open markets is executed and buys are funded with free cash only: a crypto
        purchase happens at the weekend only if the agent prefers it over everything it could buy on Monday."""
        agent = self.load_agent(spec)
        if hasattr(agent, "attach_research"):
            agent.attach_research(self.research_store())
        if getattr(agent, "client", "absent") is None and self.llm_client not in ("auto", None):
            agent.client = self.llm_client
        pf = trader.portfolio
        entry: dict[str, Any] = {"time": now.isoformat(), "mode": "decide", "equity_before": float(pf.equity),
                                 "cash_before": float(pf.cash), "symbols": len(trader.symbols)}
        n_fills = len(pf.fills)
        # Pre-flight: agents only see symbols whose recent history loads (a dead feed must not crash them).
        usable, dropped = [], {}
        for sym in trader.symbols:
            try:
                if len(trader.recent_closes(sym, 60)) >= 2:
                    usable.append(sym)
                else:
                    dropped[sym] = "not enough history"
            except DataUnavailableError as exc:
                dropped[sym] = str(exc)[:200]
        trader.symbols = usable
        if dropped:
            entry["dropped_symbols"] = dropped
        obs = trader.observe()
        if restrict is not None:
            obs.tradable = [s for s in restrict if s in usable]
            entry["session"] = "partial"
        t_act = time.perf_counter()
        action = agent.act(obs)
        entry["think_time_s"] = round(time.perf_counter() - t_act, 3)
        n_orders = _count_orders(action)
        if n_orders > self.config.risk.max_orders_per_run:
            entry.update(status="rejected", message=f"agent wanted {n_orders} orders > limit "
                                                    f"{self.config.risk.max_orders_per_run}; nothing sent")
            results = []
        else:
            items = _flatten(action)
            if restrict is not None:
                items = self._restrict(items, set(obs.tradable or []), trader, entry)
            results = trader.apply_action([a for a in items if _is_exit(a)])
            trader._settle()  # execute the exits now: their cash funds the entries below
            entries = [a for a in items if not _is_exit(a)]
            if restrict is not None:
                entries = _fund_from_cash(entries, trader, entry)
            results += trader.apply_action(entries)
            trader.sync()  # execute now against the current quotes / order books
            # Journal the orders as they ended up (filled / rejected...), not as they were submitted.
            results = [trader.orders[r["id"]].to_dict() if r.get("id") in trader.orders else r for r in results]
            entry["status"] = "ok"
        entry["orders"] = [{k: r.get(k) for k in ("id", "symbol", "side", "quantity", "order_type", "limit_price",
                                                  "stop_price", "status", "reject_reason", "tag", "filled_quantity",
                                                  "avg_fill_price")} for r in results]
        entry["fills"] = [f.to_dict() for f in pf.fills[n_fills:]]
        entry["equity_after"] = float(pf.equity)
        entry["cash_after"] = float(pf.cash)
        explain = agent.explain()
        if explain is not None:
            entry["explain"] = json.loads(dumps(explain))
        if not results and entry["status"] == "ok":
            entry["message"] = "no trades"
        self._save_agent(spec, agent)
        return entry
