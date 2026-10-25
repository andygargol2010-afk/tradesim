"""Daily live operation: several agents, each with its own account, on real-time prices.

A run (typically from a scheduler such as GitHub Actions) does, for every enabled account:

1. load the account (orders, positions, equity history) and the agent's learned state from disk;
2. ``sync``: execute orders still pending whose market is now open, mark positions to market and
   record an equity point;
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


def _count_orders(action: Any) -> int:
    if action is None:
        return 0
    if isinstance(action, (list, tuple)):
        return sum(_count_orders(a) for a in action)
    weights = getattr(action, "weights", None) or (action.get("weights") if isinstance(action, dict) else None)
    return len(weights) if weights else 1


class LiveRunner:
    """Runs the accounts of a :class:`LiveConfig` once (``run("decide")`` or ``run("mark")``).

    Args:
        config: accounts, symbols, data sources, risk limits and file locations.
        provider: market data (default: :func:`default_live_provider`).
        clock: current UTC time (inject for tests and replays).
    """

    def __init__(self, config: LiveConfig, provider: DataProvider | None = None,
                 universe: AssetUniverse | None = None, clock: Callable[[], datetime] | None = None):
        self.config = config
        self.universe = universe or AssetUniverse.default()
        self.base_provider = provider or default_live_provider(config.crypto_exchange, config.stocks_data)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        unknown = [s for s in config.symbols if s not in self.universe]
        if unknown:
            raise ValueError(f"Unknown symbols in config: {unknown}")

    # ------------------------------------------------------------------ files
    def account_dir(self, spec: AccountSpec) -> Path:
        return Path(self.config.state_dir) / spec.id

    def _sim_config(self, spec: AccountSpec) -> SimulatorConfig:
        return SimulatorConfig(initial_cash=spec.initial_cash, timeframe=self.config.timeframe, fees=spec.fees,
                               max_position_pct=self.config.risk.max_position_pct, seed=0)

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

    def run(self, mode: str = "decide", export: bool = True) -> dict[str, Any]:
        """One pass over all accounts. ``mode``: ``"decide"`` (sync + daily decision) or ``"mark"`` (sync only)."""
        if mode not in ("decide", "mark"):
            raise ValueError("mode must be 'decide' or 'mark'")
        t0 = time.perf_counter()
        now = self.clock()
        provider = RunCacheProvider(self.base_provider)
        prices = self.price_snapshot(provider)
        quoted = [s for s, p in prices.items() if "error" not in p]
        window_ok, window_msg = self.decision_window_open(now)
        record: dict[str, Any] = {"started_at": now.isoformat(), "mode": mode, "accounts": {}, "prices": prices}
        for spec in self.config.accounts:
            if not spec.enabled:
                continue
            try:
                record["accounts"][spec.id] = self._run_account(spec, provider, mode, now, quoted, window_ok, window_msg)
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
        return record

    def _run_account(self, spec: AccountSpec, provider: RunCacheProvider, mode: str, now: datetime,
                     quoted: list[str], window_ok: bool, window_msg: str) -> dict[str, Any]:
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
            elif not window_ok:
                status, message = "skipped", window_msg
            elif drawdown >= self.config.risk.halt_drawdown_pct:
                status, message = "halted", f"drawdown {drawdown:.1f}% >= halt limit {self.config.risk.halt_drawdown_pct}%"
            elif not trader.symbols:
                status, message = "skipped", "no prices available"
            else:
                entry = self._decide(spec, trader, now)
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

    def _decide(self, spec: AccountSpec, trader, now: datetime) -> dict[str, Any]:
        agent = self.load_agent(spec)
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
        t_act = time.perf_counter()
        action = agent.act(obs)
        entry["think_time_s"] = round(time.perf_counter() - t_act, 3)
        n_orders = _count_orders(action)
        if n_orders > self.config.risk.max_orders_per_run:
            entry.update(status="rejected", message=f"agent wanted {n_orders} orders > limit "
                                                    f"{self.config.risk.max_orders_per_run}; nothing sent")
            results = []
        else:
            results = trader.apply_action(action)
            trader.sync()  # execute now against the current quotes / order books
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
