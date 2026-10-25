"""Build ``dashboard/live.json`` from the live state on disk (accounts, journals, run log)."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from tradesim.assets.universe import AssetUniverse
from tradesim.persistence import read_json, write_json
from tradesim.portfolio.metrics import compute_metrics
from tradesim.portfolio.portfolio import Portfolio
from tradesim.trading.config import AccountSpec, LiveConfig

log = logging.getLogger(__name__)
JOURNAL_ENTRIES = 120
EQUITY_POINTS = 2000


def _num(x: Any, sig: int = 8) -> Any:
    if x is None:
        return None
    if isinstance(x, Decimal):
        x = float(x)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
        return float(f"{x:.{sig}g}") if isinstance(x, float) else x
    return x


def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _equity_at(curve: pd.Series, ts: datetime | None) -> float | None:
    if ts is None or curve.empty:
        return None
    i = curve.index.searchsorted(pd.Timestamp(ts), side="right") - 1
    return float(curve.iloc[max(i, 0)])


def account_payload(spec: AccountSpec, account_dir: Path, universe: AssetUniverse, now: datetime) -> dict[str, Any]:
    """Everything the dashboard shows for one account."""
    meta = read_json(account_dir / "meta.json") if (account_dir / "meta.json").exists() else {}
    base: dict[str, Any] = {"id": spec.id, "name": spec.name, "description": spec.description, "agent": spec.agent,
                            "params": spec.params, "broker": spec.broker, "fees": spec.fees,
                            "initial_cash": spec.initial_cash, "meta": meta}
    state_path = account_dir / "account.json"
    if not state_path.exists():
        return {**base, "status": "not started"}
    st = read_json(state_path)
    pf = Portfolio.from_state(st["portfolio"])
    orders = {o["id"]: o for o in st.get("orders", [])}
    tag = lambda oid: (orders.get(oid) or {}).get("tag") if oid else None  # noqa: E731

    curve = pd.Series([float(p.equity) for p in pf.equity_curve],
                      index=pd.DatetimeIndex([p.timestamp for p in pf.equity_curve]).tz_convert("UTC"))
    curve = curve[~curve.index.duplicated(keep="last")].sort_index()
    equity = float(pf.equity)
    initial = float(pf.initial_cash)

    # Day change: against the last equity point before today's UTC midnight.
    midnight = pd.Timestamp(now).tz_convert("UTC").normalize()
    before = curve[curve.index < midnight]
    day_ref = float(before.iloc[-1]) if len(before) else initial

    daily = curve.resample("1D").last().dropna()
    m = compute_metrics(daily, pf.closed_trades, periods_per_year=365.0, total_fees=float(pf.total_fees),
                        exposure=None) if len(daily) >= 2 else {}
    enough = len(daily) >= 20  # ratios on fewer days are noise

    positions = []
    for p in pf.open_positions():
        cost = p.quantity * p.avg_price
        allocated = float(cost + p.open_fees)
        eq_entry = _equity_at(curve, p.opened_at)
        days = (now - p.opened_at).total_seconds() / 86400 if p.opened_at else None
        positions.append({
            "symbol": p.symbol, "name": universe.get(p.symbol).name if p.symbol in universe else "",
            "opened_at": p.opened_at.isoformat() if p.opened_at else None, "days_held": _num(days, 4),
            "quantity": format(p.quantity.normalize(), "f"), "avg_price": _num(p.avg_price),
            "allocated": _num(allocated), "allocated_pct_at_entry": _num(allocated / eq_entry * 100, 4) if eq_entry else None,
            "last_price": _num(p.mark_price), "market_value": _num(p.market_value),
            "unrealized_pnl": _num(p.market_value - cost - p.open_fees, 6),
            "unrealized_pct": _num(float((p.market_value - cost - p.open_fees) / (cost + p.open_fees) * 100), 5)
            if cost else None,
            "weight_pct": _num(float(p.market_value) / equity * 100, 4) if equity else None,
            "entry_fees": _num(p.open_fees, 6), "entry_reason": tag(p.entry_order_id),
        })
    positions.sort(key=lambda r: -(r["market_value"] or 0))

    trades = []
    for t in reversed(pf.closed_trades):
        allocated = float(t.allocated)
        eq_entry = _equity_at(curve, t.opened_at)
        trades.append({
            "symbol": t.symbol, "direction": t.direction,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None, "closed_at": t.closed_at.isoformat(),
            "days_held": _num((t.closed_at - t.opened_at).total_seconds() / 86400, 4) if t.opened_at else None,
            "quantity": format(t.quantity.normalize(), "f"), "entry_price": _num(t.entry_price),
            "exit_price": _num(t.exit_price), "allocated": _num(allocated),
            "allocated_pct_at_entry": _num(allocated / eq_entry * 100, 4) if eq_entry else None,
            "proceeds": _num(t.quantity * t.exit_price), "pnl": _num(t.pnl, 6),
            "return_pct": _num(t.return_pct * 100, 5), "fees": _num(t.fees, 6),
            "entry_reason": tag(t.entry_order_id), "exit_reason": tag(t.exit_order_id),
        })

    open_orders = [{k: o.get(k) for k in ("id", "symbol", "side", "quantity", "order_type", "limit_price",
                                          "stop_price", "status", "created_at", "tag")}
                   for o in st.get("orders", []) if o.get("status") in ("new", "partially_filled")]

    learning = None
    try:
        from tradesim.trading.runner import LiveRunner  # noqa: F401 - agent building lives in the spec

        agent = spec.build_agent()
        agent_path = account_dir / "agent.json"
        if agent_path.exists():
            agent.load_state_dict(read_json(agent_path).get("state", {}))
        learning = agent.learning_summary()
    except Exception as exc:  # noqa: BLE001 - the dashboard must render even if an agent cannot load
        learning = {"error": f"{type(exc).__name__}: {exc}"}

    pts = curve.iloc[-EQUITY_POINTS:]
    pnls = [float(t.pnl) for t in pf.closed_trades]
    holds = [(t.closed_at - t.opened_at).total_seconds() / 86400 for t in pf.closed_trades if t.opened_at]
    return {
        **base,
        "status": meta.get("last_status", "ok"),
        "started_at": curve.index[0].isoformat() if len(curve) else None,
        "summary": {
            "equity": _num(equity), "cash": _num(pf.cash), "invested": _num(pf.gross_exposure),
            "invested_pct": _num(float(pf.gross_exposure) / equity * 100, 4) if equity else None,
            "pnl": _num(equity - initial, 8), "pnl_pct": _num((equity / initial - 1) * 100, 6),
            "day_pnl": _num(equity - day_ref, 6), "day_pnl_pct": _num((equity / day_ref - 1) * 100, 5) if day_ref else None,
            "realized_pnl": _num(pf.realized_pnl), "unrealized_pnl": _num(pf.unrealized_pnl),
            "fees": _num(pf.total_fees, 6), "slippage_cost": _num(pf.total_slippage, 6),
            "max_drawdown_pct": _num(float(pf.max_drawdown) * 100, 5),
            "current_drawdown_pct": _num(float((pf.peak_equity - pf.equity) / pf.peak_equity * 100), 5)
            if pf.peak_equity else None,
            "sharpe": _num(m.get("sharpe"), 4) if enough else None,
            "sortino": _num(m.get("sortino"), 4) if enough else None,
            "volatility_annual_pct": _num(m.get("volatility_annual_pct"), 4) if enough else None,
            "days_live": len(daily),
            "num_trades": len(pnls), "num_fills": len(pf.fills), "open_positions": len(positions),
            "win_rate_pct": _num(sum(p > 0 for p in pnls) / len(pnls) * 100, 4) if pnls else None,
            "profit_factor": _num(m.get("profit_factor"), 4) if pnls else None,
            "avg_trade_pnl": _num(sum(pnls) / len(pnls), 6) if pnls else None,
            "best_trade": _num(max(pnls), 6) if pnls else None, "worst_trade": _num(min(pnls), 6) if pnls else None,
            "avg_days_held": _num(sum(holds) / len(holds), 4) if holds else None,
        },
        "equity_curve": {"t": [t.isoformat() for t in pts.index], "equity": [_num(v) for v in pts.to_numpy()]},
        "positions": positions,
        "closed_trades": trades,
        "open_orders": open_orders,
        "journal": list(reversed(_tail_jsonl(account_dir / "journal.jsonl", JOURNAL_ENTRIES))),
        "learning": json.loads(json.dumps(learning, default=str)) if learning is not None else None,
    }


def export_dashboard(config: LiveConfig, universe: AssetUniverse | None = None,
                     now: datetime | None = None) -> dict[str, Any]:
    """Write ``config.dashboard_file`` and return its content."""
    universe = universe or AssetUniverse.default()
    now = now or datetime.now(timezone.utc)
    runs = _tail_jsonl(Path(config.runs_log), 60)
    last = runs[-1] if runs else None
    accounts = []
    for spec in config.accounts:
        try:
            accounts.append(account_payload(spec, Path(config.state_dir) / spec.id, universe, now))
        except Exception as exc:  # noqa: BLE001
            log.exception("export failed for %s", spec.id)
            accounts.append({"id": spec.id, "name": spec.name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    data = {
        "meta": {
            "generated_at": now.isoformat(),
            "symbols": config.symbols,
            "names": {s: universe.get(s).name for s in config.symbols if s in universe},
            "asset_class": {s: universe.get(s).asset_class.value for s in config.symbols if s in universe},
            "schedule": config.schedule,
            "risk": config.risk.__dict__,
            "crypto_exchange": config.crypto_exchange,
            "stocks_data": config.stocks_data,
        },
        "last_run": {k: last.get(k) for k in ("started_at", "mode", "duration_s", "accounts", "data_errors")} if last else None,
        "prices": last.get("prices") if last else {},
        "runs": [{"started_at": r.get("started_at"), "mode": r.get("mode"), "duration_s": r.get("duration_s"),
                  "errors": len(r.get("data_errors") or {}) + sum(1 for a in (r.get("accounts") or {}).values()
                                                                   if a.get("status") == "error")}
                 for r in reversed(runs[-30:])],
        "accounts": accounts,
    }
    write_json(config.dashboard_file, data)
    return data
