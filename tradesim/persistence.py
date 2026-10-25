"""Event logging (JSONL), trade store (SQLite) and JSON state snapshots."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

log = logging.getLogger("tradesim.events")


def _default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "value"):  # enums
        return o.value
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=_default, **kw)


class EventLog:
    """Append-only JSONL log of everything that happens (orders, fills, rejections, margin calls).

    Also mirrors each event to the ``tradesim.events`` Python logger at DEBUG level.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, event: str, sim_time: datetime | None, **data: Any) -> None:
        record = {"event": event, "sim_time": sim_time.isoformat() if sim_time else None, **data}
        if log.isEnabledFor(logging.DEBUG):
            log.debug(dumps(record))
        if self._fh:
            self._fh.write(dumps(record) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class TradeStore:
    """SQLite store for orders, fills, closed trades and the equity curve of a run."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS orders (run_id TEXT, id TEXT, symbol TEXT, side TEXT, order_type TEXT,
        quantity TEXT, limit_price TEXT, stop_price TEXT, time_in_force TEXT, status TEXT, filled_quantity TEXT,
        avg_fill_price TEXT, fees_paid TEXT, created_at TEXT, updated_at TEXT, reject_reason TEXT, tag TEXT,
        PRIMARY KEY (run_id, id));
    CREATE TABLE IF NOT EXISTS fills (run_id TEXT, id TEXT, order_id TEXT, symbol TEXT, side TEXT, quantity TEXT,
        price TEXT, fee TEXT, liquidity TEXT, slippage_cost TEXT, realized_pnl TEXT, timestamp TEXT,
        PRIMARY KEY (run_id, id));
    CREATE TABLE IF NOT EXISTS closed_trades (run_id TEXT, symbol TEXT, direction TEXT, quantity TEXT,
        entry_price TEXT, exit_price TEXT, pnl TEXT, fees TEXT, opened_at TEXT, closed_at TEXT);
    CREATE TABLE IF NOT EXISTS equity (run_id TEXT, timestamp TEXT, equity REAL, cash REAL, gross_exposure REAL,
        net_exposure REAL);
    """

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(self.SCHEMA)

    def upsert_order(self, o: dict[str, Any]) -> None:
        cols = ["id", "symbol", "side", "order_type", "quantity", "limit_price", "stop_price", "time_in_force",
                "status", "filled_quantity", "avg_fill_price", "fees_paid", "created_at", "updated_at",
                "reject_reason", "tag"]
        self.conn.execute(
            f"INSERT OR REPLACE INTO orders (run_id, {', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) + 1))})",
            [self.run_id] + [o.get(c) for c in cols],
        )

    def add_fill(self, f: dict[str, Any]) -> None:
        cols = ["id", "order_id", "symbol", "side", "quantity", "price", "fee", "liquidity", "slippage_cost",
                "realized_pnl", "timestamp"]
        self.conn.execute(
            f"INSERT OR REPLACE INTO fills (run_id, {', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) + 1))})",
            [self.run_id] + [f.get(c) for c in cols],
        )

    def add_closed_trade(self, t: dict[str, Any]) -> None:
        cols = ["symbol", "direction", "quantity", "entry_price", "exit_price", "pnl", "fees", "opened_at", "closed_at"]
        self.conn.execute(
            f"INSERT INTO closed_trades (run_id, {', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) + 1))})",
            [self.run_id] + [t.get(c) for c in cols],
        )

    def add_equity(self, ts: datetime, equity: Decimal, cash: Decimal, gross: Decimal, net: Decimal) -> None:
        self.conn.execute("INSERT INTO equity VALUES (?, ?, ?, ?, ?, ?)",
                          (self.run_id, ts.isoformat(), float(equity), float(cash), float(gross), float(net)))

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(data, indent=1))
    tmp.replace(path)  # atomic on POSIX


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
