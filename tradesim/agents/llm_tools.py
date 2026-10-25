"""Function-calling tools exposing the simulator to LLM agents.

``TOOLS`` is a list of tool definitions in the Anthropic Messages API format
(``name`` / ``description`` / ``input_schema``); :func:`openai_tools` converts
them for frameworks expecting the ``{"type": "function", ...}`` shape.
:func:`dispatch` executes a tool call against a simulator and always returns a
JSON string (errors are returned as ``{"error": ...}`` so the model can recover).

The same tools work for :class:`~tradesim.simulator.TradingSimulator` and
:class:`~tradesim.live.LivePaperTrader`, so an agent prompt developed in
backtests runs unchanged in live paper trading.
"""

from __future__ import annotations

import json
import math
from typing import Any

from tradesim.persistence import dumps

_SYMBOL = {"type": "string", "description": "Asset symbol, e.g. AAPL, SPY, BTC-USD, EURUSD=X"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_available_assets",
        "description": "Search the tradable universe (US stocks, ETFs, crypto, forex). Use filters; the full list has 2000+ symbols.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_class": {"type": "string", "enum": ["equity", "etf", "crypto", "forex"]},
                "query": {"type": "string", "description": "Substring of symbol or name"},
                "group": {"type": "string", "description": "e.g. sp500, nasdaq100, sp400, sp600, etf, crypto_top300, fx"},
                "max_liquidity_tier": {"type": "integer", "minimum": 1, "maximum": 5,
                                       "description": "1 = most liquid ... 5 = least"},
                "limit": {"type": "integer", "default": 50, "maximum": 500},
            },
        },
    },
    {
        "name": "get_price",
        "description": "Latest quote for a symbol: bid, ask, mid, last, spread in bps, timestamp. Buys pay the ask, sells receive the bid.",
        "input_schema": {"type": "object", "properties": {"symbol": _SYMBOL}, "required": ["symbol"]},
    },
    {
        "name": "get_portfolio",
        "description": "Account state: equity, cash, buying power, realized/unrealized PnL, fees, drawdown and all open positions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_historical_data",
        "description": "Completed OHLCV bars up to now (no future data). Optionally with indicators (SMA, EMA, RSI, ATR, MACD, Bollinger, volatility).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]},
                "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 500},
                "indicators": {"type": "boolean", "default": False},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "detect_patterns",
        "description": (
            "Technical pattern recognition on completed bars: trend, support/resistance levels, candlesticks "
            "(engulfing, hammer, stars...), chart patterns (double top/bottom, head and shoulders, triangles, wedges, "
            "breakouts), and indicator events (golden/death cross, RSI divergence, MACD cross, Bollinger squeeze). "
            "Each pattern has a direction, strength 0-1, status and key levels (neckline, target, invalidation)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "timeframe": {"type": "string", "enum": ["1h", "4h", "1d", "1w"]},
                "limit": {"type": "integer", "default": 300, "minimum": 60, "maximum": 1000},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "place_order",
        "description": (
            "Submit an order. Size with exactly ONE of quantity (units; fractions allowed, e.g. 0.003 BTC or 1.75 shares), "
            "notional (USD) or percent_of_equity (0-100). Orders execute at the next market update with realistic spread, "
            "slippage, liquidity limits and fees; check the returned status and reject_reason."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": _SYMBOL,
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "quantity": {"type": "number", "exclusiveMinimum": 0},
                "notional": {"type": "number", "exclusiveMinimum": 0},
                "percent_of_equity": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                "order_type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"], "default": "market"},
                "limit_price": {"type": "number", "description": "Required for limit and stop_limit"},
                "stop_price": {"type": "number", "description": "Required for stop and stop_limit"},
                "time_in_force": {"type": "string", "enum": ["gtc", "day", "ioc", "fok"], "default": "gtc"},
                "tag": {"type": "string", "description": "Short rationale stored with the order"},
            },
            "required": ["symbol", "side"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an open order by id.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "get_open_orders",
        "description": "List working (unfilled or partially filled) orders.",
        "input_schema": {"type": "object", "properties": {"symbol": _SYMBOL}},
    },
    {
        "name": "close_position",
        "description": "Cancel the symbol's open orders and flatten the position with a market order.",
        "input_schema": {"type": "object", "properties": {"symbol": _SYMBOL}, "required": ["symbol"]},
    },
    {
        "name": "get_trades",
        "description": "Recently closed round-trip trades with net PnL.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}},
    },
    {
        "name": "get_trade_history",
        "description": "Execution history: 'fills' = every buy/sell execution with price and fee; 'closed' = round trips with net PnL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fills", "closed"], "default": "fills"},
                "symbol": _SYMBOL,
                "limit": {"type": "integer", "default": 50, "maximum": 500},
            },
        },
    },
    {
        "name": "get_metrics",
        "description": "Performance metrics so far: return, Sharpe, Sortino, max drawdown, win rate, profit factor, fees.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def openai_tools() -> list[dict[str, Any]]:
    """Same tools in the ``{"type": "function", "function": {...}}`` format used by other frameworks."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


def _round(x: Any) -> Any:
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.6g}")
    return x


def dispatch(sim: Any, name: str, args: dict[str, Any] | None) -> str:
    """Execute tool ``name`` with ``args`` on ``sim``; returns a JSON string."""
    args = dict(args or {})
    try:
        if name == "get_available_assets":
            limit = int(args.pop("limit", 50))
            rows = sim.get_available_assets(**args)
            result: Any = {"count": len(rows), "assets": rows[:limit]}
        elif name == "get_price":
            result = sim.get_price(args["symbol"])
        elif name == "get_portfolio":
            result = sim.get_portfolio()
        elif name == "get_historical_data":
            df = sim.get_historical_data(args["symbol"], args.get("timeframe"), int(args.get("limit", 30)),
                                         bool(args.get("indicators", False)))
            records = [{"t": ts.isoformat(), **{k: _round(v) for k, v in row.items()}}
                       for ts, row in df.iterrows()]
            result = {"symbol": args["symbol"], "bars": records}
        elif name == "detect_patterns":
            result = sim.detect_patterns(args["symbol"], int(args.get("limit", 300)), args.get("timeframe")).to_dict()
        elif name == "place_order":
            result = sim.place_order(**args)
        elif name == "cancel_order":
            result = sim.cancel_order(args["order_id"])
        elif name == "get_open_orders":
            result = sim.get_open_orders(args.get("symbol"))
        elif name == "close_position":
            result = sim.close_position(args["symbol"]) or {"message": "no position"}
        elif name == "get_trades":
            result = sim.get_trades(int(args.get("limit", 20)))
        elif name == "get_trade_history":
            rows = sim.get_trade_history(args.get("kind", "fills"), args.get("symbol"))
            result = rows[-int(args.get("limit", 50)):]
        elif name == "get_metrics":
            result = {k: _round(v) for k, v in sim.get_metrics().items()}
        else:
            result = {"error": f"unknown tool {name!r}"}
    except Exception as exc:  # report to the model instead of crashing the agent loop
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return dumps(result)


def parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    return raw if isinstance(raw, dict) else json.loads(raw)
