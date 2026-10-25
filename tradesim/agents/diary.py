"""Helpers for the agents' own entries in the daily chronicle (first person, Spanish like the dashboard).

Every agent answers :meth:`tradesim.agents.base.Agent.diary` after each live decision with ``que_hice`` (what I did),
``por_que`` (why) and ``segun_situacion`` (how my behaviour relates to the market situation of the day).
"""

from __future__ import annotations

from typing import Any

DONE = ("filled", "partially_filled")
CLOSED = DONE + ("rejected", "canceled", "cancelled", "expired")


def did(orders: list[dict[str, Any]]) -> str:
    """What the orders of a decision ended up doing, in one sentence."""
    filled = [o for o in orders if o.get("status") in DONE]
    buys = sorted({o["symbol"] for o in filled if o.get("side") == "buy"})
    sells = sorted({o["symbol"] for o in filled if o.get("side") == "sell"})
    waiting = [o for o in orders if o.get("status") not in CLOSED]
    rejected = [o for o in orders if o.get("status") == "rejected"]
    parts = []
    if buys:
        parts.append("compré " + ", ".join(buys))
    if sells:
        parts.append("vendí " + ", ".join(sells))
    if waiting:
        parts.append(f"dejé {len(waiting)} órdenes esperando a que abra su mercado")
    if rejected:
        parts.append(f"me rechazaron {len(rejected)} órdenes ({(rejected[0].get('reject_reason') or '')[:80]})")
    if not parts:
        return "No hice ninguna operación."
    text = "; ".join(parts) + "."
    return text[0].upper() + text[1:]


def tag(situation: dict[str, Any]) -> str:
    return f"«{situation.get('tag') or 'sin referencia'}»"


def trend(situation: dict[str, Any]) -> str | None:
    return situation.get("trend")


def entry(que: str, por: str, segun: str) -> dict[str, str]:
    return {"que_hice": que, "por_que": por.strip(), "segun_situacion": segun.strip()}


def default(agent: Any, situation: dict[str, Any], orders: list[dict[str, Any]]) -> dict[str, str]:
    ex = agent.explain() or {}
    why = ex.get("signals") or ex.get("reasons") or []
    return entry(did(orders), "; ".join(str(x) for x in why)[:400] or "No tenía señales para operar.",
                 f"Mi regla no depende de la situación del mercado ({tag(situation)}).")
