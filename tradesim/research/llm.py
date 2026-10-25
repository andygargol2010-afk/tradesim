"""Thin layer over the Anthropic SDK shared by the research analysts and the discretionary agent.

* :func:`make_client` returns ``anthropic.Anthropic()`` when credentials are configured, else ``None``
  (every LLM feature then degrades gracefully: news stays untriaged, the agent holds).
* :func:`json_call` - one request whose answer must match a JSON schema (structured outputs).
* :func:`submit_call` - an agentic request that may use server tools (web search / web fetch) and
  must finish by calling a ``submit`` tool whose input is the result; handles ``pause_turn``.
* :class:`Usage` - token / search accounting with an approximate USD cost.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Cache reads are billed at ~0.1x input, writes at ~1.25x.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
WEB_SEARCH_USD = 10.0 / 1000  # per search

WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search"}
WEB_FETCH = {"type": "web_fetch_20260209", "name": "web_fetch"}


def make_client() -> Any | None:
    """An ``anthropic.Anthropic`` client, or ``None`` if the SDK or credentials are missing."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic is not installed: pip install anthropic")
        return None
    return anthropic.Anthropic(max_retries=4)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    """Accumulated usage per model."""

    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, usage: Any) -> None:
        if usage is None:
            return
        row = self.by_model.setdefault(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0,
                                               "cache_read_tokens": 0, "cache_write_tokens": 0, "web_searches": 0})
        row["requests"] += 1
        row["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        row["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        row["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        row["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        stu = getattr(usage, "server_tool_use", None)
        row["web_searches"] += (getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0

    def merge(self, other: "Usage") -> None:
        for model, row in other.by_model.items():
            mine = self.by_model.setdefault(model, {k: 0 for k in row})
            for k, v in row.items():
                mine[k] = mine.get(k, 0) + v

    def cost_usd(self) -> float:
        total = 0.0
        for model, r in self.by_model.items():
            pin, pout = PRICES.get(model, (4.0, 20.0))
            total += (r["input_tokens"] * pin + r["output_tokens"] * pout + r["cache_read_tokens"] * pin * 0.1
                      + r["cache_write_tokens"] * pin * 1.25) / 1e6
            total += r["web_searches"] * WEB_SEARCH_USD
        return round(total, 4)

    def to_dict(self) -> dict[str, Any]:
        return {"by_model": self.by_model, "cost_usd_estimate": self.cost_usd()}


def _text(message: Any) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")


def json_call(client: Any, model: str, system: str, user: str, schema: dict[str, Any], usage: Usage,
              max_tokens: int = 16000, effort: str | None = "low") -> dict[str, Any]:
    """One structured-output request; returns the parsed JSON object."""
    import anthropic

    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        output_config["effort"] = effort
    try:
        with client.messages.stream(model=model, max_tokens=max_tokens, cache_control={"type": "ephemeral"},
                                    system=system, messages=[{"role": "user", "content": user}],
                                    output_config=output_config) as stream:
            msg = stream.get_final_message()
    except anthropic.APIError as exc:
        raise LLMError(f"{model}: {exc}") from exc
    usage.add(model, msg.usage)
    if msg.stop_reason == "refusal":
        raise LLMError(f"{model} declined the request")
    if msg.stop_reason == "max_tokens":
        raise LLMError(f"{model} hit max_tokens ({max_tokens})")
    try:
        return json.loads(_text(msg))
    except json.JSONDecodeError as exc:
        raise LLMError(f"{model} returned invalid JSON: {exc}") from exc


def submit_call(client: Any, model: str, system: str, user: str, submit_tool: dict[str, Any], usage: Usage,
                server_tools: list[dict[str, Any]] | None = None, max_tokens: int = 32000, effort: str = "medium",
                max_rounds: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Agentic request: Claude may search/fetch the web, then must call ``submit_tool``.

    Returns ``(submit_input, sources)`` where ``sources`` lists the web results Claude looked at
    (``{"title", "url"}``). Raises :class:`LLMError` if no valid submission is produced.
    """
    import anthropic

    tool = {**submit_tool, "eager_input_streaming": True}
    tools = list(server_tools or []) + [tool]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    sources: list[dict[str, Any]] = []
    nudged = False
    for _ in range(max_rounds):
        try:
            with client.messages.stream(model=model, max_tokens=max_tokens, cache_control={"type": "ephemeral"},
                                        system=system, tools=tools, messages=messages,
                                        output_config={"effort": effort}) as stream:
                msg = stream.get_final_message()
        except ValueError as exc:  # unparseable streamed tool input
            raise LLMError(f"{model}: malformed tool input ({exc})") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"{model}: {exc}") from exc
        usage.add(model, msg.usage)
        sources += _sources(msg)
        if msg.stop_reason == "refusal":
            raise LLMError(f"{model} declined the request")
        call = next((b for b in msg.content if getattr(b, "type", "") == "tool_use" and b.name == submit_tool["name"]),
                    None)
        if msg.stop_reason == "max_tokens":
            raise LLMError(f"{model} hit max_tokens ({max_tokens}) before finishing")
        if call is not None:
            if not isinstance(call.input, dict):
                raise LLMError(f"{model}: submit input is not an object")
            return call.input, sources
        messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "pause_turn":
            continue  # server-side loop paused: re-send as is and it resumes
        if nudged:
            break
        nudged = True
        messages.append({"role": "user", "content": f"Now call {submit_tool['name']} with your final answer."})
    raise LLMError(f"{model} finished without calling {submit_tool['name']}")


def _sources(msg: Any) -> list[dict[str, Any]]:
    out = []
    for b in msg.content:
        if getattr(b, "type", "") == "web_search_tool_result":
            content = getattr(b, "content", None)
            if isinstance(content, list):
                for r in content:
                    url = getattr(r, "url", None)
                    if url:
                        out.append({"title": getattr(r, "title", "") or "", "url": url})
        elif getattr(b, "type", "") == "web_fetch_tool_result":
            content = getattr(b, "content", None)
            url = getattr(content, "url", None)
            if url:
                out.append({"title": "", "url": url})
    return out


def require_keys(obj: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise LLMError(f"{where}: missing {', '.join(missing)}")
