"""Safe changes: shadow accounts, statistical promotion and automatic rollback.

* A *shadow* account (``"shadow_of": "<account id>"`` in ``live/config.json``) runs a new version of an agent in
  parallel with real prices and its own paper money, without being shown as a competitor or used by the meta-agents.
  After ``min_days`` it replaces the account it shadows only if it is better with statistical evidence: paired
  daily-return difference with t > ``min_t`` (so a better Sharpe is not luck) and, when both measure it, a higher
  rank IC. The account keeps its money and history; it adopts the shadow's agent, parameters and learned state.
* *Checkpoints*: every week the learned state of each agent is saved in its account folder (and so in git). If an
  agent's rank IC over the last 20 days turns negative, the newest checkpoint from before that period whose IC was
  positive is restored.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradesim.persistence import read_json, write_json


def daily_returns(journal: Path) -> pd.Series:
    """Daily returns from the equity the account had at each daily decision (journal)."""
    eq = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = e.get("equity_after", e.get("equity"))
            if v and e.get("time"):
                eq[e["time"][:10]] = float(v)
    s = pd.Series(eq, dtype=float).sort_index()
    return s.pct_change().dropna()


def compare(primary: pd.Series, shadow: pd.Series, min_days: int = 20, min_t: float = 2.0) -> dict[str, Any]:
    """Paired test of daily returns (shadow - primary) over their common days."""
    common = primary.index.intersection(shadow.index)
    d = (shadow[common] - primary[common]).to_numpy(dtype=float)
    n = len(d)
    sharpe = lambda r: float(r.mean() / r.std() * math.sqrt(252)) if len(r) > 2 and r.std() > 0 else None  # noqa: E731
    out: dict[str, Any] = {"days": n, "sharpe_primary": sharpe(primary[common]), "sharpe_shadow": sharpe(shadow[common])}
    if n < 3 or d.std(ddof=1) == 0:
        return {**out, "t": None, "better": False}
    t = float(d.mean() / d.std(ddof=1) * math.sqrt(n))
    better = n >= min_days and t > min_t and (out["sharpe_shadow"] or -9) > (out["sharpe_primary"] or -9)
    return {**out, "mean_diff_pct": round(float(d.mean()) * 100, 4), "t": round(t, 2), "better": better}


def promote(config_path: Path, primary_id: str, shadow_id: str, state_dir: Path) -> None:
    """The primary account adopts the shadow's agent definition and learned state; the shadow is removed."""
    raw = json.loads(config_path.read_text())
    accounts = raw["accounts"]
    sh = next(a for a in accounts if a["id"] == shadow_id)
    pr = next(a for a in accounts if a["id"] == primary_id)
    for k in ("agent", "params", "auto_stop", "stop_loss_atr"):
        if k in sh:
            pr[k] = sh[k]
    pr.setdefault("history", []).append({"promoted_from": shadow_id, "at": datetime.now().isoformat()[:10]})
    raw["accounts"] = [a for a in accounts if a["id"] != shadow_id]
    config_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    src = state_dir / shadow_id / "agent.json"
    if src.exists():
        shutil.copy(src, state_dir / primary_id / "agent.json")


def checkpoint(account_dir: Path, now: datetime, ic: float | None, every_days: int = 7, keep: int = 8) -> bool:
    """Save the learned state weekly (``checkpoints/<date>.json``) with the IC it had at the time."""
    src = account_dir / "agent.json"
    if not src.exists():
        return False
    folder = account_dir / "checkpoints"
    folder.mkdir(exist_ok=True)
    existing = sorted(folder.glob("*.json"))
    if existing and (now.date() - datetime.fromisoformat(existing[-1].stem).date()).days < every_days:
        return False
    data = read_json(src)
    data["checkpoint_ic"] = ic
    write_json(folder / f"{now.date().isoformat()}.json", data)
    for old in sorted(folder.glob("*.json"))[:-keep]:
        old.unlink()
    return True


def rollback(account_dir: Path, now: datetime, recent_ic: float | None, meta: dict[str, Any],
             window_days: int = 20) -> str | None:
    """Restore the newest checkpoint older than ``window_days`` with a positive IC when the recent IC is negative
    (at most once per window). Returns the restored checkpoint date, if any."""
    if recent_ic is None or recent_ic >= 0:
        return None
    last = meta.get("rolled_back_at")
    if last and (now - datetime.fromisoformat(last)).days < window_days:
        return None
    cutoff = (now - timedelta(days=window_days)).date().isoformat()
    for cp in sorted((account_dir / "checkpoints").glob("*.json"), reverse=True):
        if cp.stem <= cutoff and (read_json(cp).get("checkpoint_ic") or 0) > 0:
            data = read_json(cp)
            data.pop("checkpoint_ic", None)
            write_json(account_dir / "agent.json", data)
            meta["rolled_back_at"] = now.isoformat()
            meta.setdefault("rollbacks", []).append({"at": now.isoformat(), "checkpoint": cp.stem,
                                                     "recent_ic": round(recent_ic, 4)})
            return cp.stem
    return None


def recent_ic(agent: Any, days: int = 20) -> float | None:
    """Mean daily rank IC of an agent over its last ``days`` sample dates (``None`` if it does not measure IC)."""
    hist = getattr(agent, "history", None)
    if not hist or not hasattr(agent, "ic_stats"):
        return None
    from tradesim.agents.learning import rank_ic_by_day

    rows = [(r[2], r[0], r[1]) if len(r) >= 3 and isinstance(r[2], str) else (r[0], r[1], r[2]) for r in hist]
    ics = rank_ic_by_day(rows, every=1, window=days)
    return float(np.mean(ics)) if len(ics) >= 5 else None
