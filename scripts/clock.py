"""Self-sustaining scheduler for the live workflow (GitHub's ``schedule`` trigger proved unreliable).

Runs inside ``.github/workflows/clock.yml``: sleeps until each time in ``live/config.json`` -> ``schedule``,
dispatches ``live-trading.yml`` with that entry's mode, and after ``--hours`` hands over to a fresh clock run
(right after a dispatch, so no slot is missed). Needs ``gh`` and ``GH_TOKEN`` with ``actions: write``.

    python scripts/clock.py --hours 5.5           # in the clock workflow
    python scripts/clock.py --dry-run --hours 24  # print the next slots
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _field(spec: str, value: int) -> bool:
    if spec == "*":
        return True
    for part in spec.split(","):
        if part.startswith("*/"):
            if value % int(part[2:]) == 0:
                return True
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
            if a <= value <= b:
                return True
        elif int(part) == value:
            return True
    return False


def matches(cron: str, t: datetime) -> bool:
    """5-field cron (minute hour day month weekday; weekday 0 = Sunday), UTC."""
    m, h, dom, mon, dow = cron.split()
    return (_field(m, t.minute) and _field(h, t.hour) and _field(dom, t.day) and _field(mon, t.month)
            and _field(dow, (t.weekday() + 1) % 7))


def next_slot(schedule: list[dict], after: datetime) -> tuple[datetime, list[str]]:
    """First minute strictly after ``after`` matching any entry, with the modes due then."""
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(8 * 24 * 60):
        modes = [e["mode"] for e in schedule if matches(e["cron"], t)]
        if modes:
            return t, modes
        t += timedelta(minutes=1)
    raise RuntimeError("no schedule slot in the next 8 days")


def dispatch(workflow: str, mode: str | None, dry: bool) -> None:
    cmd = ["gh", "workflow", "run", workflow, "--ref", "main"] + (["-f", f"mode={mode}"] if mode else [])
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {' '.join(cmd)}", flush=True)
    if not dry:
        subprocess.run(cmd, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.5, help="hand over to a new clock run after this long")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    schedule = json.loads((ROOT / "live" / "config.json").read_text())["schedule"]
    start = datetime.now(timezone.utc)
    now = start
    while True:
        slot, modes = next_slot(schedule, now)
        if args.dry_run:
            print(slot.isoformat(), modes)
        else:
            time.sleep(max(0.0, (slot - datetime.now(timezone.utc)).total_seconds()))
        # several entries in the same minute: the most complete run wins (decide includes research + mark)
        mode = next(m for m in ("decide", "research", "mark") if m in modes)
        dispatch("live-trading.yml", mode, args.dry_run)
        now = slot
        if (now - start) >= timedelta(hours=args.hours):
            dispatch("clock.yml", None, args.dry_run)  # successor; queued until this run ends
            return


if __name__ == "__main__":
    main()
