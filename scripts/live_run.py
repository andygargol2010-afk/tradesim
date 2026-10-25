"""Run the live accounts once (called by the scheduler; safe to run by hand).

    python scripts/live_run.py --mode decide     # sync + daily decision (once per day, markets open)
    python scripts/live_run.py --mode mark       # sync + mark to market only
    python scripts/live_run.py --mode research   # news / macro / calendar / patterns refresh, no trading
    python scripts/live_run.py --export-only     # just rebuild dashboard/live.json from the saved state
    python scripts/live_run.py --chronicle 2026-10-26  # write / rewrite that day's entry of the daily chronicle

Exit code 1 if any account failed (its state and the others' are still saved).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesim.trading import LiveConfig, LiveRunner  # noqa: E402
from tradesim.trading.export import export_dashboard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["decide", "mark", "research"], default="mark")
    ap.add_argument("--config", default=str(ROOT / "live" / "config.json"))
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--chronicle", metavar="YYYY-MM-DD", help="write the chronicle entry of that UTC day and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = LiveConfig.load(args.config).resolve(ROOT)
    if args.chronicle:
        from datetime import date

        from tradesim.trading.chronicle import write_entry

        entry = write_entry(config, date.fromisoformat(args.chronicle))
        print(f"chronicle {entry['date']} written with {entry['engine']}: {entry['text'].get('titulo')}")
        return 0
    if args.export_only:
        export_dashboard(config)
        print(f"exported {config.dashboard_file}")
        return 0
    record = LiveRunner(config).run(args.mode)
    print(f"{record['mode']} run at {record['started_at']} took {record['duration_s']}s")
    if record.get("chronicle"):
        print("  chronicle:", record["chronicle"])
    if record.get("research"):
        print("  research:", record["research"])
    for sym, p in record["prices"].items():
        print(f"  {sym:8s} " + (f"{p['last']:>12.4f}  ({p.get('source')})" if "last" in p else f"ERROR {p['error'][:80]}"))
    failed = False
    for acc, r in record["accounts"].items():
        print(f"  [{r['status']:>8s}] {acc:16s} equity {r.get('equity', float('nan')):>12,.2f}  {r.get('message', '')}")
        failed |= r["status"] == "error"
    if record["data_errors"]:
        print("  data errors:", record["data_errors"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
