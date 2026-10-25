"""Honest validation of the agents (nightly) and the gate a change must pass before it goes live.

    python scripts/validate.py --cv --backtests                          # live universe, 5 years
    python scripts/validate.py --cv --universe all --max-symbols 600     # learner CV on the big universe
    python scripts/validate.py --gate --baseline-ref origin/main --years 2  # exit 1 if a change does not improve

``--cv``: purged k-fold cross-validation of the learner's models (rank IC out of sample, weekly samples).
``--backtests``: walk-forward backtest of every rule-based live account after costs, against buy & hold.
``--gate``: the accounts changed since ``--baseline-ref`` (config, or every account if agent code changed) are
backtested in both versions; each must beat the current version and buy & hold out of sample.

Results go to ``live/validation.json``. Bars are cached in ``.cache/bars`` for a day. Note the survivorship bias
(see :data:`tradesim.validation.SURVIVORSHIP_WARNING`).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesim.assets.universe import AssetUniverse  # noqa: E402
from tradesim.trading.config import AccountSpec, LiveConfig  # noqa: E402
from tradesim.validation import SURVIVORSHIP_WARNING, backtest, cv_models, gate  # noqa: E402

log = logging.getLogger("validate")
SKIP = {"DiscretionaryAgent", "ClaudePortfolioManager", "HedgeMetaAgent"}  # LLM / needs the other live accounts


def load_bars(symbols: list[str], universe: AssetUniverse, start: datetime, end: datetime, cache: Path
              ) -> dict[str, pd.DataFrame]:
    from tradesim.trading.provider import default_live_provider

    cache.mkdir(parents=True, exist_ok=True)
    provider = None
    out = {}
    for i, sym in enumerate(symbols):
        f = cache / f"{sym.replace('/', '_')}.pkl"
        if f.exists() and time.time() - f.stat().st_mtime < 20 * 3600:
            df = pd.read_pickle(f)
        else:
            provider = provider or default_live_provider()
            try:
                df = provider.get_bars(universe.get(sym), "1d", start, end)
            except Exception as exc:  # noqa: BLE001 - delisted / no data: skip
                log.warning("%s: %s", sym, str(exc)[:120])
                continue
            df.to_pickle(f)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if len(df) > 120:
            out[sym] = df
        if i % 100 == 99:
            print(f"  bars: {i + 1}/{len(symbols)}", flush=True)
    return out


def run_cv(frames: dict[str, pd.DataFrame], universe: AssetUniverse, horizon: int) -> dict:
    from tradesim.agents.learning import build_panel, default_groups, demean_targets

    groups, markets = default_groups(list(frames), universe)
    samples = build_panel(frames, groups, markets, horizon=horizon, every_days=7)  # weekly: no overlapping outcomes
    demean_targets(samples)
    res = cv_models(samples, horizon, k=5, embargo_days=7 * max(1, horizon // 5 + 1))
    res["symbols"] = len(frames)
    return res


def run_backtests(specs: list[AccountSpec], cfg: LiveConfig, frames: dict, universe: AssetUniverse,
                  start: datetime, end: datetime) -> dict:
    out = {}
    for spec in specs:
        t = time.time()
        stop = (spec.stop_loss_atr or cfg.risk.stop_loss_atr) if spec.auto_stop else None
        try:
            out[spec.id] = backtest(spec.build_agent, frames, universe, start, end, fees=spec.fees,
                                    initial_cash=spec.initial_cash, max_position_pct=cfg.risk.max_position_pct,
                                    auto_stop_atr=stop)
        except Exception as exc:  # noqa: BLE001
            out[spec.id] = {"error": str(exc)[:300]}
        out[spec.id]["secs"] = round(time.time() - t, 1)
        print(f"  {spec.id:16s} {out[spec.id]}", flush=True)
    return out


def _spec_key(a: dict) -> str:
    return json.dumps({k: a.get(k) for k in ("agent", "params", "auto_stop", "stop_loss_atr", "fees")}, sort_keys=True)


def changed_accounts(cfg_now: dict, ref: str) -> tuple[list[str], dict | None, bool]:
    """Accounts whose definition changed since ``ref``, the baseline config, and whether agent code changed."""
    try:
        base = json.loads(subprocess.run(["git", "show", f"{ref}:live/config.json"], cwd=ROOT, check=True,
                                         capture_output=True, text=True).stdout)
    except subprocess.CalledProcessError:
        return [a["id"] for a in cfg_now["accounts"]], None, True
    code = subprocess.run(["git", "diff", "--quiet", ref, "--", "tradesim/agents", "tradesim/simulator.py",
                           "tradesim/patterns.py"], cwd=ROOT).returncode != 0
    old = {a["id"]: _spec_key(a) for a in base["accounts"]}
    ids = [a["id"] for a in cfg_now["accounts"] if code or old.get(a["id"]) != _spec_key(a)]
    return ids, base, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["live", "all"], default="live")
    ap.add_argument("--max-symbols", type=int, default=600, help="random sample of the big universe (0 = all)")
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--backtests", action="store_true")
    ap.add_argument("--agents", default="", help="comma-separated account ids (default: every rule-based one)")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--baseline-ref", default="origin/main")
    ap.add_argument("--config", default=str(ROOT / "live" / "config.json"))
    ap.add_argument("--out", default=str(ROOT / "live" / "validation.json"))
    ap.add_argument("--cache", default=str(ROOT / ".cache" / "bars"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = LiveConfig.load(args.config)
    universe = AssetUniverse.default()
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(365 * args.years))
    history_from = start - timedelta(days=1200)  # warm-up / pre-training history before the tested period
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "period": [start.date().isoformat(),
                    end.date().isoformat()], "survivorship_warning": SURVIVORSHIP_WARNING}
    live_frames = None
    if args.backtests or args.gate or (args.cv and args.universe == "live"):
        print(f"loading {len(cfg.symbols)} live symbols", flush=True)
        live_frames = load_bars(list(cfg.symbols), universe, history_from, end, Path(args.cache))
    if args.cv:
        if args.universe == "all":
            pool = [a.symbol for a in universe if a.asset_class.value in ("equity", "etf", "crypto")]
            random.Random(0).shuffle(pool)
            syms = list(dict.fromkeys(list(cfg.symbols) + pool))[: args.max_symbols or None]
            print(f"loading {len(syms)} symbols for the CV", flush=True)
            frames = load_bars(syms, universe, start - timedelta(days=120), end, Path(args.cache))
        else:
            frames = {s: f[f.index >= pd.Timestamp(start - timedelta(days=120))] for s, f in live_frames.items()}
        report["cv"] = run_cv(frames, universe, args.horizon)
        print("CV:", json.dumps(report["cv"]), flush=True)
    wanted = {x for x in args.agents.split(",") if x}
    specs = [a for a in cfg.accounts if a.enabled and a.agent not in SKIP and (not wanted or a.id in wanted)]
    if args.backtests:
        report["backtests"] = run_backtests(specs, cfg, live_frames, universe, start, end)
    status = 0
    if args.gate:
        raw = json.loads(Path(args.config).read_text())
        ids, base, code_changed = changed_accounts(raw, args.baseline_ref)
        todo = [s for s in specs if s.id in ids]
        cand = run_backtests(todo, cfg, live_frames, universe, start, end)
        baseline: dict = {}
        if base is not None and code_changed:  # old code: run the baseline's own script in a worktree
            with tempfile.TemporaryDirectory() as tmp:
                wt = Path(tmp) / "base"
                subprocess.run(["git", "worktree", "add", "--detach", str(wt), args.baseline_ref], cwd=ROOT, check=True)
                try:
                    if (wt / "scripts" / "validate.py").exists():
                        out = Path(tmp) / "base.json"
                        subprocess.run([sys.executable, "scripts/validate.py", "--backtests", "--years", str(args.years),
                                        "--agents", ",".join(s.id for s in todo), "--out", str(out),
                                        "--cache", args.cache], cwd=wt, check=False)
                        if out.exists():
                            baseline = json.loads(out.read_text()).get("backtests", {})
                finally:
                    subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT, check=False)
        elif base is not None:  # same code, previous settings
            old = [AccountSpec(**a) for a in base["accounts"] if a["id"] in {s.id for s in todo}]
            baseline = run_backtests([s for s in old if s.agent not in SKIP], cfg, live_frames, universe, start, end)
        report["gate"] = {sid: {"candidate": cand.get(sid), "baseline": baseline.get(sid),
                                **gate(cand.get(sid) or {}, baseline.get(sid))} for sid in cand}
        for sid, g in report["gate"].items():
            print(f"  gate {sid}: {'PASA' if g['pass'] else 'NO PASA'} {[(c['check'], c['ok']) for c in g['checks']]}")
        status = 0 if all(g["pass"] for g in report["gate"].values()) else 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
