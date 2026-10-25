import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from tradesim.agents.stacking import STACK_FEATURES, SignalStackingAgent
from tradesim.trading.promotion import checkpoint, compare, daily_returns, promote, rollback

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _journal(path, equities):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"time": (T0 + timedelta(days=i)).isoformat(), "equity_after": e})
                              for i, e in enumerate(equities)) + "\n")


def test_shadow_promoted_only_with_evidence(tmp_path):
    rng = np.random.default_rng(0)
    base = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, 60))
    better = base * np.cumprod(1 + rng.normal(0.004, 0.002, 60))  # consistently ahead
    _journal(tmp_path / "acc" / "a" / "journal.jsonl", base)
    _journal(tmp_path / "acc" / "a2" / "journal.jsonl", better)
    _journal(tmp_path / "acc" / "noise" / "journal.jsonl", base * np.cumprod(1 + rng.normal(0, 0.01, 60)))
    pr = daily_returns(tmp_path / "acc" / "a" / "journal.jsonl")
    assert compare(pr, daily_returns(tmp_path / "acc" / "a2" / "journal.jsonl"))["better"]
    assert not compare(pr, daily_returns(tmp_path / "acc" / "noise" / "journal.jsonl"))["better"]
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"accounts": [{"id": "a", "agent": "X", "params": {"k": 1}},
                                            {"id": "a2", "agent": "Y", "params": {"k": 2}, "shadow_of": "a"}]}))
    (tmp_path / "acc" / "a2" / "agent.json").write_text('{"state": {"learned": 2}}')
    promote(cfg, "a", "a2", tmp_path / "acc")
    accs = json.loads(cfg.read_text())["accounts"]
    assert [a["id"] for a in accs] == ["a"] and accs[0]["agent"] == "Y" and accs[0]["params"] == {"k": 2}
    assert json.loads((tmp_path / "acc" / "a" / "agent.json").read_text())["state"] == {"learned": 2}


def test_checkpoint_and_rollback(tmp_path):
    acc = tmp_path / "a"
    acc.mkdir()
    (acc / "agent.json").write_text('{"state": {"v": 1}}')
    assert checkpoint(acc, T0, 0.05)
    assert not checkpoint(acc, T0 + timedelta(days=3), 0.05)  # weekly
    (acc / "agent.json").write_text('{"state": {"v": 2}}')
    meta: dict = {}
    assert rollback(acc, T0 + timedelta(days=30), 0.02, meta) is None  # IC still positive: nothing to do
    assert rollback(acc, T0 + timedelta(days=30), -0.03, meta) == T0.date().isoformat()
    assert json.loads((acc / "agent.json").read_text())["state"] == {"v": 1} and meta["rollbacks"]
    assert rollback(acc, T0 + timedelta(days=35), -0.03, meta) is None  # once per window


class _Obs(SimpleNamespace):
    def closes(self, s, n):
        return self.hist[s][-n:]

    def history(self, s, limit=100):
        import pandas as pd

        c = self.hist[s][-limit:]
        return pd.DataFrame({"close": c}, index=pd.date_range(end=self.timestamp, periods=len(c), freq="D"))

    def weight(self, s):
        return 0.0

    def position(self, s):
        return 0


def test_stacking_learns_which_signal_works():
    rng = np.random.default_rng(1)
    syms = [f"A{k}" for k in range(10)]
    hist = {s: list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))) for s in syms}
    agent = SignalStackingAgent(min_samples=200, ic_min_days=10)
    for d in range(160):
        preds = {s: float(rng.normal()) for s in syms}
        rsi = {s: float(rng.uniform(20, 80)) for s in syms}  # noise
        agent.attach_peers({"aprendiz": {"explain": {"predictions": preds, "regime": 0}},
                            "rsi-adaptativo": {"explain": {"rsi": rsi}}})
        obs = _Obs(symbols=syms, prices={s: hist[s][-1] for s in syms}, timestamp=T0 + timedelta(days=d),
                   positions={}, open_orders=0, tradable=None, equity=100.0, hist={s: np.array(h) for s, h in hist.items()},
                   _sim=None, exposure_cap=1.0)
        agent.act(obs)
        for s in syms:  # the learner's prediction carries the edge
            hist[s].append(hist[s][-1] * float(np.exp(0.004 * preds[s] + rng.normal(0, 0.006))))
    w = dict(zip(STACK_FEATURES, agent.model.w))
    assert w["learner_pred"] > 3 * abs(w["rsi_dev"]) and agent.ic_stats()["rank_ic_mean"] > 0.05
