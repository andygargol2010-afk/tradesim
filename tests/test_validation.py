from test_learning import mean_reverting_frames
from tradesim.agents.learning import build_panel, default_groups, demean_targets
from tradesim.validation import cv_models, gate, purged_kfold


def _samples():
    frames = {s: f["close"] for s, f in mean_reverting_frames(n_symbols=8, n=700, seed=2).items()}
    groups, markets = default_groups(list(frames))
    samples = build_panel(frames, groups, markets, horizon=1)
    demean_targets(samples)
    return samples


def test_purged_kfold_never_trains_on_overlapping_outcomes():
    samples = _samples()
    folds = list(purged_kfold(samples, k=5, embargo_days=3))
    assert len(folds) == 5
    for train, test in folds:
        lo = min(samples[i]["ts"][:10] for i in test)
        hi = max(samples[i]["ts"][:10] for i in test)
        for i in train:
            s = samples[i]
            assert s["realized"][:10] < lo or s["ts"][:10] > hi  # outcome window never touches the test block


def test_cv_scores_models_out_of_sample_and_gate():
    res = cv_models(_samples(), horizon=1, k=4, embargo_days=2)
    assert res["folds"] == 4 and res["linear"]["rank_ic_mean"] > 0.05 and res["linear"]["rank_ic_t"] > 2
    good = {"sharpe": 1.2, "buy_hold_sharpe": 0.8, "rank_ic": {"rank_ic_mean": 0.03}}
    old = {"sharpe": 1.0, "rank_ic": {"rank_ic_mean": 0.02}}
    assert gate(good, old)["pass"]
    assert not gate({**good, "sharpe": 0.9}, old)["pass"]  # worse than the current version
    assert not gate({"sharpe": 0.5, "buy_hold_sharpe": 0.8}, None)["pass"]  # worse than buy & hold
