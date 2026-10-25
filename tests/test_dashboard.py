import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_dashboard_data(tmp_path):
    out = tmp_path / "data.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dashboard.py"), "--episodes", "2", "--length", "30",
                    "--symbols", "SPY", "AAPL", "BTC-USD", "--out", str(out)], check=True, capture_output=True)
    data = json.loads(out.read_text())  # strict JSON: no NaN / Infinity literals
    assert data["meta"]["n_episodes"] == 2 and data["meta"]["synthetic"] is True
    assert len(data["benchmark_path"]["mean"]) == 31
    starts = None
    for agent in data["agents"]:
        assert {"total_return_pct", "sharpe", "best_episode", "pct_beat_benchmark"} <= set(agent["summary"])
        eps = agent["episodes"]
        assert len(eps) == 2
        # every agent faces the same episodes
        assert starts in (None, [e["start"] for e in eps])
        starts = [e["start"] for e in eps]
        for e in eps:
            assert len(e["t"]) == len(e["equity"]) == len(e["benchmark"])
            for f in e["fills"]:
                assert f["symbol"] in e["prices"] or len(e["prices"]) == 6


def test_dashboard_files_exist():
    for name in ("index.html", "live.js", "backtest.html", "backtest.js", "charts.js", "styles.css"):
        assert (ROOT / "dashboard" / name).exists()
    cfg = json.loads((ROOT / "vercel.json").read_text())
    assert cfg["outputDirectory"] == "dashboard"
