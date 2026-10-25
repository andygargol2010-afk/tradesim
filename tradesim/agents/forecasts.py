"""What the discretionary analyst learns from its own track record.

* :class:`SignalLearner`: every decision logs each asset's technical, news and trend scores; once the outcome is
  known (``horizon_days`` later) an online regression learns how much each signal is really worth. The learned
  weights replace the fixed ``0.55 / 0.30 / 0.15`` of the local engine only after they prove better out of sample
  (rank IC with t >= ``min_t``).
* :class:`ForecastBook`: every position comes with a structured forecast (probability of going up, horizon,
  expected return). When the horizon is over the forecast is scored (Brier score) and aggregated by conviction,
  regime and chart pattern. Lessons are derived from these statistics and kept only when the data back them
  (``n >= 30`` and ``|t| > 2``); free-text lessons nobody checks are no longer stored.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from tradesim.agents.learning import RecursiveLeastSquares, _ic_summary, rank_ic_by_day

SIGNALS = ["tech", "news", "trend"]
DEFAULT_WEIGHTS = {"tech": 0.55, "news": 0.30, "trend": 0.15}
CONVICTION_ES = {"high": "alta", "medium": "media", "low": "baja"}
REGIME_ES = {"risk_on": "apetito por el riesgo", "neutral": "neutral", "risk_off": "aversión al riesgo"}


def _days(a: str, b: datetime) -> float:
    return (b - datetime.fromisoformat(a)).total_seconds() / 86400


class SignalLearner:
    def __init__(self, horizon_days: int = 7, min_samples: int = 500, min_t: float = 2.0, keep: int = 6000):
        self.horizon_days, self.min_samples, self.min_t, self.keep = horizon_days, min_samples, min_t, keep
        self.model = RecursiveLeastSquares(1 + len(SIGNALS), forgetting=0.9995, ridge=10.0)
        self.pending: list[dict[str, Any]] = []
        self.history: list[list[Any]] = []  # [date, fixed-weight score, learned prediction, relative outcome]

    def record(self, now: datetime, signals: dict[str, dict[str, float]], prices: dict[str, float],
               sigmas: dict[str, float]) -> None:
        day = now.date().isoformat()
        if any(p["date"][:10] == day for p in self.pending):
            return  # one sample per asset and day
        for sym, s in signals.items():
            px, sg = prices.get(sym), sigmas.get(sym)
            if not (px and px > 0 and sg and sg > 0):
                continue
            x = [1.0] + [float(s[k]) for k in SIGNALS]
            self.pending.append({"date": now.isoformat(), "sym": sym, "x": x, "price": px, "sigma": sg,
                                 "fixed": sum(DEFAULT_WEIGHTS[k] * s[k] for k in SIGNALS),
                                 "learned": self.model.predict(np.array(x))})

    def resolve(self, now: datetime, prices: dict[str, float]) -> int:
        """Learn from samples ``horizon_days`` old: return relative to the other assets of the same day."""
        due = defaultdict(list)
        waiting = []
        for p in self.pending:
            px = prices.get(p["sym"])
            if _days(p["date"], now) >= self.horizon_days and px and px > 0:
                y = math.log(px / p["price"]) / (p["sigma"] * math.sqrt(self.horizon_days * 5 / 7))
                due[p["date"][:10]].append((p, float(np.clip(y, -6, 6))))
            elif _days(p["date"], now) < 3 * self.horizon_days:
                waiting.append(p)
        self.pending = waiting
        n = 0
        for day in sorted(due):
            group = due[day]
            mu = float(np.mean([y for _, y in group])) if len(group) >= 5 else 0.0
            for p, y in group:
                self.model.update(np.array(p["x"]), y - mu)
                self.history.append([day, round(p["fixed"], 4), round(p["learned"], 4), round(y - mu, 4)])
                n += 1
        del self.history[:-self.keep]
        return n

    def ic(self) -> dict[str, dict[str, Any]]:
        every = max(1, round(self.horizon_days * 5 / 7))
        return {name: _ic_summary(rank_ic_by_day([(r[0], r[col], r[3]) for r in self.history], every=every,
                                                 window=60))
                for name, col in (("fixed", 1), ("learned", 2))}

    def weights(self) -> tuple[dict[str, float], str]:
        """Weights for the local engine: learned (normalised to sum 1 in absolute value) once they beat the fixed
        ones out of sample with a significant rank IC; the fixed defaults otherwise."""
        st = self.ic()
        learned, fixed = st["learned"], st["fixed"]
        w = self.model.w[1:]
        if (self.model.n >= self.min_samples and (learned["rank_ic_t"] or 0) >= self.min_t
                and (learned["rank_ic_mean"] or 0) > (fixed["rank_ic_mean"] or 0) and np.abs(w).sum() > 0):
            return {k: float(v / np.abs(w).sum()) for k, v in zip(SIGNALS, w)}, "aprendidos"
        return dict(DEFAULT_WEIGHTS), "fijos"

    def lessons(self) -> list[str]:
        st = self.ic()
        out = []
        for name, label in (("fixed", "Los pesos fijos"), ("learned", "Los pesos aprendidos")):
            s = st[name]
            if s["days"] >= 30 and s["rank_ic_t"] is not None and abs(s["rank_ic_t"]) > 2:
                out.append(f"{label} de técnico/noticias/tendencia {'sí' if s['rank_ic_mean'] > 0 else 'no'} "
                           f"ordenan bien los activos (IC {s['rank_ic_mean']:+.3f}, t {s['rank_ic_t']:.1f}, "
                           f"{s['days']} días).")
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "pending": self.pending, "history": self.history}

    def load_state_dict(self, d: dict[str, Any] | None) -> None:
        if not d:
            return
        self.model = RecursiveLeastSquares.from_state(d["model"])
        self.pending, self.history = d.get("pending", []), d.get("history", [])


def _bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rets = np.array([r["ret"] for r in rows], dtype=float)
    hit = float(np.mean([r["up"] for r in rows]))
    prob = float(np.mean([r["prob"] for r in rows]))
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    se_hit = math.sqrt(max(hit * (1 - hit), 1e-9) / n)
    return {"n": n, "hit_rate": round(hit, 3), "mean_prob": round(prob, 3),
            "brier": round(float(np.mean([r["brier"] for r in rows])), 4),
            "mean_ret_pct": round(float(rets.mean()) * 100, 2),
            "t_ret": round(float(rets.mean() / sd * math.sqrt(n)), 2) if sd > 0 else None,
            "t_overconfidence": round((prob - hit) / se_hit, 2)}


class ForecastBook:
    def __init__(self, keep: int = 800, min_n: int = 30, min_t: float = 2.0):
        self.keep, self.min_n, self.min_t = keep, min_n, min_t
        self.open: list[dict[str, Any]] = []
        self.closed: list[dict[str, Any]] = []

    def record(self, now: datetime, positions: list[dict[str, Any]], prices: dict[str, float], regime: str | None,
               patterns: dict[str, list[str]]) -> int:
        """One forecast per asset at a time (a held position is not re-counted every day)."""
        busy = {f["sym"] for f in self.open}
        n = 0
        for p in positions:
            sym, px = p.get("symbol"), prices.get(p.get("symbol"))
            if sym in busy or not (px and px > 0) or p.get("prob_up") is None:
                continue
            self.open.append({"date": now.isoformat(), "sym": sym, "price": px,
                              "prob": float(np.clip(float(p["prob_up"]), 0.0, 1.0)),
                              "horizon": int(max(1, min(int(p.get("horizon_days") or 20), 120))),
                              "exp_ret_pct": float(p.get("expected_return_pct") or 0.0),
                              "conviction": p.get("conviction") or "medium", "regime": regime or "neutral",
                              "patterns": list(patterns.get(sym, []))[:4]})
            n += 1
        return n

    def resolve(self, now: datetime, prices: dict[str, float]) -> int:
        still, n = [], 0
        for f in self.open:
            px = prices.get(f["sym"])
            if _days(f["date"], now) >= f["horizon"] and px and px > 0:
                ret = px / f["price"] - 1
                up = ret > 0
                self.closed.append({**f, "closed": now.isoformat(), "ret": round(ret, 5), "up": up,
                                    "brier": round((f["prob"] - up) ** 2, 4)})
                n += 1
            else:
                still.append(f)
        self.open = still
        del self.closed[:-self.keep]
        return n

    def stats(self) -> dict[str, Any]:
        rows = self.closed
        if not rows:
            return {"n": 0}
        out: dict[str, Any] = {"overall": _bucket(rows), "open": len(self.open)}
        bins: dict[str, list] = defaultdict(list)
        for r in rows:
            lo = min(int(r["prob"] * 10), 9) / 10
            bins[f"{lo:.1f}-{lo + 0.1:.1f}"].append(r)
        out["calibration"] = {k: _bucket(v) for k, v in sorted(bins.items())}
        for key in ("conviction", "regime"):
            groups: dict[str, list] = defaultdict(list)
            for r in rows:
                groups[r[key]].append(r)
            out[f"by_{key}"] = {k: _bucket(v) for k, v in groups.items()}
        pats: dict[str, list] = defaultdict(list)
        for r in rows:
            for p in r.get("patterns") or []:
                pats[p].append(r)
        out["by_pattern"] = {k: _bucket(v) for k, v in sorted(pats.items(), key=lambda kv: -len(kv[1]))[:12]}
        return out

    def lessons(self) -> list[str]:
        """Only conclusions the data support: n >= min_n and |t| > min_t."""
        st = self.stats()
        if not st.get("n", 1):
            return []
        out = []
        o = st["overall"]
        if o["n"] >= self.min_n and abs(o["t_overconfidence"]) > self.min_t:
            out.append(f"{'Sobreestimo' if o['t_overconfidence'] > 0 else 'Subestimo'} mis probabilidades: predigo "
                       f"{o['mean_prob']:.0%} de subidas de media y aciertan el {o['hit_rate']:.0%} "
                       f"(n={o['n']}, t={o['t_overconfidence']:+.1f}).")
        labels = [("by_conviction", "Con convicción {}", CONVICTION_ES), ("by_regime", "En régimen de {}", REGIME_ES),
                  ("by_pattern", "Tras «{}»", {})]
        for key, fmt, names in labels:
            for k, b in st.get(key, {}).items():
                if b["n"] >= self.min_n and b["t_ret"] is not None and abs(b["t_ret"]) > self.min_t:
                    out.append(f"{fmt.format(names.get(k, k))} mis posiciones {'ganan' if b['t_ret'] > 0 else 'pierden'}"
                               f" de media {b['mean_ret_pct']:+.2f} % (acierto {b['hit_rate']:.0%}, n={b['n']}, "
                               f"t={b['t_ret']:+.1f}).")
        return out

    def summary_text(self) -> str:
        """Short track record for the prompt."""
        st = self.stats()
        if not st.get("n", 1):
            return "No forecast has reached its horizon yet."
        o = st["overall"]
        lines = [f"{o['n']} scored forecasts: Brier {o['brier']:.3f} (0.25 = coin flip), mean predicted "
                 f"P(up) {o['mean_prob']:.0%} vs realised {o['hit_rate']:.0%}, mean return {o['mean_ret_pct']:+.2f}%."]
        for key in ("by_conviction", "by_regime"):
            parts = [f"{k}: hit {b['hit_rate']:.0%} vs predicted {b['mean_prob']:.0%}, ret {b['mean_ret_pct']:+.2f}% "
                     f"(n={b['n']})" for k, b in st.get(key, {}).items()]
            if parts:
                lines.append(key.replace("by_", "By ") + ": " + "; ".join(parts))
        return "\n".join(lines)

    def state_dict(self) -> dict[str, Any]:
        return {"open": self.open, "closed": self.closed}

    def load_state_dict(self, d: dict[str, Any] | None) -> None:
        if d:
            self.open, self.closed = d.get("open", []), d.get("closed", [])
