"""Technical pattern recognition on OHLCV bars (causal: only bars up to the last one are used).

:func:`detect_patterns` returns a :class:`PatternReport` for one asset: trend, support/resistance levels,
and the patterns active on the latest bars, each with a direction, a strength in [0, 1] and, for chart
patterns, the levels that matter (neckline, target, invalidation).

Detected patterns
-----------------
* Candlesticks (last ``candle_lookback`` bars): doji, hammer / hanging man, inverted hammer / shooting
  star, bullish / bearish engulfing, harami, piercing line / dark cloud cover, morning / evening star,
  three white soldiers / three black crows, inside bar.
* Chart patterns (swing pivots): support / resistance levels and their breakouts, double top / bottom,
  head and shoulders (and inverse), ascending / descending / symmetrical triangles, rising / falling
  wedges, new 52-week highs / lows.
* Indicator events: golden / death cross (SMA 50/200), RSI divergences, RSI extremes, MACD crosses,
  Bollinger squeeze, gaps, volume spikes.

These are the classic textbook definitions with tolerances scaled by ATR, so the same rules work for an
ETF and a volatile token. They are *descriptions* of the chart, not edges: whether a pattern is worth
trading is for the agent to learn (``OnlineLinearAgent`` uses :func:`pattern_features`) or to judge
(``ClaudePortfolioManager`` reads the report).

Example::

    report = sim.detect_patterns("AAPL")        # or obs.patterns("AAPL") inside an agent
    print(report.summary())
    report.score()                               # net bullishness in [-1, 1]
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"
_SIGN = {BULLISH: 1.0, BEARISH: -1.0, NEUTRAL: 0.0}


@dataclass
class Pattern:
    """One detected pattern.

    Attributes:
        name: identifier (``"double_bottom"``, ``"bullish_engulfing"``...).
        kind: ``"candlestick"`` | ``"chart"`` | ``"indicator"``.
        direction: ``"bullish"`` | ``"bearish"`` | ``"neutral"``.
        strength: 0-1, how textbook-clean and how recent the pattern is.
        bars_ago: 0 = completed on the last bar.
        status: ``"confirmed"`` (e.g. neckline broken) or ``"forming"``.
        levels: prices that matter (``neckline``, ``target``, ``invalidation``, ``level``...).
        note: one-line human description.
    """

    name: str
    kind: str
    direction: str
    strength: float
    bars_ago: int = 0
    status: str = "confirmed"
    levels: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strength"] = round(self.strength, 3)
        d["levels"] = {k: _round_px(v) for k, v in self.levels.items()}
        return d


@dataclass
class PatternReport:
    """Everything :func:`detect_patterns` found for one symbol at one bar."""

    symbol: str
    as_of: str
    close: float
    trend: dict[str, Any]
    support: list[dict[str, Any]]
    resistance: list[dict[str, Any]]
    indicators: dict[str, float | None]
    patterns: list[Pattern]

    def score(self) -> float:
        """Net bullishness in [-1, 1] (direction x strength of every pattern, plus the trend)."""
        s = sum(_SIGN[p.direction] * p.strength for p in self.patterns)
        s += 0.5 * _SIGN.get(self.trend.get("direction", NEUTRAL), 0.0) * float(self.trend.get("quality") or 0.0)
        return float(math.tanh(s / 2.0))

    def bullish(self) -> list[Pattern]:
        return [p for p in self.patterns if p.direction == BULLISH]

    def bearish(self) -> list[Pattern]:
        return [p for p in self.patterns if p.direction == BEARISH]

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "as_of": self.as_of, "close": _round_px(self.close), "score": round(self.score(), 3),
                "trend": self.trend, "support": self.support, "resistance": self.resistance,
                "indicators": self.indicators, "patterns": [p.to_dict() for p in self.patterns]}

    def summary(self, max_patterns: int = 8) -> str:
        """Compact one-paragraph description (convenient for LLM prompts and logs)."""
        t = self.trend
        parts = [f"{self.symbol} {_fmt_px(self.close)}",
                 f"trend {t.get('direction')} (60-bar {t.get('change_pct', 0):+.1f}%, R2 {t.get('r2', 0):.2f}, "
                 f"{t.get('ma_alignment', 'n/a')})"]
        if self.support:
            parts.append("support " + ", ".join(f"{_fmt_px(s['price'])}x{s['touches']}" for s in self.support[:2]))
        if self.resistance:
            parts.append("resistance " + ", ".join(f"{_fmt_px(r['price'])}x{r['touches']}" for r in self.resistance[:2]))
        ind = self.indicators
        parts.append(f"RSI {ind.get('rsi_14') or float('nan'):.0f}, ATR {ind.get('atr_pct') or float('nan'):.1f}%")
        pats = sorted(self.patterns, key=lambda p: -p.strength)[:max_patterns]
        if pats:
            parts.append("patterns: " + "; ".join(
                f"{p.name} [{p.direction}, {p.status}, {p.strength:.2f}"
                + (f", {p.bars_ago}b ago" if p.bars_ago else "")
                + (", " + ", ".join(f"{k} {_fmt_px(v)}" for k, v in p.levels.items()) if p.levels else "") + "]"
                for p in pats))
        else:
            parts.append("patterns: none")
        parts.append(f"score {self.score():+.2f}")
        return " | ".join(parts)


# ----------------------------------------------------------------------------- helpers
def _round_px(x: float) -> float:
    if x is None or not math.isfinite(x):
        return x
    mag = 10 ** (4 - int(math.floor(math.log10(abs(x))))) if x else 1
    return round(x * mag) / mag if x else 0.0


def _fmt_px(x: float) -> str:
    return f"{_round_px(x):g}"


def atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    out = np.empty_like(tr)
    out[0] = tr[0]
    a = 1.0 / n
    for i in range(1, len(tr)):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


def rsi_array(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI for every bar (nan before ``n`` bars)."""
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    d = np.diff(close)
    gain = loss = 0.0
    a = 1.0 / n
    for i, x in enumerate(d):
        g, l = (x if x > 0 else 0.0), (-x if x < 0 else 0.0)
        if i == 0:
            gain, loss = g, l
        else:
            gain += a * (g - gain)
            loss += a * (l - loss)
        if i + 1 >= n:
            out[i + 1] = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    a = 2.0 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def find_pivots(high: np.ndarray, low: np.ndarray, order: int = 4) -> tuple[list[int], list[int]]:
    """Swing highs / lows: bar ``i`` is a pivot high if its high exceeds the ``order`` bars before it and is
    not exceeded by the ``order`` bars after it (so a pivot is only known ``order`` bars later: causal)."""
    n = len(high)
    highs, lows = [], []
    for i in range(order, n - order):
        if high[i] > high[i - order:i].max() and high[i] >= high[i + 1:i + order + 1].max():
            highs.append(i)
        if low[i] < low[i - order:i].min() and low[i] <= low[i + 1:i + order + 1].min():
            lows.append(i)
    return highs, lows


def _levels(prices: list[tuple[int, float]], tol: float) -> list[dict[str, Any]]:
    """Cluster pivot prices closer than ``tol`` into levels."""
    if not prices:
        return []
    pts = sorted(prices, key=lambda t: t[1])
    clusters: list[list[tuple[int, float]]] = [[pts[0]]]
    for idx, px in pts[1:]:
        if px - np.mean([q for _, q in clusters[-1]]) <= tol:  # vs the cluster mean: no chaining
            clusters[-1].append((idx, px))
        else:
            clusters.append([(idx, px)])
    return [{"price": float(np.mean([p for _, p in c])), "touches": len(c), "last_touch": max(i for i, _ in c)}
            for c in clusters]


# -------------------------------------------------------------------------- candlesticks
def candlestick_patterns(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int | None = None,
                         ) -> list[Pattern]:
    """Candlestick patterns completed at bar ``i`` (default: the last bar)."""
    n = len(c)
    i = n - 1 if i is None else i
    if i < 10:
        return []
    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = float(body[i - 10:i].mean()) or 1e-12
    trend = c[i - 1] - c[i - 6]  # direction of the 5 bars before the pattern
    down, up = trend < 0, trend > 0
    bull = lambda k: c[k] > o[k]  # noqa: E731
    bear = lambda k: c[k] < o[k]  # noqa: E731
    out: list[Pattern] = []

    def add(name: str, direction: str, strength: float, note: str) -> None:
        out.append(Pattern(name, "candlestick", direction, float(min(1.0, strength)), note=note))

    r = rng[i]
    if r > 0:
        if body[i] <= 0.1 * r:
            add("doji", NEUTRAL, 0.3, "open ~ close: indecision")
        elif lower[i] >= 2 * body[i] and upper[i] <= max(0.35 * body[i], 0.1 * r) and body[i] <= 0.4 * r:
            if down:
                add("hammer", BULLISH, 0.55, "long lower wick after a decline: buyers rejected lower prices")
            elif up:
                add("hanging_man", BEARISH, 0.4, "hammer shape after a rally: selling appeared intraday")
        elif upper[i] >= 2 * body[i] and lower[i] <= max(0.35 * body[i], 0.1 * r) and body[i] <= 0.4 * r:
            if up:
                add("shooting_star", BEARISH, 0.55, "long upper wick after a rally: higher prices rejected")
            elif down:
                add("inverted_hammer", BULLISH, 0.35, "long upper wick after a decline")
    j = i - 1
    if bear(j) and bull(i) and o[i] <= c[j] and c[i] >= o[j] and body[i] > body[j]:
        add("bullish_engulfing", BULLISH, 0.5 + 0.2 * down, "bullish body engulfs the previous bearish body")
    elif bull(j) and bear(i) and o[i] >= c[j] and c[i] <= o[j] and body[i] > body[j]:
        add("bearish_engulfing", BEARISH, 0.5 + 0.2 * up, "bearish body engulfs the previous bullish body")
    elif body[j] > 1.2 * avg_body and body[i] < 0.5 * body[j] and max(o[i], c[i]) <= max(o[j], c[j]) \
            and min(o[i], c[i]) >= min(o[j], c[j]):
        if bear(j) and bull(i) and down:
            add("bullish_harami", BULLISH, 0.35, "small bullish body inside a large bearish one")
        elif bull(j) and bear(i) and up:
            add("bearish_harami", BEARISH, 0.35, "small bearish body inside a large bullish one")
    if bear(j) and body[j] > avg_body and bull(i) and o[i] <= c[j] and (o[j] + c[j]) / 2 < c[i] < o[j] and down:
        add("piercing_line", BULLISH, 0.45, "closes above the midpoint of the prior bearish body")
    elif bull(j) and body[j] > avg_body and bear(i) and o[i] >= c[j] and o[j] < c[i] < (o[j] + c[j]) / 2 and up:
        add("dark_cloud_cover", BEARISH, 0.45, "closes below the midpoint of the prior bullish body")
    k = i - 2
    if body[k] > avg_body and body[j] < 0.35 * body[k]:
        if bear(k) and bull(i) and c[i] > (o[k] + c[k]) / 2 and c[i - 3] > c[k]:
            add("morning_star", BULLISH, 0.6, "large bearish bar, small pause, strong bullish bar")
        elif bull(k) and bear(i) and c[i] < (o[k] + c[k]) / 2 and c[i - 3] < c[k]:
            add("evening_star", BEARISH, 0.6, "large bullish bar, small pause, strong bearish bar")
    last3 = range(i - 2, i + 1)
    if all(bull(m) and body[m] > 0.6 * avg_body and upper[m] < 0.4 * body[m] for m in last3) \
            and c[i] > c[j] > c[k] and o[i] >= o[j] >= o[k] and o[i] <= c[j] and o[j] <= c[k]:
        add("three_white_soldiers", BULLISH, 0.55, "three strong rising bullish bars")
    elif all(bear(m) and body[m] > 0.6 * avg_body and lower[m] < 0.4 * body[m] for m in last3) \
            and c[i] < c[j] < c[k] and o[i] <= o[j] <= o[k] and o[i] >= c[j] and o[j] >= c[k]:
        add("three_black_crows", BEARISH, 0.55, "three strong falling bearish bars")
    if h[i] < h[j] and l[i] > l[j] and not any(p.name.endswith("harami") for p in out):
        add("inside_bar", NEUTRAL, 0.2, "range inside the previous bar: compression before a move")
    return out


def candle_signal(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int | None = None) -> float:
    """Net candlestick signal at bar ``i`` in [-1, 1] (used as a learning feature)."""
    pats = candlestick_patterns(o, h, l, c, i)
    return float(np.clip(sum(_SIGN[p.direction] * p.strength for p in pats), -1.0, 1.0))


# ------------------------------------------------------------------------------- report
def detect_patterns(df: pd.DataFrame, symbol: str = "", candle_lookback: int = 3, pivot_order: int = 4,
                    ) -> PatternReport:
    """Analyse an OHLCV frame (oldest first; at least ~60 bars, 250+ for 52-week and SMA 200 signals)."""
    df = df.dropna(subset=["close"])
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float) if "volume" in df else np.full(len(c), np.nan)
    n = len(c)
    as_of = df.index[-1].isoformat() if n else ""
    if n < 30:
        return PatternReport(symbol, as_of, float(c[-1]) if n else float("nan"), {"direction": NEUTRAL}, [], [],
                             {}, [])
    atr = atr_array(h, l, c)
    a = float(atr[-1]) or 1e-12
    px = float(c[-1])
    rsi = rsi_array(c)
    sma = lambda k: float(c[-k:].mean()) if n >= k else None  # noqa: E731
    sma20, sma50, sma200 = sma(20), sma(50), sma(200)
    pats: list[Pattern] = []

    for back in range(candle_lookback):
        for p in candlestick_patterns(o, h, l, c, n - 1 - back):
            p.bars_ago = back
            p.strength *= 1.0 - 0.25 * back
            pats.append(p)

    trend = _trend(c, sma20, sma50, sma200)
    piv_h, piv_l = find_pivots(h, l, pivot_order)
    tol = max(0.8 * a, 0.008 * px)
    window = max(0, n - 250)
    levels = _levels([(i, h[i]) for i in piv_h if i >= window] + [(i, l[i]) for i in piv_l if i >= window], tol)
    support = sorted([lv for lv in levels if lv["price"] < px - 0.1 * a], key=lambda lv: -lv["price"])[:3]
    resistance = sorted([lv for lv in levels if lv["price"] > px + 0.1 * a], key=lambda lv: lv["price"])[:3]

    pats += _breakouts(c, v, levels, a)
    pats += _double(h, l, c, piv_h, piv_l, a, top=True) + _double(h, l, c, piv_h, piv_l, a, top=False)
    pats += _head_shoulders(h, l, c, piv_h, piv_l, a, inverse=False)
    pats += _head_shoulders(h, l, c, piv_h, piv_l, a, inverse=True)
    pats += _triangles(h, l, c, a)
    pats += _indicator_events(o, h, l, c, v, rsi, atr, piv_h, piv_l)

    indicators = {
        "rsi_14": None if math.isnan(rsi[-1]) else round(float(rsi[-1]), 1),
        "atr_pct": round(a / px * 100, 2),
        "sma_20": sma20, "sma_50": sma50, "sma_200": sma200,
        "dist_sma200_pct": round((px / sma200 - 1) * 100, 2) if sma200 else None,
        "high_52w": float(h[-252:].max()), "low_52w": float(l[-252:].min()),
        "volume_ratio_20": round(float(v[-1] / np.nanmean(v[-21:-1])), 2)
        if n > 21 and np.isfinite(v[-21:]).all() and np.nanmean(v[-21:-1]) > 0 else None,
    }
    indicators = {k: (_round_px(x) if isinstance(x, float) and k.startswith(("sma", "high", "low")) else x)
                  for k, x in indicators.items()}
    fmt = lambda lv: {"price": _round_px(lv["price"]), "touches": lv["touches"],  # noqa: E731
                      "bars_since_touch": n - 1 - lv["last_touch"]}
    pats.sort(key=lambda p: -p.strength)
    return PatternReport(symbol, as_of, px, trend, [fmt(s) for s in support], [fmt(r) for r in resistance],
                         indicators, pats)


def _trend(c: np.ndarray, sma20, sma50, sma200) -> dict[str, Any]:
    m = min(60, len(c))
    y = np.log(c[-m:])
    x = np.arange(m)
    slope, icpt = np.polyfit(x, y, 1)
    fit = slope * x + icpt
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - fit) ** 2).sum()) / ss if ss > 0 else 0.0
    change = (math.exp(slope * (m - 1)) - 1) * 100
    if sma20 and sma50 and sma200:
        align = "bullish stack (price > SMA20 > SMA50 > SMA200)" if c[-1] > sma20 > sma50 > sma200 else \
            "bearish stack (price < SMA20 < SMA50 < SMA200)" if c[-1] < sma20 < sma50 < sma200 else \
            ("above SMA200" if c[-1] > sma200 else "below SMA200")
    elif sma20 and sma50:
        align = "price > SMA20 > SMA50" if c[-1] > sma20 > sma50 else \
            "price < SMA20 < SMA50" if c[-1] < sma20 < sma50 else "mixed"
    else:
        align = "n/a"
    direction = NEUTRAL
    if r2 >= 0.35 and abs(change) >= 3:
        direction = BULLISH if slope > 0 else BEARISH
    return {"direction": direction, "change_pct": round(change, 2), "r2": round(r2, 3),
            "quality": round(r2 if direction != NEUTRAL else 0.0, 3), "ma_alignment": align}


def _breakouts(c: np.ndarray, v: np.ndarray, levels: list[dict[str, Any]], a: float) -> list[Pattern]:
    n = len(c)
    out: list[Pattern] = []
    vol_ok = n > 22 and np.isfinite(v[-22:]).all() and np.mean(v[-22:-2]) > 0
    for lv in levels:
        if lv["touches"] < 2 or lv["last_touch"] > n - 4:
            continue
        L = lv["price"]
        for back in range(3):
            j = n - 1 - back
            vol_mult = float(v[j] / np.mean(v[-22:-2])) if vol_ok else 1.0
            strength = min(1.0, 0.4 + 0.06 * min(lv["touches"], 5) + (0.2 if vol_mult >= 1.5 else 0.0)) * (1 - 0.2 * back)
            if c[j - 1] <= L < c[j] and c[-1] > L + 0.1 * a:
                out.append(Pattern("resistance_breakout", "chart", BULLISH, strength, back, levels={"level": L},
                                   note=f"closed above a {lv['touches']}-touch resistance"
                                        + (f" on {vol_mult:.1f}x volume" if vol_mult >= 1.5 else "")))
                break
            if c[j - 1] >= L > c[j] and c[-1] < L - 0.1 * a:
                out.append(Pattern("support_breakdown", "chart", BEARISH, strength, back, levels={"level": L},
                                   note=f"closed below a {lv['touches']}-touch support"
                                        + (f" on {vol_mult:.1f}x volume" if vol_mult >= 1.5 else "")))
                break
    best: dict[str, Pattern] = {}  # one breakout / breakdown: the strongest level crossed
    for p in out:
        if p.name not in best or p.strength > best[p.name].strength:
            best[p.name] = p
    return list(best.values())


def _double(h, l, c, piv_h, piv_l, a: float, top: bool) -> list[Pattern]:
    n = len(c)
    piv = [i for i in (piv_h if top else piv_l) if i >= n - 120]
    if len(piv) < 2:
        return []
    i1, i2 = piv[-2], piv[-1]
    if i2 - i1 < 8 or i2 < n - 40:
        return []
    ext = h if top else l
    p1, p2 = float(ext[i1]), float(ext[i2])
    tol = max(1.0 * a, 0.015 * p1)
    if abs(p1 - p2) > tol:
        return []
    if top:
        neck = float(l[i1:i2 + 1].min())
        height = max(p1, p2) - neck
    else:
        neck = float(h[i1:i2 + 1].max())
        height = neck - min(p1, p2)
    if height < 2.0 * a:
        return []
    px = c[-1]
    crossed = (px < neck) if top else (px > neck)
    invalid = (px > max(p1, p2) + 0.5 * a) if top else (px < min(p1, p2) - 0.5 * a)
    if invalid:
        return []
    if crossed:
        since = [k for k in range(i2, n) if (c[k] < neck if top else c[k] > neck)]
        if since and since[0] < n - 15:  # broke long ago: no longer a fresh signal
            return []
        status, strength = "confirmed", 0.75
    else:
        # Still forming: price must be travelling from the second extreme towards the neckline (no
        # opposite pivot since), otherwise a sideways range would show a double top and bottom at once.
        opposite = piv_l if top else piv_h
        if any(k > i2 for k in opposite) or abs(px - ext[i2]) < 0.5 * a:
            return []
        status, strength = "forming", 0.4
    name = "double_top" if top else "double_bottom"
    target = neck - height if top else neck + height
    return [Pattern(name, "chart", BEARISH if top else BULLISH, strength, n - 1 - i2, status,
                    {"neckline": neck, "target": target, "invalidation": max(p1, p2) if top else min(p1, p2)},
                    note=f"two {'peaks' if top else 'troughs'} {i2 - i1} bars apart; "
                         f"{'neckline broken' if crossed else 'waiting for a close beyond the neckline'}")]


def _head_shoulders(h, l, c, piv_h, piv_l, a: float, inverse: bool) -> list[Pattern]:
    n = len(c)
    piv = [i for i in (piv_l if inverse else piv_h) if i >= n - 150]
    if len(piv) < 3:
        return []
    ls, hd, rs = piv[-3:]
    if rs < n - 30:
        return []
    sgn = -1.0 if inverse else 1.0
    ext = l if inverse else h
    pls, phd, prs = sgn * ext[ls], sgn * ext[hd], sgn * ext[rs]  # in "top" orientation
    tol = max(0.75 * a, 0.01 * abs(ext[hd]))
    if not (phd > pls + tol and phd > prs + tol and abs(pls - prs) <= 1.5 * tol):
        return []
    opp = h if inverse else l
    pick = np.argmax if inverse else np.argmin
    t1 = ls + int(pick(opp[ls:hd + 1]))
    t2 = hd + int(pick(opp[hd:rs + 1]))
    slope = (opp[t2] - opp[t1]) / max(t2 - t1, 1)
    neck_now = float(opp[t2] + slope * (n - 1 - t2))
    height = abs(float(ext[hd]) - float(opp[t1] + slope * (hd - t1)))
    if height < 2 * a:
        return []
    px = c[-1]
    crossed = px > neck_now if inverse else px < neck_now
    if (inverse and px < ext[rs] - 0.5 * a) or (not inverse and px > ext[rs] + 0.5 * a):
        return []
    status, strength = ("confirmed", 0.8) if crossed else ("forming", 0.45)
    target = neck_now + height if inverse else neck_now - height
    return [Pattern("inverse_head_and_shoulders" if inverse else "head_and_shoulders", "chart",
                    BULLISH if inverse else BEARISH, strength, n - 1 - rs, status,
                    {"neckline": neck_now, "target": target, "invalidation": float(ext[hd])},
                    note="three " + ("troughs" if inverse else "peaks") + " with the middle one most extreme; "
                         + ("neckline broken" if crossed else "neckline not yet broken"))]


def _triangles(h, l, c, a: float, lookback: int = 60) -> list[Pattern]:
    n = len(c)
    if n < lookback:
        return []
    base = n - lookback
    ph, pl = find_pivots(h[base:], l[base:], 3)
    if len(ph) < 2 or len(pl) < 2 or len(ph) + len(pl) < 5:
        return []
    xh, yh = np.array(ph, float), h[base:][ph]
    xl, yl = np.array(pl, float), l[base:][pl]
    sh, ih = np.polyfit(xh, yh, 1)
    sl, il = np.polyfit(xl, yl, 1)
    if np.abs(yh - (sh * xh + ih)).max() > 1.2 * a or np.abs(yl - (sl * xl + il)).max() > 1.2 * a:
        return []  # pivots do not line up
    x0, x1 = min(xh[0], xl[0]), lookback - 1
    w0 = (sh * x0 + ih) - (sl * x0 + il)
    w1 = (sh * x1 + ih) - (sl * x1 + il)
    if w0 <= 0 or w1 <= 0 or w1 > 0.8 * w0:
        return []  # not converging
    flat = 0.5 * a / 20  # moves less than half an ATR every 20 bars
    up_t, dn_t = sh > flat, sh < -flat
    up_b, dn_b = sl > flat, sl < -flat
    if not up_t and not dn_t and up_b:
        name, direction = "ascending_triangle", BULLISH
    elif dn_t and not up_b and not dn_b:
        name, direction = "descending_triangle", BEARISH
    elif dn_t and up_b:
        name, direction = "symmetrical_triangle", NEUTRAL
    elif up_t and up_b:
        name, direction = "rising_wedge", BEARISH
    elif dn_t and dn_b:
        name, direction = "falling_wedge", BULLISH
    else:
        return []
    upper, lower = sh * x1 + ih, sl * x1 + il
    px = c[-1]
    status, strength, note = "forming", 0.4, "price coiling between converging trendlines"
    if px > upper + 0.2 * a:
        direction, status, strength, note = BULLISH, "confirmed", 0.7, "closed above the upper trendline"
    elif px < lower - 0.2 * a:
        direction, status, strength, note = BEARISH, "confirmed", 0.7, "closed below the lower trendline"
    height = w0
    levels = {"upper": float(upper), "lower": float(lower)}
    if status == "confirmed":
        levels["target"] = float(px + height if direction == BULLISH else px - height)
    return [Pattern(name, "chart", direction, strength, 0, status, levels, note)]


def _indicator_events(o, h, l, c, v, rsi, atr, piv_h, piv_l) -> list[Pattern]:
    n = len(c)
    out: list[Pattern] = []
    a = float(atr[-1])
    if n >= 210:
        s50 = np.convolve(c, np.ones(50) / 50, "valid")[-11:]
        s200 = np.convolve(c, np.ones(200) / 200, "valid")[-11:]
        d = s50 - s200
        for back in range(10):
            k = len(d) - 1 - back
            if d[k - 1] <= 0 < d[k]:
                out.append(Pattern("golden_cross", "indicator", BULLISH, 0.5 * (1 - back / 15), back,
                                   note="SMA50 crossed above SMA200"))
                break
            if d[k - 1] >= 0 > d[k]:
                out.append(Pattern("death_cross", "indicator", BEARISH, 0.5 * (1 - back / 15), back,
                                   note="SMA50 crossed below SMA200"))
                break
    if n >= 252:
        if c[-1] >= c[-252:].max():
            out.append(Pattern("new_52w_high", "chart", BULLISH, 0.45, note="highest close of the last year"))
        elif c[-1] <= c[-252:].min():
            out.append(Pattern("new_52w_low", "chart", BEARISH, 0.45, note="lowest close of the last year"))
    for pivots, ext, bullish in ((piv_l, l, True), (piv_h, h, False)):
        recent = [i for i in pivots if i >= n - 60 and not math.isnan(rsi[i])]
        if len(recent) >= 2:
            i1, i2 = recent[-2], recent[-1]
            if i2 - i1 >= 5 and i2 >= n - 15:
                if bullish and ext[i2] < ext[i1] and rsi[i2] > rsi[i1] + 2:
                    out.append(Pattern("bullish_rsi_divergence", "indicator", BULLISH, 0.5, n - 1 - i2,
                                       note="lower low in price, higher low in RSI: selling momentum fading"))
                if not bullish and ext[i2] > ext[i1] and rsi[i2] < rsi[i1] - 2:
                    out.append(Pattern("bearish_rsi_divergence", "indicator", BEARISH, 0.5, n - 1 - i2,
                                       note="higher high in price, lower high in RSI: buying momentum fading"))
    r = rsi[-1]
    if not math.isnan(r):
        if r <= 30:
            out.append(Pattern("rsi_oversold", "indicator", BULLISH, 0.25 + (30 - r) / 60, note=f"RSI {r:.0f}"))
        elif r >= 70:
            out.append(Pattern("rsi_overbought", "indicator", BEARISH, 0.25 + (r - 70) / 60, note=f"RSI {r:.0f}"))
    if n >= 40:
        line = _ema(c, 12) - _ema(c, 26)
        sig = _ema(line, 9)
        hist = line - sig
        for back in range(3):
            k = n - 1 - back
            if hist[k - 1] <= 0 < hist[k]:
                out.append(Pattern("macd_bullish_cross", "indicator", BULLISH, 0.3 + 0.1 * (line[k] < 0), back,
                                   note="MACD crossed above its signal" + (" below zero" if line[k] < 0 else "")))
                break
            if hist[k - 1] >= 0 > hist[k]:
                out.append(Pattern("macd_bearish_cross", "indicator", BEARISH, 0.3 + 0.1 * (line[k] > 0), back,
                                   note="MACD crossed below its signal" + (" above zero" if line[k] > 0 else "")))
                break
    if n >= 140:
        mid = pd.Series(c).rolling(20).mean().to_numpy()
        sd = pd.Series(c).rolling(20).std().to_numpy()
        width = 4 * sd / mid
        hist_w = width[-120:]
        if np.isfinite(hist_w).all() and width[-1] <= np.quantile(hist_w, 0.1):
            out.append(Pattern("bollinger_squeeze", "indicator", NEUTRAL, 0.4,
                               note="Bollinger bands at their narrowest in 6 months: a large move often follows"))
    if n >= 2 and a > 0:
        if o[-1] > h[-2] and o[-1] - h[-2] >= 0.5 * a:
            out.append(Pattern("gap_up", "chart", BULLISH, 0.35, levels={"gap_fill": float(h[-2])},
                               note="opened above the previous high"))
        elif o[-1] < l[-2] and l[-2] - o[-1] >= 0.5 * a:
            out.append(Pattern("gap_down", "chart", BEARISH, 0.35, levels={"gap_fill": float(l[-2])},
                               note="opened below the previous low"))
    if n > 21 and np.isfinite(v[-21:]).all() and v[-21:-1].mean() > 0:
        mult = v[-1] / v[-21:-1].mean()
        if mult >= 2.5:
            direction = BULLISH if c[-1] > o[-1] else BEARISH if c[-1] < o[-1] else NEUTRAL
            out.append(Pattern("volume_spike", "indicator", direction, min(0.6, 0.2 + mult / 20),
                               note=f"volume {mult:.1f}x its 20-bar average"))
    return out


# ---------------------------------------------------------------------- learning features
PATTERN_FEATURES = ["breakout_20", "trend_quality", "candle"]


def pattern_features(c: np.ndarray, o: np.ndarray | None = None, h: np.ndarray | None = None,
                     l: np.ndarray | None = None) -> np.ndarray:
    """Cheap per-bar pattern features (last bar) for learning agents, each in [-1, 1]:

    * ``breakout_20``: +1 close above the prior 20-bar high, -1 below the prior 20-bar low, else 0;
    * ``trend_quality``: sign of the 50-bar log-price slope times its R^2 (clean trends near +-1);
    * ``candle``: net candlestick signal of the last bar (0 when only closes are available).
    """
    c = np.asarray(c, dtype=float)
    brk = 1.0 if c[-1] > c[-21:-1].max() else -1.0 if c[-1] < c[-21:-1].min() else 0.0
    y = np.log(c[-50:])
    x = np.arange(len(y), dtype=float)
    slope, icpt = np.polyfit(x, y, 1)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - (slope * x + icpt)) ** 2).sum()) / ss if ss > 0 else 0.0
    trend = math.copysign(max(r2, 0.0), slope)
    candle = 0.0
    if o is not None and h is not None and l is not None and len(o) == len(c):
        candle = candle_signal(np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), c)
    return np.array([brk, trend, candle])
