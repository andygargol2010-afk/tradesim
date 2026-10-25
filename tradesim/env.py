"""Gymnasium environment wrapping :class:`TradingSimulator`.

Observation (flat ``Box``, float32), for ``N`` symbols and a ``window`` of bars:

* per symbol, per bar in the window: log return, high/close-1, low/close-1,
  log-volume z-score (``N * window * 4``);
* per symbol, latest: RSI/100 - 0.5, realised-vol ratio, close/SMA20 - 1,
  current portfolio weight, modelled spread in bps / 100 (``N * 5``);
* account: cash fraction, gross leverage, current drawdown, unrealised-PnL
  fraction, fraction of the episode elapsed (``5``).

Actions (``action_mode``):

* ``"weights"`` (default): ``Box`` of target portfolio weights per symbol, in
  ``[0, 1]`` (long only) or ``[-1, 1]`` (shorting enabled). If the gross sum
  exceeds ``max_leverage`` the vector is scaled down. Rebalancing uses market
  orders, so the agent pays spread, impact and fees on every change.
* ``"discrete"``: ``MultiDiscrete([3] * N)``: 0 = hold, 1 = buy
  ``trade_fraction`` of equity, 2 = sell ``trade_fraction`` of equity (capped
  at the position when shorting is disabled).
* Any mode also accepts ``{"orders": [{symbol, side, quantity|notional|percent_of_equity, ...}, ...]}``
  (the same arguments as ``place_order``), which is convenient for LLM agents.

Reward (``reward``): ``"log_return"`` of equity (default), ``"pnl"`` (equity change
/ initial equity) or ``"sharpe"`` (differential Sharpe ratio, Moody & Saffell 2001),
minus ``drawdown_penalty`` x increase in drawdown.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tradesim.assets.universe import AssetUniverse
from tradesim.data.base import DataProvider
from tradesim.simulator import SimulatorConfig, TradingSimulator
from tradesim.types import Side

N_BAR_FEATURES = 4
N_SYMBOL_FEATURES = 5
N_ACCOUNT_FEATURES = 5


class PaperTradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        provider: DataProvider | None = None,
        universe: AssetUniverse | None = None,
        config: SimulatorConfig | None = None,
        episode_length: int | None = 252,
        window: int = 30,
        action_mode: Literal["weights", "discrete"] = "weights",
        trade_fraction: float = 0.1,
        reward: Literal["log_return", "pnl", "sharpe"] = "log_return",
        drawdown_penalty: float = 0.0,
        min_rebalance_fraction: float = 0.005,
        random_start: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.symbols = list(symbols)
        self.start, self.end = start, end
        self.sim = TradingSimulator(universe=universe, provider=provider, config=config or SimulatorConfig())
        self.episode_length = episode_length
        self.window = window
        self.action_mode = action_mode
        self.trade_fraction = trade_fraction
        self.reward_mode = reward
        self.drawdown_penalty = drawdown_penalty
        self.min_rebalance_fraction = min_rebalance_fraction
        self.random_start = random_start
        self.render_mode = render_mode

        # Load data once to fix the symbol set (symbols without data are dropped).
        self.sim.reset(self.symbols, start, end)
        self.symbols = list(self.sim.symbols)
        n = len(self.symbols)
        self.allow_short = bool(self.sim.config.margin.allow_short)
        self.max_leverage = float(self.sim.config.margin.max_leverage)
        if action_mode == "weights":
            low = -1.0 if self.allow_short else 0.0
            self.action_space = spaces.Box(low=low, high=1.0, shape=(n,), dtype=np.float32)
        elif action_mode == "discrete":
            self.action_space = spaces.MultiDiscrete([3] * n)
        else:
            raise ValueError(f"unknown action_mode {action_mode!r}")
        obs_dim = n * window * N_BAR_FEATURES + n * N_SYMBOL_FEATURES + N_ACCOUNT_FEATURES
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._prev_equity = 1.0
        self._prev_dd = 0.0
        self._steps = 0
        self._dsr_a = 0.0
        self._dsr_b = 0.0
        self._last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        # Earliest start keeps `window` bars of in-window history visible; warm-up bars before
        # `start` also count, so offset 0 is fine when warmup_bars >= window.
        n_bars = self.sim.num_bars
        length = self.episode_length or (n_bars - 1)
        max_offset = max(n_bars - 1 - length, 0)
        if "start_offset" in options:
            offset = int(options["start_offset"])
        elif self.random_start and max_offset > 0:
            offset = int(self.np_random.integers(0, max_offset + 1))
        else:
            offset = 0
        # Execution noise / scenario draws follow the env's seeded RNG => reproducible episodes.
        self.sim.config.seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim.reset(self.symbols, self.start, self.end, start_offset=offset)
        self._steps = 0
        self._prev_equity = float(self.sim.portfolio.equity)
        self._prev_dd = 0.0
        self._dsr_a = self._dsr_b = 0.0
        obs = self._observation()
        info = self._info([])
        return obs, info

    def step(self, action):
        rejected = self._apply_action(action)
        fills = self.sim.step()
        self._steps += 1
        equity = float(self.sim.portfolio.equity)
        reward = self._reward(equity)
        self._prev_equity = equity
        terminated = bool(self.sim.bankrupt)
        truncated = bool(self.sim.done and not terminated) or (
            self.episode_length is not None and self._steps >= self.episode_length)
        obs = self._observation()
        info = self._info(fills, rejected)
        if self.render_mode == "human":
            print(self.render())
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str:
        p = self.sim.portfolio
        weights = {pos.symbol: round(float(pos.market_value / p.equity), 3) for pos in p.open_positions()} if p.equity > 0 else {}
        return (f"{self.sim.now:%Y-%m-%d %H:%M} equity={float(p.equity):,.2f} cash={float(p.cash):,.2f} "
                f"dd={float(self._current_dd()) * 100:.2f}% weights={weights}")

    def close(self) -> None:
        self.sim.close()

    # --------------------------------------------------------------- actions
    def _apply_action(self, action) -> list[dict[str, Any]]:
        sim = self.sim
        rejected: list[dict[str, Any]] = []
        if isinstance(action, dict) and "orders" in action:
            for o in action["orders"]:
                res = sim.place_order(**o)
                if res["status"] == "rejected":
                    rejected.append(res)
            return rejected
        if self.action_mode == "weights":
            w = np.clip(np.asarray(action, dtype=float).reshape(-1), self.action_space.low, self.action_space.high)
            gross = np.abs(w).sum()
            if gross > self.max_leverage * 0.98:  # keep a little cash for fees / price moves
                w = w * (self.max_leverage * 0.98 / gross)
            eq = float(sim.portfolio.equity)
            current = {p.symbol: float(p.market_value) / eq for p in sim.portfolio.open_positions()} if eq > 0 else {}
            targets = {s: float(x) for s, x in zip(self.symbols, w)}
            # No-trade band: skip symbols whose weight change is too small to be worth the costs.
            changed = {s: t for s, t in targets.items() if abs(t - current.get(s, 0.0)) >= self.min_rebalance_fraction}
            if changed:
                full = {s: current.get(s, 0.0) for s in self.symbols}
                full.update(changed)
                for res in sim.target_weights({s: v for s, v in full.items()},
                                              min_trade_notional=eq * self.min_rebalance_fraction):
                    if res["status"] == "rejected":
                        rejected.append(res)
        else:
            a = np.asarray(action, dtype=int).reshape(-1)
            eq = float(sim.portfolio.equity)
            for sym, act in zip(self.symbols, a):
                if act == 0:
                    continue
                notional = eq * self.trade_fraction
                if act == 1:
                    res = sim.place_order(sym, Side.BUY, notional=notional, tag="discrete")
                else:
                    held = float(sim.portfolio.quantity(sym))
                    if held <= 0 and not self.allow_short:
                        continue
                    price = sim.get_price(sym)["bid"]
                    qty = notional / price if price > 0 else 0.0
                    if not self.allow_short:
                        qty = min(qty, held)
                    if qty <= 0:
                        continue
                    res = sim.place_order(sym, Side.SELL, quantity=qty, tag="discrete")
                if res["status"] == "rejected":
                    rejected.append(res)
        return rejected

    # ---------------------------------------------------------------- reward
    def _current_dd(self) -> float:
        p = self.sim.portfolio
        return float((p.peak_equity - p.equity) / p.peak_equity) if p.peak_equity > 0 else 0.0

    def _reward(self, equity: float) -> float:
        prev = max(self._prev_equity, 1e-9)
        r = equity / prev - 1.0
        if self.reward_mode == "log_return":
            reward = math.log(max(equity, 1e-9) / prev)
        elif self.reward_mode == "pnl":
            reward = (equity - prev) / float(self.sim.portfolio.initial_cash)
        else:  # differential Sharpe ratio
            eta = 0.01
            da, db = r - self._dsr_a, r * r - self._dsr_b
            denom = (self._dsr_b - self._dsr_a ** 2) ** 1.5
            reward = (self._dsr_b * da - 0.5 * self._dsr_a * db) / denom if denom > 1e-12 else 0.0
            self._dsr_a += eta * da
            self._dsr_b += eta * db
        dd = self._current_dd()
        if self.drawdown_penalty:
            reward -= self.drawdown_penalty * max(dd - self._prev_dd, 0.0)
        self._prev_dd = dd
        return reward

    # ----------------------------------------------------------- observation
    def _observation(self) -> np.ndarray:
        sim = self.sim
        assert sim.data is not None
        ts_ns = int(sim._calendar[sim.step_index])
        n, w = len(self.symbols), self.window
        bars = np.zeros((n, w, N_BAR_FEATURES), dtype=np.float32)
        sym_feats = np.zeros((n, N_SYMBOL_FEATURES), dtype=np.float32)
        eq = float(sim.portfolio.equity)
        for j, sym in enumerate(self.symbols):
            s = sim.data.series[sym]
            i = s.index_le(ts_ns)
            if i is None:
                continue
            lo = max(0, i - w)
            c = s.close[lo: i + 1]
            if len(c) >= 2:
                lr = np.diff(np.log(c))
                hi = s.high[lo + 1: i + 1] / c[1:] - 1
                lw = s.low[lo + 1: i + 1] / c[1:] - 1
                v = s.volume[lo + 1: i + 1]
                lv = np.log1p(np.nan_to_num(v, nan=0.0))
                vz = (lv - lv.mean()) / (lv.std() + 1e-9)
                k = len(lr)
                bars[j, w - k:, 0] = lr / max(s.sigma_typical, 1e-9) * 0.1  # scale: ~0.1 per typical move
                bars[j, w - k:, 1] = hi / max(s.sigma_typical, 1e-9) * 0.1
                bars[j, w - k:, 2] = lw / max(s.sigma_typical, 1e-9) * 0.1
                bars[j, w - k:, 3] = vz
            closes = s.close[max(0, i - 50): i + 1]
            delta = np.diff(closes)
            if len(delta) >= 14:
                gain = delta[-14:].clip(min=0).mean()
                loss = (-delta[-14:]).clip(min=0).mean()
                rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
                sym_feats[j, 0] = rsi / 100 - 0.5
            sym_feats[j, 1] = float(s.sigma[min(i + 1, len(s.sigma) - 1)] / max(s.sigma_typical, 1e-12)) - 1.0
            if len(closes) >= 20:
                sym_feats[j, 2] = float(closes[-1] / closes[-20:].mean() - 1.0) * 10
            pos = sim.portfolio.positions.get(sym)
            if pos is not None and eq > 0:
                sym_feats[j, 3] = float(pos.market_value) / eq
            q = sim.get_price(sym)
            sym_feats[j, 4] = q["spread_bps"] / 100.0
        p = sim.portfolio
        account = np.array([
            float(p.cash) / eq if eq > 0 else 0.0,
            float(p.gross_exposure) / eq if eq > 0 else 0.0,
            self._current_dd(),
            float(p.unrealized_pnl) / eq if eq > 0 else 0.0,
            self._steps / self.episode_length if self.episode_length else 0.0,
        ], dtype=np.float32)
        obs = np.concatenate([bars.reshape(-1), sym_feats.reshape(-1), account]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self, fills: list[dict[str, Any]], rejected: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        p = self.sim.portfolio
        return {
            "timestamp": self.sim.now.isoformat(),
            "equity": float(p.equity),
            "cash": float(p.cash),
            "drawdown": self._current_dd(),
            "max_drawdown": float(p.max_drawdown),
            "gross_exposure": float(p.gross_exposure),
            "total_fees": float(p.total_fees),
            "fills": fills,
            "rejected_orders": rejected or [],
            "step": self._steps,
        }
