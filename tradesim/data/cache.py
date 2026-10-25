"""On-disk bar cache wrapping any provider (parquet if ``pyarrow`` is installed, else pickle)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError, OrderBookSnapshot
from tradesim.data.timeframes import Timeframe
from tradesim.types import Quote

log = logging.getLogger(__name__)

try:
    import pyarrow  # noqa: F401

    _EXT = "parquet"
except ImportError:  # pragma: no cover
    _EXT = "pkl"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class CachedProvider(DataProvider):
    """Caches historical bars per ``(provider, symbol, timeframe)``.

    Tracks which ``[start, end)`` intervals have been fetched, so a request fully
    inside a known interval never hits the network. Intervals touching the most
    recent bar are not marked as covered (the last bar may still be forming).
    Quotes and order books are never cached.
    """

    def __init__(self, provider: DataProvider, cache_dir: str | Path = "~/.tradesim/cache"):
        self.provider = provider
        self.name = f"cached:{provider.name}"
        self.root = Path(cache_dir).expanduser() / _safe(provider.name)
        self.root.mkdir(parents=True, exist_ok=True)

    def supports(self, asset: Asset) -> bool:
        return self.provider.supports(asset)

    def _paths(self, symbol: str, tf: Timeframe) -> tuple[Path, Path]:
        base = self.root / f"{_safe(symbol)}__{tf.value}"
        return base.with_suffix(f".{_EXT}"), base.with_suffix(".json")

    def _read(self, symbol: str, tf: Timeframe) -> tuple[pd.DataFrame | None, list[list[str]]]:
        data_path, meta_path = self._paths(symbol, tf)
        if not data_path.exists() or not meta_path.exists():
            return None, []
        df = pd.read_parquet(data_path) if _EXT == "parquet" else pd.read_pickle(data_path)
        return df, json.loads(meta_path.read_text())["intervals"]

    def _write(self, symbol: str, tf: Timeframe, df: pd.DataFrame, intervals: list[list[str]]) -> None:
        data_path, meta_path = self._paths(symbol, tf)
        if _EXT == "parquet":
            df.to_parquet(data_path)
        else:
            df.to_pickle(data_path)
        meta_path.write_text(json.dumps({"intervals": intervals}))

    @staticmethod
    def _merge_intervals(intervals: list[list[str]]) -> list[list[str]]:
        parsed = sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in intervals)
        merged: list[list[pd.Timestamp]] = []
        for a, b in parsed:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [[a.isoformat(), b.isoformat()] for a, b in merged]

    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
        e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
        cached, intervals = self._read(asset.symbol, tf)
        if cached is not None:
            for a, b in intervals:
                if pd.Timestamp(a) <= s and e <= pd.Timestamp(b):
                    return cached[(cached.index >= s) & (cached.index < e)]
        fresh = self.provider.get_bars(asset, tf, s.to_pydatetime(), e.to_pydatetime())
        merged = fresh if cached is None else pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        # Only mark as covered what is safely in the past (last bar may still be forming).
        safe_end = min(e, pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(seconds=2 * tf.seconds))
        if safe_end > s:
            intervals = self._merge_intervals(intervals + [[s.isoformat(), safe_end.isoformat()]])
        self._write(asset.symbol, tf, merged, intervals)
        return fresh

    def get_bars_many(self, assets, timeframe, start, end) -> dict[str, pd.DataFrame]:
        tf = Timeframe.parse(timeframe)
        out: dict[str, pd.DataFrame] = {}
        to_fetch: list[Asset] = []
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        for a in assets:
            cached, intervals = self._read(a.symbol, tf)
            if cached is not None and any(pd.Timestamp(x) <= s and e <= pd.Timestamp(y) for x, y in intervals):
                out[a.symbol] = cached[(cached.index >= s) & (cached.index < e)]
            else:
                to_fetch.append(a)
        if to_fetch:
            fetched = self.provider.get_bars_many(to_fetch, tf, start, end)
            safe_end = min(e, pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(seconds=2 * tf.seconds))
            for sym, df in fetched.items():
                cached, intervals = self._read(sym, tf)
                merged = df if cached is None else pd.concat([cached, df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                if safe_end > s:
                    intervals = self._merge_intervals(intervals + [[s.isoformat(), safe_end.isoformat()]])
                self._write(sym, tf, merged, intervals)
                out[sym] = df
        return out

    def get_quote(self, asset: Asset) -> Quote:
        return self.provider.get_quote(asset)

    def get_order_book(self, asset: Asset, depth: int = 50) -> OrderBookSnapshot:
        return self.provider.get_order_book(asset, depth)

    @property
    def supports_live(self) -> bool:
        return self.provider.supports_live


__all__ = ["CachedProvider", "DataUnavailableError"]
