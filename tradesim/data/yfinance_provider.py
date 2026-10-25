"""Yahoo Finance via ``yfinance`` (no API key). Equities, ETFs, crypto, FX, indices.

Caveats you must know before trusting results:

* Unofficial API: rate-limited (HTTP 429) and occasionally changes. Use the
  disk cache (:class:`tradesim.data.cache.CachedProvider`) to avoid re-downloads.
* Intraday history is short: 1m bars only for the last ~30 days (max 8 days
  per request), 5m-30m for the last 60 days, 1h for the last 730 days.
* Prices are split- and dividend-adjusted (``auto_adjust=True``), so returns
  include dividends. Volumes are split-adjusted.
* No historical bid/ask; spreads are modelled by :mod:`tradesim.market.liquidity`.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from tradesim.assets.universe import Asset
from tradesim.data.base import DataProvider, DataUnavailableError
from tradesim.data.timeframes import Timeframe, normalize_bars, resample_bars
from tradesim.types import AssetClass, Quote

log = logging.getLogger(__name__)

_INTERVAL = {
    Timeframe.M1: "1m", Timeframe.M5: "5m", Timeframe.M15: "15m", Timeframe.M30: "30m",
    Timeframe.H1: "1h", Timeframe.H4: "1h", Timeframe.D1: "1d", Timeframe.W1: "1wk",
}
_MAX_LOOKBACK = {Timeframe.M1: 29, Timeframe.M5: 59, Timeframe.M15: 59, Timeframe.M30: 59,
                 Timeframe.H1: 729, Timeframe.H4: 729}
_MAX_SPAN_PER_REQUEST = {Timeframe.M1: 7}


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, max_retries: int = 4, backoff_seconds: float = 2.0, batch_size: int = 100,
                 max_observed_spread_bps: float = 100.0):
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("pip install yfinance  (or: pip install 'tradesim[yfinance]')") from exc
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.batch_size = batch_size
        self.max_observed_spread_bps = max_observed_spread_bps

    def supports(self, asset: Asset) -> bool:
        return asset.asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO, AssetClass.FOREX, AssetClass.INDEX)

    # ------------------------------------------------------------------ helpers
    def _retry(self, fn, what: str):
        delay = self.backoff_seconds
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:  # yfinance raises a zoo of exception types
                last = exc
                msg = str(exc).lower()
                if "rate" in msg or "429" in msg or "too many" in msg:
                    log.warning("yfinance rate-limited on %s, retry %d in %.0fs", what, attempt + 1, delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise DataUnavailableError(f"yfinance {what}: {exc}") from exc
        raise DataUnavailableError(f"yfinance {what}: {last}")

    @staticmethod
    def _postprocess(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        if df is None or df.empty:
            return normalize_bars(pd.DataFrame())
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        if tf in (Timeframe.D1, Timeframe.W1):
            # Yahoo labels daily bars with exchange-local midnight; relabel to UTC midnight of that date.
            idx = pd.DatetimeIndex(df.index)
            dates = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(idx.tz).tz_localize(None)
            df.index = pd.DatetimeIndex(dates.normalize()).tz_localize("UTC")
        out = normalize_bars(df)
        return resample_bars(out, Timeframe.H4) if tf is Timeframe.H4 else out

    def _check_range(self, tf: Timeframe, start: datetime) -> None:
        limit = _MAX_LOOKBACK.get(tf)
        if limit is not None and _utc(start) < datetime.now(timezone.utc) - timedelta(days=limit):
            raise DataUnavailableError(
                f"Yahoo only serves {tf.value} bars for the last {limit} days; use a coarser timeframe, "
                f"another provider (Alpaca / ccxt), or the synthetic provider."
            )

    # --------------------------------------------------------------------- API
    def get_bars(self, asset: Asset, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf

        tf = Timeframe.parse(timeframe)
        self._check_range(tf, start)
        span = _MAX_SPAN_PER_REQUEST.get(tf)
        chunks: list[tuple[datetime, datetime]] = []
        cur = _utc(start)
        end = _utc(end)
        while cur < end:
            nxt = min(end, cur + timedelta(days=span)) if span else end
            chunks.append((cur, nxt))
            cur = nxt
        frames = []
        ticker = yf.Ticker(asset.symbol)
        for a, b in chunks:
            df = self._retry(
                lambda a=a, b=b: ticker.history(start=a, end=b, interval=_INTERVAL[tf], auto_adjust=True,
                                                actions=False, prepost=False, raise_errors=True),
                f"history({asset.symbol}, {tf.value})",
            )
            frames.append(df)
        df = pd.concat(frames) if frames else pd.DataFrame()
        out = self._postprocess(df, tf)
        if out.empty:
            raise DataUnavailableError(f"yfinance returned no {tf.value} bars for {asset.symbol}")
        return out[(out.index >= pd.Timestamp(start)) & (out.index < pd.Timestamp(end))]

    def get_bars_many(self, assets, timeframe, start, end) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        tf = Timeframe.parse(timeframe)
        self._check_range(tf, start)
        assets = list(assets)
        if tf in _MAX_SPAN_PER_REQUEST:
            return super().get_bars_many(assets, tf, start, end)
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(assets), self.batch_size):
            batch = assets[i: i + self.batch_size]
            symbols = [a.symbol for a in batch]
            raw = self._retry(
                lambda: yf.download(symbols, start=_utc(start), end=_utc(end), interval=_INTERVAL[tf],
                                    auto_adjust=True, actions=False, group_by="ticker", threads=True,
                                    progress=False),
                f"download({len(symbols)} symbols)",
            )
            for sym in symbols:
                try:
                    df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    continue
                df = self._postprocess(df.dropna(how="all"), tf)
                if not df.empty:
                    out[sym] = df
        missing = [a.symbol for a in assets if a.symbol not in out]
        if missing:
            log.warning("yfinance: no data for %d symbols (e.g. %s)", len(missing), missing[:5])
        return out

    def get_quote(self, asset: Asset) -> Quote:
        import yfinance as yf

        t = yf.Ticker(asset.symbol)
        fi = self._retry(lambda: t.fast_info, f"fast_info({asset.symbol})")
        last = float(fi["last_price"])
        bid = ask = None
        state = None
        try:
            info = t.info  # slow, but the only place Yahoo exposes bid/ask
            bid, ask, state = info.get("bid"), info.get("ask"), info.get("marketState")
        except Exception:  # noqa: BLE001 - quotes are best effort
            pass
        # Outside the regular session Yahoo serves stale bid/ask (spreads of several %, mid away from the last
        # trade): only trust them in session, bracketing the last price and reasonably tight; else model the spread.
        observed = bool(bid and ask and 0 < bid <= ask and state in (None, "REGULAR")
                        and bid <= last * 1.001 and ask >= last * 0.999
                        and (ask - bid) / last * 1e4 <= self.max_observed_spread_bps)
        return Quote(
            symbol=asset.symbol, timestamp=datetime.now(timezone.utc),
            bid=float(bid) if observed else last, ask=float(ask) if observed else last, last=last,
            modelled=not observed,
        )

    @property
    def supports_live(self) -> bool:
        return True
