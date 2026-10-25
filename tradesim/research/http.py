"""Small HTTP helper for the research feeds (retries, browser user agent, proxy settings from the env)."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36 tradesim-research")


class FetchError(RuntimeError):
    pass


def fetch_text(url: str, timeout: float = 20.0, retries: int = 2, backoff: float = 2.0,
               headers: dict[str, str] | None = None) -> str:
    """GET ``url`` and return the body as text; raises :class:`FetchError` after ``retries`` retries."""
    import requests

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})})
            if r.status_code == 200:
                if "charset" not in r.headers.get("content-type", "").lower():
                    return r.content.decode("utf-8-sig", errors="replace")  # requests would assume latin-1
                return r.text
            last = FetchError(f"HTTP {r.status_code} for {url}")
            if r.status_code in (400, 401, 403, 404):
                break
        except Exception as exc:  # noqa: BLE001 - network errors of any kind are retried
            last = exc
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    raise FetchError(f"{url}: {last}")
