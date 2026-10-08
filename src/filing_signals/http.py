"""A polite EDGAR client: declared User-Agent, rate limit, retries.

SEC fair-access policy: at most 10 requests per second, and every request must
declare who is making it (a name and a contact email in the User-Agent).
Undeclared clients get a 403. The User-Agent comes from the EDGAR_USER_AGENT
environment variable so no contact address is ever committed.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field

import requests

DEFAULT_RATE = 8.0          # requests/second: headroom under the SEC's 10
RETRY_STATUSES = {429, 500, 502, 503, 504}


class ConfigError(RuntimeError):
    pass


def user_agent_from_env() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if "@" not in ua:
        raise ConfigError(
            "Set EDGAR_USER_AGENT to '<project or name> <contact email>'; "
            "the SEC rejects requests without a declared contact."
        )
    return ua


class RateLimiter:
    """Thread-safe: hands out one slot every 1/rate seconds."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self._next = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


@dataclass
class Stats:
    requests: int = 0
    retries: int = 0
    bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, requests: int = 0, retries: int = 0, nbytes: int = 0) -> None:
        with self._lock:
            self.requests += requests
            self.retries += retries
            self.bytes += nbytes


class NotFound(Exception):
    pass


class EdgarClient:
    def __init__(self, user_agent: str, rate: float = DEFAULT_RATE, max_tries: int = 6,
                 session: requests.Session | None = None):
        self.limiter = RateLimiter(rate)
        self.max_tries = max_tries
        self.stats = Stats()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def get(self, url: str) -> bytes:
        """GET with rate limiting and exponential backoff. 404 raises NotFound."""
        for attempt in range(1, self.max_tries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=60)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.max_tries:
                    raise
                self._backoff(attempt, None)
                continue
            self.stats.add(requests=1)
            if resp.status_code == 200:
                self.stats.add(nbytes=len(resp.content))
                return resp.content
            if resp.status_code == 404:
                raise NotFound(url)
            if resp.status_code == 403:
                # the SEC answers an undeclared or over-limit client with 403
                raise ConfigError(f"403 from SEC for {url}: check EDGAR_USER_AGENT and rate")
            if resp.status_code in RETRY_STATUSES and attempt < self.max_tries:
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"gave up on {url}")

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        self.stats.add(retries=1)
        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
        time.sleep(delay + random.uniform(0, 0.5))
