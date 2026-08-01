"""A cache-aware, retrying HTTP client shared by both upstream integrations.

Everything that leaves this process goes through :class:`ResilientClient`, which
layers four concerns in a deliberate order:

1. **Cache** — checked first, so a hit costs zero network and zero rate-limit
   budget.
2. **Rate limit** — a minimum interval between calls to the same host.
   Codeforces explicitly asks for no more than one request per second and will
   return HTTP 403 to clients that ignore that.
3. **Retry with exponential backoff + jitter** — for timeouts, connection
   resets, 429s and 5xxs. Jitter matters: without it, several concurrent
   requests that fail together would retry in lockstep and hammer the API in
   synchronised waves.
4. **Fail loudly** — 4xx responses (bad handle, unknown user) are *not*
   retried, because retrying a permanent error just wastes time and quota.

Splitting these out of the Codeforces/LeetCode clients keeps those files about
domain parsing, which is what the unit tests care about.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.cache import Cache, get_cache, make_cache_key
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Status codes worth trying again: rate limiting and transient server faults.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ExternalAPIError(RuntimeError):
    """Base class for upstream failures surfaced to the API layer."""


class RetryableHTTPError(ExternalAPIError):
    """Transient upstream failure; worth retrying."""


class PermanentHTTPError(ExternalAPIError):
    """Upstream rejected the request; retrying will not help."""


class RateLimiter:
    """Blocks until at least ``min_interval`` has passed since the last call."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        "retrying upstream request",
        extra={
            "attempt": state.attempt_number,
            "sleep_seconds": round(state.idle_for, 2),
            "error": str(exc) if exc else None,
        },
    )


class ResilientClient:
    """Thin wrapper over ``httpx.Client`` adding cache, throttling and retries."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        min_interval_seconds: float = 1.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache if cache is not None else get_cache()
        self._limiter = RateLimiter(min_interval_seconds)

        headers = {"User-Agent": self._settings.http_user_agent, "Accept": "application/json"}
        headers.update(default_headers or {})
        self._client = httpx.Client(
            timeout=self._settings.http_timeout_seconds,
            headers=headers,
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ResilientClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API --------------------------------------------------------
    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> Any:
        """GET a JSON document, using the cache unless ``force_refresh``."""
        key = make_cache_key("GET", url, params)
        ttl = self._settings.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        return self._cache.get_or_set(
            key,
            lambda: self._request_json("GET", url, params=params),
            ttl,
            force_refresh=force_refresh,
        )

    def post_json(
        self,
        url: str,
        *,
        json_body: dict[str, Any],
        headers: dict[str, str] | None = None,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> Any:
        """POST a JSON body and cache the response (used for GraphQL queries)."""
        key = make_cache_key("POST", url, json_body)
        ttl = self._settings.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        return self._cache.get_or_set(
            key,
            lambda: self._request_json("POST", url, json_body=json_body, headers=headers),
            ttl,
            force_refresh=force_refresh,
        )

    # -- internals ---------------------------------------------------------
    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        # Build the retry policy from settings at call time so that tests can
        # dial retries down to 1 and stay fast.
        runner = retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.http_max_retries),
            wait=wait_exponential_jitter(
                initial=self._settings.http_backoff_base_seconds, max=15.0
            ),
            retry=retry_if_exception_type((RetryableHTTPError, httpx.TransportError)),
            before_sleep=_log_retry,
        )
        return runner(self._send)(method, url, params, json_body, headers)

    def _send(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> Any:
        self._limiter.acquire()
        started = time.monotonic()

        response = self._client.request(method, url, params=params, json=json_body, headers=headers)
        elapsed_ms = round((time.monotonic() - started) * 1000)

        if response.status_code in RETRYABLE_STATUS_CODES:
            raise RetryableHTTPError(f"{method} {url} returned {response.status_code}")

        if response.status_code >= 400:
            logger.error(
                "upstream rejected request",
                extra={"url": url, "status": response.status_code, "elapsed_ms": elapsed_ms},
            )
            raise PermanentHTTPError(
                f"{method} {url} returned {response.status_code}: {response.text[:200]}"
            )

        logger.debug(
            "upstream request ok",
            extra={"url": url, "status": response.status_code, "elapsed_ms": elapsed_ms},
        )

        try:
            return response.json()
        except ValueError as exc:
            raise PermanentHTTPError(f"{method} {url} returned non-JSON body") from exc
