"""Tests for the resilient HTTP layer: retries, backoff and cache integration.

``respx`` intercepts httpx at the transport level, so these exercise the real
``ResilientClient`` code path — including the tenacity policy — with no network.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.cache import MemoryCache
from app.clients.http import (
    PermanentHTTPError,
    RateLimiter,
    ResilientClient,
    RetryableHTTPError,
)
from app.config import Settings

URL = "https://example.test/api"


@pytest.fixture
def client(settings: Settings) -> ResilientClient:
    # No throttling in tests: rate limiting is asserted separately.
    return ResilientClient(settings=settings, cache=MemoryCache(), min_interval_seconds=0.0)


@respx.mock
def test_successful_response_is_returned(client) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, json={"status": "OK"}))
    assert client.get_json(URL) == {"status": "OK"}


@respx.mock
def test_a_second_identical_request_is_served_from_cache(client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"n": 1}))
    client.get_json(URL)
    client.get_json(URL)
    assert route.call_count == 1, "the cache should have absorbed the second call"


@respx.mock
def test_force_refresh_bypasses_the_cache(client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"n": 1}))
    client.get_json(URL)
    client.get_json(URL, force_refresh=True)
    assert route.call_count == 2


@respx.mock
def test_different_params_are_cached_separately(client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"n": 1}))
    client.get_json(URL, params={"handle": "a"})
    client.get_json(URL, params={"handle": "b"})
    assert route.call_count == 2


@respx.mock
def test_server_errors_are_retried_then_succeed(client) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json={"status": "OK"}),
        ]
    )
    assert client.get_json(URL) == {"status": "OK"}
    assert route.call_count == 2


@respx.mock
def test_rate_limiting_is_retried(client) -> None:
    """429 is the response Codeforces gives when you ignore its limits."""
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )
    assert client.get_json(URL) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_retries_are_bounded_and_then_reraise(client) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(RetryableHTTPError):
        client.get_json(URL)
    assert route.call_count == 2  # HTTP_MAX_RETRIES is 2 in the test environment


@respx.mock
def test_client_errors_are_not_retried(client) -> None:
    """Retrying a bad handle just burns quota — fail fast instead."""
    route = respx.get(URL).mock(return_value=httpx.Response(404, text="not found"))
    with pytest.raises(PermanentHTTPError):
        client.get_json(URL)
    assert route.call_count == 1


@respx.mock
def test_transport_errors_are_retried(client) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"ok": True})]
    )
    assert client.get_json(URL) == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_a_failed_request_is_not_cached(client) -> None:
    respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(PermanentHTTPError):
        client.get_json(URL)

    respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    assert client.get_json(URL) == {"ok": True}


@respx.mock
def test_non_json_body_is_a_permanent_error(client) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(PermanentHTTPError, match="non-JSON"):
        client.get_json(URL)


@respx.mock
def test_post_json_sends_the_body_and_caches_on_it(client) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"data": {}}))
    client.post_json(URL, json_body={"query": "a", "variables": {}})
    client.post_json(URL, json_body={"query": "a", "variables": {}})
    client.post_json(URL, json_body={"query": "b", "variables": {}})
    assert route.call_count == 2, "identical GraphQL queries should share a cache entry"


def test_rate_limiter_spaces_calls_out() -> None:
    import time

    limiter = RateLimiter(0.05)
    started = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    assert time.monotonic() - started >= 0.05
