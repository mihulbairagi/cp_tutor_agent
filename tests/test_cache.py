"""Tests for the TTL cache backends and key derivation."""

from __future__ import annotations

import time

import pytest

from app.cache import MemoryCache, SQLiteCache, make_cache_key


@pytest.fixture(params=["memory", "sqlite"])
def any_cache(request, tmp_path):
    """Run the shared contract against both backends."""
    if request.param == "memory":
        return MemoryCache()
    return SQLiteCache(tmp_path / "cache.db")


def test_set_then_get_round_trips(any_cache) -> None:
    any_cache.set("k", {"status": "OK", "result": [1, 2, 3]}, 60)
    assert any_cache.get("k") == {"status": "OK", "result": [1, 2, 3]}


def test_missing_key_is_a_miss(any_cache) -> None:
    assert any_cache.get("nope") is None


def test_entries_expire(any_cache) -> None:
    any_cache.set("k", "v", 0)
    time.sleep(0.01)
    assert any_cache.get("k") is None


def test_delete_and_clear(any_cache) -> None:
    any_cache.set("a", 1, 60)
    any_cache.set("b", 2, 60)
    any_cache.delete("a")
    assert any_cache.get("a") is None
    assert any_cache.get("b") == 2
    any_cache.clear()
    assert any_cache.get("b") is None


def test_purge_expired_reports_count(any_cache) -> None:
    any_cache.set("fresh", 1, 60)
    any_cache.set("stale", 2, 0)
    time.sleep(0.01)
    assert any_cache.purge_expired() == 1
    assert any_cache.get("fresh") == 1


def test_get_or_set_only_calls_producer_on_a_miss(any_cache) -> None:
    calls = []

    def producer():
        calls.append(1)
        return "value"

    assert any_cache.get_or_set("k", producer, 60) == "value"
    assert any_cache.get_or_set("k", producer, 60) == "value"
    assert len(calls) == 1, "second call should have been served from cache"


def test_force_refresh_bypasses_a_live_entry(any_cache) -> None:
    """This is what powers POST /sync?force=true."""
    any_cache.set("k", "old", 60)
    assert any_cache.get_or_set("k", lambda: "new", 60, force_refresh=True) == "new"
    assert any_cache.get("k") == "new"


def test_sqlite_cache_survives_reopening(tmp_path) -> None:
    """A container restart must not force a cold re-sync."""
    path = tmp_path / "cache.db"
    SQLiteCache(path).set("k", {"a": 1}, 60)
    assert SQLiteCache(path).get("k") == {"a": 1}


class TestCacheKeys:
    def test_param_order_does_not_change_the_key(self) -> None:
        left = make_cache_key("GET", "https://x/api", {"handle": "a", "count": 10})
        right = make_cache_key("GET", "https://x/api", {"count": 10, "handle": "a"})
        assert left == right

    def test_different_params_produce_different_keys(self) -> None:
        left = make_cache_key("GET", "https://x/api", {"handle": "a"})
        right = make_cache_key("GET", "https://x/api", {"handle": "b"})
        assert left != right

    def test_method_is_part_of_the_key(self) -> None:
        assert make_cache_key("GET", "https://x") != make_cache_key("POST", "https://x")
