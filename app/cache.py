"""A tiny TTL cache used to keep us off Codeforces' and LeetCode's rate limits.

Why this exists
---------------
Codeforces asks clients not to hit the API more than once per second, and
`user.status` for an active competitor is a multi-megabyte response. A single
"what should I practice today" question touches submissions, the problemset and
several LeetCode question lookups. Without caching, every page refresh would
replay all of that.

Two backends, one interface:

* ``SQLiteCache`` — survives process restarts, so a container restart doesn't
  cost a fresh cold sync. This is the default.
* ``MemoryCache`` — no I/O, used by the test suite.

Design notes
------------
* Values are stored as JSON text, not pickle. Pickle would happily execute code
  from a corrupted cache file; JSON cannot.
* Expiry is *lazy* (checked on read) plus an explicit ``purge_expired()`` the
  scheduler can call. Lazy expiry alone would let a cache file grow forever.
* ``sqlite3`` connections are created with ``check_same_thread=False`` and
  guarded by a lock, because FastAPI serves requests from a thread pool and
  APScheduler runs the sync job on yet another thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MISS = object()


def make_cache_key(method: str, url: str, payload: Any = None) -> str:
    """Build a stable cache key from a request description.

    ``payload`` is serialised with sorted keys so that two logically identical
    requests whose dicts happen to iterate in a different order still collide on
    the same key.
    """
    blob = json.dumps(
        {"method": method.upper(), "url": url, "payload": payload},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache(ABC):
    """Minimal cache interface shared by both backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def purge_expired(self) -> int: ...

    def get_or_set(self, key: str, producer, ttl_seconds: int, *, force_refresh: bool = False):
        """Return the cached value, or call ``producer`` and cache its result.

        ``force_refresh`` is what powers ``POST /sync?force=true`` — the user can
        always bypass a stale entry on demand.
        """
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                logger.debug("cache hit", extra={"cache_key": key})
                return cached

        logger.debug("cache miss", extra={"cache_key": key, "forced": force_refresh})
        value = producer()
        if value is not None:
            self.set(key, value, ttl_seconds)
        return value


class MemoryCache(Cache):
    """In-process dictionary cache. Fast, ephemeral, used in tests."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key, _MISS)
            if entry is _MISS:
                return None
            expires_at, value = entry  # type: ignore[misc]
            if expires_at <= time.time():
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl_seconds, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            stale = [k for k, (exp, _) in self._data.items() if exp <= now]
            for key in stale:
                del self._data[key]
        return len(stale)


class SQLiteCache(Cache):
    """Disk-backed TTL cache. Default backend; survives restarts."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        # WAL lets the scheduler write while a request reads, instead of the
        # reader hitting "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            raw, expires_at = row
            if expires_at <= time.time():
                self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self._conn.commit()
                return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A truncated write should degrade to a cache miss, never a crash.
            logger.warning("discarding corrupt cache entry", extra={"cache_key": key})
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        payload = json.dumps(value, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_entries (key, value, expires_at) VALUES (?, ?, ?)",
                (key, payload, time.time() + ttl_seconds),
            )
            self._conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE expires_at <= ?", (time.time(),)
            )
            self._conn.commit()
            return cursor.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_cache_singleton: Cache | None = None
_singleton_lock = threading.Lock()


def get_cache() -> Cache:
    """Return the process-wide cache instance chosen by configuration."""
    global _cache_singleton
    if _cache_singleton is None:
        with _singleton_lock:
            if _cache_singleton is None:
                from app.config import get_settings

                settings = get_settings()
                if settings.cache_backend == "memory":
                    _cache_singleton = MemoryCache()
                else:
                    _cache_singleton = SQLiteCache(settings.cache_db_path)
                logger.info("cache initialised", extra={"backend": settings.cache_backend})
    return _cache_singleton


def reset_cache() -> None:
    """Drop the singleton. Used by tests to get a clean slate."""
    global _cache_singleton
    with _singleton_lock:
        _cache_singleton = None
