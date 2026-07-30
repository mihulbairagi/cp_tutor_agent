"""Ingestion: pull both platforms, merge, persist, index.

This is the orchestration layer. It owns three responsibilities:

1. **Fault isolation.** Each source is fetched through its ``safe_fetch_*``
   wrapper, so a LeetCode outage degrades the sync to Codeforces-only and is
   reported per-source in the response instead of failing the whole request.
2. **Persistence of the merged profile.** The vector store is optimised for
   *similarity* lookups, not for "give me every record so I can count topics".
   Rather than stand up a second database for that, the merged record list is
   written to the existing SQLite cache under a long TTL. A container restart
   therefore keeps a usable profile without re-hitting either API.
3. **Indexing.** Records are upserted into ChromaDB for semantic search.

Keeping this separate from the analysis service matters: analysis stays a pure
function of a record list, which is why it can be unit-tested without any I/O.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from app.cache import Cache, get_cache
from app.clients.codeforces import CodeforcesClient
from app.clients.leetcode import LeetCodeClient
from app.config import Settings, get_settings
from app.models import ProblemRecord, SourceSyncResult, SyncResult, UserRating
from app.store.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Versioned keys so a change to ProblemRecord's shape invalidates old entries
# rather than deserialising into something subtly wrong.
RECORDS_KEY = "profile:records:v1"
RATING_KEY = "profile:rating:v1"
LAST_SYNC_KEY = "profile:last_sync:v1"

# The merged profile is derived data we can always rebuild, but it should
# comfortably outlive the raw API caches so restarts stay warm.
PROFILE_TTL_SECONDS = 30 * 24 * 3600


class IngestionService:
    """Fetches, merges and indexes the user's problem history."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: Cache | None = None,
        codeforces: CodeforcesClient | None = None,
        leetcode: LeetCodeClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache if cache is not None else get_cache()
        self._codeforces = codeforces or CodeforcesClient(settings=self._settings)
        self._leetcode = leetcode or LeetCodeClient(settings=self._settings)
        self._vector_store = vector_store or get_vector_store()
        # A sync can take tens of seconds. The lock stops the scheduler and a
        # manual POST /sync from duplicating the work and racing on the cache.
        self._sync_lock = threading.Lock()

    # -- read side ---------------------------------------------------------
    def get_records(self) -> list[ProblemRecord]:
        """Return the merged profile, or an empty list if nothing is synced yet."""
        raw = self._cache.get(RECORDS_KEY)
        if not raw:
            return []
        return [ProblemRecord.model_validate(item) for item in raw]

    def get_user_rating(self) -> UserRating:
        raw = self._cache.get(RATING_KEY)
        if not raw:
            return UserRating(handle=self._settings.codeforces_handle)
        return UserRating.model_validate(raw)

    def get_last_sync(self) -> datetime | None:
        raw = self._cache.get(LAST_SYNC_KEY)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return None

    def has_data(self) -> bool:
        return bool(self._cache.get(RECORDS_KEY))

    def ensure_synced(self) -> None:
        """Sync on first use so a fresh deployment answers questions immediately."""
        if not self.has_data():
            logger.info("no cached profile found; running initial sync")
            self.sync()

    # -- write side --------------------------------------------------------
    def sync(self, *, force_refresh: bool = False, reset_index: bool = False) -> SyncResult:
        """Fetch both platforms, merge, persist and index."""
        with self._sync_lock:
            started = datetime.now(UTC)
            logger.info(
                "sync started",
                extra={
                    "force_refresh": force_refresh,
                    "codeforces_handle": self._settings.codeforces_handle,
                    "leetcode_username": self._settings.leetcode_username,
                },
            )

            if reset_index:
                try:
                    self._vector_store.reset()
                except Exception:  # noqa: BLE001 - a missing collection is fine
                    logger.warning("vector store reset skipped", exc_info=True)

            source_results: list[SourceSyncResult] = []
            records: list[ProblemRecord] = []

            cf_records, cf_error = self._fetch_codeforces(force_refresh=force_refresh)
            records.extend(cf_records)
            source_results.append(self._summarise("codeforces", cf_records, cf_error))

            lc_records, lc_error = self._leetcode.safe_fetch_solved()
            records.extend(lc_records)
            source_results.append(self._summarise("leetcode", lc_records, lc_error))

            merged = self._merge(records)

            # Only persist when something came back. Overwriting a good profile
            # with an empty one because both APIs were down would be worse than
            # serving slightly stale data.
            if merged:
                self._cache.set(
                    RECORDS_KEY,
                    [record.model_dump(mode="json") for record in merged],
                    PROFILE_TTL_SECONDS,
                )
                self._cache.set(LAST_SYNC_KEY, started.isoformat(), PROFILE_TTL_SECONDS)
            elif any(result.ok for result in source_results):
                logger.warning("sync produced no records; keeping previous profile")

            indexed = 0
            if merged:
                try:
                    indexed = self._vector_store.upsert_records(merged)
                except Exception:  # noqa: BLE001 - search is degraded, data is not lost
                    logger.exception("indexing failed; profile saved without semantic search")

            finished = datetime.now(UTC)
            result = SyncResult(
                started_at=started,
                finished_at=finished,
                duration_seconds=round((finished - started).total_seconds(), 3),
                sources=source_results,
                total_records=len(merged),
                indexed=indexed,
            )
            logger.info(
                "sync finished",
                extra={
                    "records": result.total_records,
                    "indexed": indexed,
                    "duration_seconds": result.duration_seconds,
                    "ok": result.ok,
                },
            )
            return result

    # -- internals ---------------------------------------------------------
    def _fetch_codeforces(self, *, force_refresh: bool) -> tuple[list[ProblemRecord], str]:
        records, error = self._codeforces.safe_fetch_submissions()
        if error:
            return records, error
        try:
            rating = self._codeforces.fetch_user_rating(force_refresh=force_refresh)
            self._cache.set(RATING_KEY, rating.model_dump(mode="json"), PROFILE_TTL_SECONDS)
        except Exception:  # noqa: BLE001 - rating only tunes the difficulty band
            logger.warning("could not refresh Codeforces rating", exc_info=True)
        return records, ""

    @staticmethod
    def _summarise(source: str, records: list[ProblemRecord], error: str) -> SourceSyncResult:
        return SourceSyncResult(
            source=source,  # type: ignore[arg-type]
            fetched=len(records),
            solved=sum(1 for record in records if record.solved),
            attempted_unsolved=sum(1 for record in records if not record.solved),
            ok=not error,
            detail=error or "ok",
        )

    @staticmethod
    def _merge(records: list[ProblemRecord]) -> list[ProblemRecord]:
        """De-duplicate by problem id, keeping the most informative version.

        Problem ids are namespaced per platform (``cf:`` / ``lc:``) so the two
        sources cannot collide. This guards against duplicates *within* a source,
        which the LeetCode public path can produce when the same problem appears
        twice in the recent-submissions feed.
        """
        merged: dict[str, ProblemRecord] = {}
        for record in records:
            existing = merged.get(record.problem_id)
            if existing is None:
                merged[record.problem_id] = record
                continue
            existing.attempt_count += record.attempt_count
            if record.solved and not existing.solved:
                existing.solved = True
                existing.solved_at = record.solved_at
            if record.tags and not existing.tags:
                existing.tags = record.tags
                existing.raw_tags = record.raw_tags
        return list(merged.values())


_service_singleton: IngestionService | None = None
_singleton_lock = threading.Lock()


def get_ingestion_service() -> IngestionService:
    global _service_singleton
    if _service_singleton is None:
        with _singleton_lock:
            if _service_singleton is None:
                _service_singleton = IngestionService()
    return _service_singleton
