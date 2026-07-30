"""Background sync scheduler.

Without this the app only knows what it knew at the last manual ``POST /sync``.
A coach whose data is a week stale gives bad advice, so APScheduler refreshes
the profile on an interval.

Two details that matter in production:

* **The job takes the same lock as manual syncs.** A scheduled refresh firing
  while the user has just triggered one by hand would otherwise duplicate every
  upstream call. :class:`~app.services.ingestion.IngestionService` owns that
  lock, so both paths serialise naturally.
* **`max_instances=1` plus `coalesce=True`.** If a sync runs longer than the
  interval, APScheduler will not stack up a queue of pending runs — it drops the
  backlog and executes once. Ingestion is idempotent, so skipping a run costs
  nothing; piling them up would hammer the APIs.

Expired cache rows are purged on the same tick, which keeps the SQLite file from
growing without bound.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.cache import get_cache
from app.config import Settings, get_settings
from app.services.ingestion import IngestionService, get_ingestion_service

logger = logging.getLogger(__name__)

SYNC_JOB_ID = "periodic-profile-sync"


def run_sync_job(ingestion: IngestionService | None = None) -> None:
    """One scheduled tick: refresh the profile, then purge expired cache rows."""
    service = ingestion or get_ingestion_service()
    try:
        result = service.sync(force_refresh=True)
        logger.info(
            "scheduled sync complete",
            extra={"records": result.total_records, "ok": result.ok},
        )
    except Exception:  # noqa: BLE001 - an exception here would kill the job thread
        logger.exception("scheduled sync failed")

    try:
        purged = get_cache().purge_expired()
        if purged:
            logger.info("purged expired cache entries", extra={"count": purged})
    except Exception:  # noqa: BLE001 - housekeeping must never break the app
        logger.warning("cache purge failed", exc_info=True)


class SyncScheduler:
    """Owns the APScheduler lifecycle, started and stopped by the app lifespan."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._scheduler: BackgroundScheduler | None = None

    def start(self) -> None:
        if not self._settings.sync_enabled:
            logger.info("scheduled sync disabled by configuration")
            return
        if self._scheduler is not None:
            return

        self._scheduler = BackgroundScheduler(daemon=True)
        self._scheduler.add_job(
            run_sync_job,
            trigger="interval",
            minutes=self._settings.sync_interval_minutes,
            id=SYNC_JOB_ID,
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "sync scheduler started",
            extra={"interval_minutes": self._settings.sync_interval_minutes},
        )

        if self._settings.sync_on_startup:
            # Fire immediately on a background thread so startup is not blocked
            # by a cold sync that can take tens of seconds.
            self._scheduler.add_job(run_sync_job, id=f"{SYNC_JOB_ID}-startup")
            logger.info("startup sync queued")

    def shutdown(self) -> None:
        if self._scheduler is None:
            return
        # wait=False: do not hold up shutdown for an in-flight sync. The next
        # start will simply re-run it.
        self._scheduler.shutdown(wait=False)
        self._scheduler = None
        logger.info("sync scheduler stopped")
