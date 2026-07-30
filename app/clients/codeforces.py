"""Codeforces ingestion.

Codeforces publishes a clean, unauthenticated, well-documented REST API, so this
is the richer of the two sources and carries most of the analysis:

* ``user.status``           — every submission the user ever made
* ``user.info``             — current and peak rating
* ``problemset.problems``   — the full problem archive, our recommendation pool

The design decision worth calling out
-------------------------------------
Most "track my competitive programming" projects filter ``user.status`` down to
``verdict == "OK"`` and throw the rest away. We keep the failures. A problem you
submitted to eleven times and never solved is a far sharper weakness signal than
a topic you simply have not touched, and it is sitting right there in the same
response. :func:`parse_submissions` therefore folds *all* submissions into one
:class:`~app.models.ProblemRecord` per problem, carrying both the solved flag
and the attempt count.

Every function that turns JSON into domain objects is pure and takes a plain
dict, so the unit tests exercise real recorded payloads without touching the
network.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.clients.http import PermanentHTTPError, ResilientClient
from app.config import Settings, get_settings
from app.models import CandidateProblem, ProblemRecord, UserRating
from app.services.taxonomy import difficulty_from_rating, normalize_tags

logger = logging.getLogger(__name__)

API_BASE = "https://codeforces.com/api"


class CodeforcesError(RuntimeError):
    """Codeforces returned ``status: FAILED`` (bad handle, unknown contest...)."""


def _unwrap(payload: dict) -> object:
    """Return ``result`` from a Codeforces envelope, or raise with its comment.

    Codeforces signals errors *inside* a 200 response body, so checking the HTTP
    status alone is not enough.
    """
    if not isinstance(payload, dict):
        raise CodeforcesError("expected a JSON object from Codeforces")
    if payload.get("status") != "OK":
        raise CodeforcesError(payload.get("comment") or "Codeforces request failed")
    return payload.get("result")


def _problem_key(problem: dict) -> str:
    """Stable id for a problem across submissions and the problemset."""
    contest_id = problem.get("contestId")
    index = problem.get("index", "")
    if contest_id is not None:
        return f"cf:{contest_id}/{index}"
    # Gym and a handful of archive problems have no contestId.
    return f"cf:{problem.get('problemsetName', 'unknown')}/{problem.get('name', '')}"


def _problem_url(problem: dict) -> str:
    contest_id = problem.get("contestId")
    index = problem.get("index", "")
    if contest_id is None:
        return "https://codeforces.com/problemset"
    return f"https://codeforces.com/problemset/problem/{contest_id}/{index}"


def _to_datetime(seconds: int | None) -> datetime | None:
    if not seconds:
        return None
    return datetime.fromtimestamp(seconds, tz=UTC)


def parse_submissions(result: list[dict]) -> list[ProblemRecord]:
    """Fold a ``user.status`` payload into one record per problem.

    Codeforces returns submissions newest-first and repeats a problem once per
    submission, so we aggregate:

    * ``solved``          — true if *any* submission got ``OK``
    * ``solved_at``       — the **earliest** accepted submission, since that is
      when the user actually learned the topic; using the latest would make
      re-submitting an old solve look like fresh practice
    * ``attempt_count``   — every submission, including compile errors
    * ``last_attempt_at`` — the most recent activity of any kind
    """
    records: dict[str, ProblemRecord] = {}

    for submission in result or []:
        problem = submission.get("problem")
        if not isinstance(problem, dict):
            continue

        key = _problem_key(problem)
        verdict = submission.get("verdict")
        accepted = verdict == "OK"
        moment = _to_datetime(submission.get("creationTimeSeconds"))
        rating = problem.get("rating")
        raw_tags = list(problem.get("tags") or [])

        record = records.get(key)
        if record is None:
            records[key] = ProblemRecord(
                source="codeforces",
                problem_id=key,
                name=problem.get("name") or key,
                url=_problem_url(problem),
                tags=normalize_tags(raw_tags),
                raw_tags=raw_tags,
                rating=rating,
                difficulty=difficulty_from_rating(rating),
                solved=accepted,
                attempt_count=1,
                solved_at=moment if accepted else None,
                last_attempt_at=moment,
            )
            continue

        record.attempt_count += 1
        if accepted:
            record.solved = True
            if record.solved_at is None or (moment and moment < record.solved_at):
                record.solved_at = moment
        if moment and (record.last_attempt_at is None or moment > record.last_attempt_at):
            record.last_attempt_at = moment

    return list(records.values())


def parse_user_info(result: list[dict], handle: str) -> UserRating:
    """Extract the rating snapshot from a ``user.info`` payload."""
    if not result:
        return UserRating(handle=handle)
    info = result[0]
    return UserRating(
        handle=info.get("handle", handle),
        rating=info.get("rating"),
        max_rating=info.get("maxRating"),
        rank=info.get("rank"),
    )


def parse_problemset(result: dict) -> list[CandidateProblem]:
    """Turn ``problemset.problems`` into recommendation candidates.

    The endpoint returns problems and their statistics as two parallel lists
    rather than nested objects, so we index the statistics by problem key and
    join them.
    """
    problems = (result or {}).get("problems") or []
    statistics = (result or {}).get("problemStatistics") or []

    solved_counts = {_problem_key(stat): stat.get("solvedCount", 0) for stat in statistics}

    candidates: list[CandidateProblem] = []
    for problem in problems:
        key = _problem_key(problem)
        raw_tags = list(problem.get("tags") or [])
        candidates.append(
            CandidateProblem(
                problem_id=key,
                contest_id=problem.get("contestId"),
                index=problem.get("index", ""),
                name=problem.get("name") or key,
                url=_problem_url(problem),
                rating=problem.get("rating"),
                tags=normalize_tags(raw_tags),
                raw_tags=raw_tags,
                solved_count=solved_counts.get(key, 0),
            )
        )
    return candidates


class CodeforcesClient:
    """Network-facing wrapper. All parsing is delegated to the pure functions."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: ResilientClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Codeforces asks for at most one request per second from a given client.
        self._client = client or ResilientClient(
            settings=self._settings, min_interval_seconds=1.0
        )

    def fetch_submissions(
        self, handle: str | None = None, *, force_refresh: bool = False
    ) -> list[ProblemRecord]:
        handle = handle or self._settings.codeforces_handle
        if not handle:
            logger.warning("no Codeforces handle configured; skipping")
            return []

        payload = self._client.get_json(
            f"{API_BASE}/user.status",
            params={"handle": handle, "from": 1, "count": 100_000},
            ttl_seconds=self._settings.cache_ttl_seconds,
            force_refresh=force_refresh,
        )
        result = _unwrap(payload)
        records = parse_submissions(result if isinstance(result, list) else [])
        logger.info(
            "fetched Codeforces submissions",
            extra={
                "handle": handle,
                "problems": len(records),
                "solved": sum(1 for r in records if r.solved),
            },
        )
        return records

    def fetch_user_rating(
        self, handle: str | None = None, *, force_refresh: bool = False
    ) -> UserRating:
        handle = handle or self._settings.codeforces_handle
        if not handle:
            return UserRating(handle="")

        payload = self._client.get_json(
            f"{API_BASE}/user.info",
            params={"handles": handle},
            ttl_seconds=self._settings.cache_ttl_seconds,
            force_refresh=force_refresh,
        )
        result = _unwrap(payload)
        rating = parse_user_info(result if isinstance(result, list) else [], handle)
        logger.info("fetched Codeforces rating", extra={"handle": handle, "rating": rating.rating})
        return rating

    def fetch_problemset(self, *, force_refresh: bool = False) -> list[CandidateProblem]:
        """Fetch the whole problem archive (long TTL — it rarely changes)."""
        payload = self._client.get_json(
            f"{API_BASE}/problemset.problems",
            ttl_seconds=self._settings.problemset_cache_ttl_seconds,
            force_refresh=force_refresh,
        )
        result = _unwrap(payload)
        candidates = parse_problemset(result if isinstance(result, dict) else {})
        logger.info("fetched Codeforces problemset", extra={"problems": len(candidates)})
        return candidates

    def safe_fetch_submissions(self, handle: str | None = None) -> tuple[list[ProblemRecord], str]:
        """Fetch submissions, converting failures into a message.

        A LeetCode outage should not take down the Codeforces half of a sync, and
        vice versa, so the ingestion service uses these tolerant wrappers and
        reports per-source status.
        """
        try:
            return self.fetch_submissions(handle), ""
        except (CodeforcesError, PermanentHTTPError) as exc:
            logger.error("Codeforces sync failed", extra={"error": str(exc)})
            return [], str(exc)
        except Exception as exc:  # noqa: BLE001 - one source must not kill the sync
            logger.exception("unexpected Codeforces failure")
            return [], f"unexpected error: {exc}"
