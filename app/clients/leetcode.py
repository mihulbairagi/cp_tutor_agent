"""LeetCode ingestion via the public GraphQL endpoint.

LeetCode has no official REST API. The queries below are the same ones the site
itself issues, and match the shapes documented by the widely-used open-source
wrapper `alfaarghya/alfa-leetcode-api`.

The honest limitation
---------------------
The **unauthenticated** endpoint will only tell you a user's *aggregate* solved
counts and their ~20 most recent accepted submissions. There is no public way to
page a stranger's entire solved list — that data is behind a login.

Rather than pretend otherwise, this client has two modes and logs which one it
used:

* **public** (default) — ``recentAcSubmissionList`` for the recent solves, then
  ``question(titleSlug:)`` to enrich each one with difficulty and topic tags.
  Aggregate counts still come through, so LeetCode contributes real signal even
  in this mode.
* **authenticated** — if ``LEETCODE_SESSION`` is set in the environment, the
  client pages ``problemsetQuestionList`` filtered to ``status: AC`` and gets the
  complete solved history with tags in one pass.

Because Codeforces exposes full history publicly and is the stronger dataset,
the analyser treats LeetCode as a supplementary source and degrades cleanly when
only the public slice is available.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.clients.http import PermanentHTTPError, ResilientClient
from app.config import Settings, get_settings
from app.models import ProblemRecord
from app.services.taxonomy import normalize_tags

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://leetcode.com/graphql"
PROBLEM_URL_TEMPLATE = "https://leetcode.com/problems/{slug}/"

# Page size for the authenticated solved-list query. LeetCode rejects very large
# limits, so we page in chunks.
SOLVED_PAGE_SIZE = 100

USER_PROFILE_QUERY = """
query userProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile { ranking reputation }
    submitStatsGlobal {
      acSubmissionNum { difficulty count }
      totalSubmissionNum { difficulty count }
    }
  }
}
"""

RECENT_AC_QUERY = """
query recentAcSubmissions($username: String!, $limit: Int!) {
  recentAcSubmissionList(username: $username, limit: $limit) {
    id
    title
    titleSlug
    timestamp
  }
}
"""

QUESTION_DETAIL_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionFrontendId
    title
    titleSlug
    difficulty
    topicTags { name slug }
  }
}
"""

SOLVED_LIST_QUERY = """
query problemsetQuestionList(
  $categorySlug: String
  $limit: Int
  $skip: Int
  $filters: QuestionListFilterInput
) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      questionFrontendId
      title
      titleSlug
      difficulty
      status
      topicTags { name slug }
    }
  }
}
"""


class LeetCodeError(RuntimeError):
    """LeetCode returned GraphQL errors or an unknown user."""


def _unwrap(payload: dict) -> dict:
    """Return ``data`` from a GraphQL envelope, raising on GraphQL errors.

    GraphQL reports failures with HTTP 200 and an ``errors`` array, so the
    transport layer cannot catch these for us.
    """
    if not isinstance(payload, dict):
        raise LeetCodeError("expected a JSON object from LeetCode")
    if payload.get("errors"):
        messages = "; ".join(
            str(err.get("message", err)) for err in payload["errors"] if isinstance(err, dict)
        )
        raise LeetCodeError(messages or "LeetCode GraphQL error")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise LeetCodeError("LeetCode response contained no data")
    return data


def _record_from_question(
    question: dict, *, solved_at: datetime | None = None, solved: bool = True
) -> ProblemRecord:
    """Build a ProblemRecord from a LeetCode question object."""
    slug = question.get("titleSlug") or ""
    raw_tags = [tag.get("name", "") for tag in question.get("topicTags") or []]
    return ProblemRecord(
        source="leetcode",
        problem_id=f"lc:{slug}",
        name=question.get("title") or slug,
        url=PROBLEM_URL_TEMPLATE.format(slug=slug),
        tags=normalize_tags(raw_tags),
        raw_tags=raw_tags,
        rating=None,  # LeetCode exposes no numeric rating
        difficulty=question.get("difficulty"),
        solved=solved,
        attempt_count=1,
        solved_at=solved_at,
        last_attempt_at=solved_at,
    )


def parse_profile_stats(data: dict) -> dict[str, int]:
    """Extract ``{difficulty: solved_count}`` from a ``matchedUser`` payload."""
    user = data.get("matchedUser")
    if not user:
        raise LeetCodeError("LeetCode user not found")
    stats = (user.get("submitStatsGlobal") or {}).get("acSubmissionNum") or []
    return {entry.get("difficulty", "?"): entry.get("count", 0) for entry in stats}


def parse_recent_accepted(data: dict) -> list[dict]:
    """Extract the recent accepted submission list (slug + timestamp)."""
    submissions = data.get("recentAcSubmissionList") or []
    parsed: list[dict] = []
    for item in submissions:
        slug = item.get("titleSlug")
        if not slug:
            continue
        raw_ts = item.get("timestamp")
        moment: datetime | None = None
        if raw_ts is not None:
            try:
                moment = datetime.fromtimestamp(int(raw_ts), tz=UTC)
            except (TypeError, ValueError, OSError):
                moment = None
        parsed.append({"titleSlug": slug, "title": item.get("title") or slug, "solved_at": moment})
    return parsed


def parse_question_detail(data: dict) -> dict | None:
    """Extract the question object (difficulty + topicTags) from a detail query."""
    question = data.get("question")
    return question if isinstance(question, dict) else None


def parse_solved_list(data: dict) -> list[ProblemRecord]:
    """Parse the authenticated ``problemsetQuestionList`` page into records.

    Defensive filter on ``status == "ac"``: the same query is used for browsing
    all problems, so relying on the caller having passed the right filter would
    be fragile.
    """
    payload = data.get("problemsetQuestionList") or {}
    questions = payload.get("questions") or []
    records: list[ProblemRecord] = []
    for question in questions:
        status = (question.get("status") or "").lower()
        if status and status != "ac":
            continue
        records.append(_record_from_question(question))
    return records


def parse_solved_total(data: dict) -> int:
    """Total number of matches reported by ``problemsetQuestionList``."""
    return int((data.get("problemsetQuestionList") or {}).get("total") or 0)


class LeetCodeClient:
    """GraphQL client that picks the public or authenticated path automatically."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: ResilientClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        headers = {
            "Content-Type": "application/json",
            # LeetCode's edge rejects GraphQL POSTs without a matching Referer.
            "Referer": "https://leetcode.com",
            "Origin": "https://leetcode.com",
        }
        if self._settings.leetcode_authenticated:
            headers["Cookie"] = f"LEETCODE_SESSION={self._settings.leetcode_session}"
        self._client = client or ResilientClient(
            settings=self._settings,
            min_interval_seconds=0.5,
            default_headers=headers,
        )

    def _query(
        self,
        query: str,
        variables: dict,
        *,
        ttl_seconds: int | None = None,
        force_refresh: bool = False,
    ) -> dict:
        payload = self._client.post_json(
            GRAPHQL_URL,
            json_body={"query": query, "variables": variables},
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )
        return _unwrap(payload)

    # -- public path -------------------------------------------------------
    def fetch_profile_stats(
        self, username: str | None = None, *, force_refresh: bool = False
    ) -> dict[str, int]:
        username = username or self._settings.leetcode_username
        data = self._query(USER_PROFILE_QUERY, {"username": username}, force_refresh=force_refresh)
        return parse_profile_stats(data)

    def fetch_question(self, slug: str, *, force_refresh: bool = False) -> dict | None:
        """Look up one question's difficulty and tags.

        Cached for a full day: a problem's tags essentially never change, and
        enriching 20 recent solves would otherwise mean 20 requests per sync.
        """
        data = self._query(
            QUESTION_DETAIL_QUERY,
            {"titleSlug": slug},
            ttl_seconds=self._settings.problemset_cache_ttl_seconds,
            force_refresh=force_refresh,
        )
        return parse_question_detail(data)

    def fetch_recent_solved(
        self, username: str | None = None, *, limit: int = 20, force_refresh: bool = False
    ) -> list[ProblemRecord]:
        """Recent accepted submissions, enriched with difficulty and tags."""
        username = username or self._settings.leetcode_username
        data = self._query(
            RECENT_AC_QUERY,
            {"username": username, "limit": limit},
            force_refresh=force_refresh,
        )
        recent = parse_recent_accepted(data)

        records: list[ProblemRecord] = []
        seen: set[str] = set()
        for item in recent:
            slug = item["titleSlug"]
            if slug in seen:
                continue
            seen.add(slug)
            question = self.fetch_question(slug, force_refresh=force_refresh)
            if question is None:
                # Enrichment failed; keep the solve, lose only the tags.
                question = {"titleSlug": slug, "title": item["title"], "topicTags": []}
            records.append(_record_from_question(question, solved_at=item["solved_at"]))
        return records

    # -- authenticated path ------------------------------------------------
    def fetch_all_solved(self, *, force_refresh: bool = False) -> list[ProblemRecord]:
        """Page the complete solved list. Requires ``LEETCODE_SESSION``."""
        records: list[ProblemRecord] = []
        skip = 0
        while True:
            data = self._query(
                SOLVED_LIST_QUERY,
                {
                    "categorySlug": "",
                    "limit": SOLVED_PAGE_SIZE,
                    "skip": skip,
                    "filters": {"status": "AC"},
                },
                force_refresh=force_refresh,
            )
            page = parse_solved_list(data)
            records.extend(page)
            total = parse_solved_total(data)
            skip += SOLVED_PAGE_SIZE
            if not page or skip >= total:
                break
        return records

    # -- entry point -------------------------------------------------------
    def fetch_solved(
        self, username: str | None = None, *, force_refresh: bool = False
    ) -> list[ProblemRecord]:
        """Fetch solved problems using the best path available."""
        username = username or self._settings.leetcode_username
        if not username:
            logger.warning("no LeetCode username configured; skipping")
            return []

        if self._settings.leetcode_authenticated:
            records = self.fetch_all_solved(force_refresh=force_refresh)
            mode = "authenticated"
        else:
            records = self.fetch_recent_solved(username, force_refresh=force_refresh)
            mode = "public"

        logger.info(
            "fetched LeetCode solves",
            extra={"username": username, "mode": mode, "problems": len(records)},
        )
        return records

    def safe_fetch_solved(self, username: str | None = None) -> tuple[list[ProblemRecord], str]:
        """Fetch solved problems, converting failures into a message."""
        try:
            return self.fetch_solved(username), ""
        except (LeetCodeError, PermanentHTTPError) as exc:
            logger.error("LeetCode sync failed", extra={"error": str(exc)})
            return [], str(exc)
        except Exception as exc:  # noqa: BLE001 - one source must not kill the sync
            logger.exception("unexpected LeetCode failure")
            return [], f"unexpected error: {exc}"
