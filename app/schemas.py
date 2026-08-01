"""API request and response schemas.

These are intentionally separate from the domain models in :mod:`app.models`.
The domain layer mirrors what Codeforces and LeetCode actually return; this
layer is the contract with HTTP clients. Keeping them apart means an upstream
change does not automatically become a breaking API change, and it lets the API
present a friendlier shape than the internal one.

Every model carries a ``json_schema_extra`` example. That is what turns the
generated ``/docs`` page from a list of field names into something a consumer
can actually read and copy from — real values, not ``"string"`` placeholders.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import Recommendation, SyncResult, WeakTopic

Intent = Literal["weak_topics", "recommend", "search", "profile", "general"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness and readiness at a glance."""

    status: Literal["ok", "degraded"] = Field(description="'degraded' when no data is synced yet.")
    version: str
    codeforces_handle: str
    leetcode_username: str
    llm_enabled: bool = Field(description="False means answers come from templates.")
    leetcode_full_history: bool = Field(
        description="True when a LEETCODE_SESSION cookie unlocks the complete solved list."
    )
    indexed_problems: int = Field(description="Documents currently in the vector store.")
    last_sync_at: datetime | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "codeforces_handle": "Jenitsu444",
                "leetcode_username": "Mihul",
                "llm_enabled": False,
                "leetcode_full_history": False,
                "indexed_problems": 412,
                "last_sync_at": "2026-07-30T09:14:22Z",
            }
        }
    )


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------
class SyncResponse(BaseModel):
    """Per-source outcome of an ingestion run."""

    result: SyncResult

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": {
                    "started_at": "2026-07-30T09:14:02Z",
                    "finished_at": "2026-07-30T09:14:22Z",
                    "duration_seconds": 19.8,
                    "total_records": 412,
                    "indexed": 412,
                    "sources": [
                        {
                            "source": "codeforces",
                            "fetched": 398,
                            "solved": 301,
                            "attempted_unsolved": 97,
                            "ok": True,
                            "detail": "ok",
                        },
                        {
                            "source": "leetcode",
                            "fetched": 14,
                            "solved": 14,
                            "attempted_unsolved": 0,
                            "ok": True,
                            "detail": "ok",
                        },
                    ],
                }
            }
        }
    )


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
class ProfileResponse(BaseModel):
    """Headline statistics across both platforms."""

    codeforces_handle: str
    leetcode_username: str
    current_rating: int | None = None
    max_rating: int | None = None
    rank: str | None = None

    total_problems_touched: int
    total_solved: int
    total_attempted_unsolved: int
    solved_by_source: dict[str, int]
    solved_by_difficulty: dict[str, int]
    distinct_topics: int
    max_rating_solved: int | None = None
    avg_rating_solved: float | None = None
    last_solved_at: datetime | None = None
    last_sync_at: datetime | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "codeforces_handle": "Jenitsu444",
                "leetcode_username": "Mihul",
                "current_rating": 1387,
                "max_rating": 1421,
                "rank": "pupil",
                "total_problems_touched": 412,
                "total_solved": 315,
                "total_attempted_unsolved": 97,
                "solved_by_source": {"codeforces": 301, "leetcode": 14},
                "solved_by_difficulty": {"Easy": 180, "Medium": 121, "Hard": 14},
                "distinct_topics": 27,
                "max_rating_solved": 1900,
                "avg_rating_solved": 1142.6,
                "last_solved_at": "2026-07-28T18:02:11Z",
                "last_sync_at": "2026-07-30T09:14:22Z",
            }
        }
    )


# ---------------------------------------------------------------------------
# Weak topics
# ---------------------------------------------------------------------------
class WeakTopicsResponse(BaseModel):
    """Ranked weaknesses with the arithmetic behind each score."""

    topics: list[WeakTopic]
    analysed_records: int = Field(description="Problem records the ranking was computed from.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysed_records": 412,
                "topics": [
                    {
                        "topic": "shortest paths",
                        "score": 0.7412,
                        "confidence": 0.4,
                        "solved": 2,
                        "attempted_unsolved": 6,
                        "days_since_last_solve": 214,
                        "max_rating": 1300,
                        "signals": {
                            "coverage": 0.8,
                            "staleness": 1.0,
                            "difficulty": 0.75,
                            "struggle": 0.75,
                        },
                        "reasons": [
                            "6 attempted but unsolved vs 2 solved (75% struggle rate, 19 failed submissions).",
                            "Last solved 214 days ago — practice has gone stale.",
                            "Only 2 solved, below the 10-problem target for a topic to count as practised.",
                        ],
                    }
                ],
            }
        }
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
class RecommendationsResponse(BaseModel):
    """Real, unsolved Codeforces problems chosen for the user's weak topics."""

    recommendations: list[Recommendation]
    based_on_topics: list[str] = Field(description="Weak topics that drove this list.")
    rating_band: str = Field(description="Difficulty window the results were drawn from.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "based_on_topics": ["shortest paths", "segment tree", "combinatorics"],
                "rating_band": "1287-1687",
                "recommendations": [
                    {
                        "score": 0.7891,
                        "matched_topics": ["shortest paths", "graphs"],
                        "reason": "Targets your weak topics shortest paths, graphs; rated 1500, inside your 1287-1687 practice band.",
                        "problem": {
                            "problem_id": "cf:20/C",
                            "contest_id": 20,
                            "index": "C",
                            "name": "Dijkstra?",
                            "url": "https://codeforces.com/problemset/problem/20/C",
                            "rating": 1500,
                            "tags": ["graphs", "shortest paths"],
                            "raw_tags": ["graphs", "shortest paths"],
                            "solved_count": 48213,
                        },
                    }
                ],
            }
        }
    )


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------
class SearchHit(BaseModel):
    """One semantically similar problem from the user's own history."""

    name: str
    url: str
    source: str
    tags: list[str]
    rating: int | None = None
    difficulty: str | None = None
    solved: bool
    similarity: float | None = Field(default=None, description="1.0 is an exact match.")


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "graph traversal problems I struggled with",
                "hits": [
                    {
                        "name": "Fake NP",
                        "url": "https://codeforces.com/problemset/problem/805/A",
                        "source": "codeforces",
                        "tags": ["graphs", "graph traversal"],
                        "rating": 1000,
                        "difficulty": "Easy",
                        "solved": False,
                        "similarity": 0.8123,
                    }
                ],
            }
        }
    )


# ---------------------------------------------------------------------------
# Ask (router entry point)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """A natural-language question for the tutor."""

    question: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=25, description="Max topics or problems to return.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "what should I practice today", "limit": 5},
                {"question": "why am I weak at dp", "limit": 3},
                {"question": "show me problems like the graph ones I failed", "limit": 5},
            ]
        }
    )


class AskResponse(BaseModel):
    """The tutor's answer, plus the structured data it was derived from."""

    question: str
    intent: Intent
    intent_method: str = Field(description="'llm' or 'keyword' — how the question was routed.")
    answer: str
    answer_method: str = Field(description="'llm' or 'template' — how the prose was produced.")
    trace: list[str] = Field(description="Graph nodes visited, in order.")

    weak_topics: list[WeakTopic] = []
    recommendations: list[Recommendation] = []
    search_hits: list[SearchHit] = []
    profile: dict = {}

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "what should I practice today",
                "intent": "recommend",
                "intent_method": "keyword",
                "answer": "Practice these next:\n1. Dijkstra? [1500] (shortest paths, graphs) — https://codeforces.com/problemset/problem/20/C",
                "answer_method": "template",
                "trace": ["classify", "recommend", "synthesize"],
                "weak_topics": [],
                "recommendations": [],
                "search_hits": [],
                "profile": {},
            }
        }
    )


class ErrorResponse(BaseModel):
    """Uniform error body for 4xx and 5xx responses."""

    detail: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "No profile data yet. Call POST /sync first."}}
    )
