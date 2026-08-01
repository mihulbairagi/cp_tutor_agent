"""Shared pytest fixtures.

The environment is configured **before** any ``app`` module is imported, because
``get_settings()`` is an ``lru_cache`` singleton — importing first and patching
after would leave the cached instance pointing at the developer's real ``.env``.

Every test in this suite runs with:

* an in-memory cache (no SQLite files left behind)
* no Anthropic key, so the deterministic router and templates are exercised
* the scheduler disabled, so a TestClient does not spawn background threads

and with no network access whatsoever — upstream payloads are recorded fixtures.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

# --- must precede any app import -------------------------------------------
os.environ.update(
    {
        "CODEFORCES_HANDLE": "TestUser",
        "LEETCODE_USERNAME": "testuser",
        "LEETCODE_SESSION": "",
        "ANTHROPIC_API_KEY": "",
        "CACHE_BACKEND": "memory",
        "SYNC_ENABLED": "false",
        "SYNC_ON_STARTUP": "false",
        "LOG_LEVEL": "WARNING",
        "HTTP_MAX_RETRIES": "2",
        "HTTP_BACKOFF_BASE_SECONDS": "0.01",
    }
)

import pytest  # noqa: E402

from app.cache import MemoryCache, reset_cache  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.models import CandidateProblem, ProblemRecord  # noqa: E402

#: Fixed clock so staleness assertions never drift with the calendar.
NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset cached globals between tests so state cannot leak."""
    get_settings.cache_clear()
    reset_cache()
    yield
    get_settings.cache_clear()
    reset_cache()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def cache() -> MemoryCache:
    return MemoryCache()


def _days_ago(days: int) -> datetime:
    return NOW - timedelta(days=days)


@pytest.fixture
def sample_records() -> list[ProblemRecord]:
    """A hand-built profile with one clearly-defined weakness per topic.

    Shaped so each scoring signal can be asserted in isolation:

    * ``greedy``          — well practised, recent, and hard. The control.
    * ``shortest paths``  — high struggle rate, stale, low ceiling. The weakness.
    * ``dynamic programming`` — solved often but only easy problems (difficulty gap).
    * ``geometry``        — barely touched (coverage gap).
    """
    records: list[ProblemRecord] = []

    # greedy: 12 solves, recent, up to 1800
    for index in range(12):
        records.append(
            ProblemRecord(
                source="codeforces",
                problem_id=f"cf:100/{index}",
                name=f"Greedy Problem {index}",
                url=f"https://codeforces.com/problemset/problem/100/{index}",
                tags=["greedy"],
                raw_tags=["greedy"],
                rating=1200 + index * 50,
                difficulty="Medium",
                solved=True,
                attempt_count=1,
                solved_at=_days_ago(5 + index),
                last_attempt_at=_days_ago(5 + index),
            )
        )

    # shortest paths: 2 solves (old, low rated), 6 failures
    for index in range(2):
        records.append(
            ProblemRecord(
                source="codeforces",
                problem_id=f"cf:200/{index}",
                name=f"Shortest Path Solved {index}",
                url=f"https://codeforces.com/problemset/problem/200/{index}",
                tags=["shortest paths", "graphs"],
                raw_tags=["shortest paths", "graphs"],
                rating=1100,
                difficulty="Easy",
                solved=True,
                attempt_count=2,
                solved_at=_days_ago(240),
                last_attempt_at=_days_ago(240),
            )
        )
    for index in range(6):
        records.append(
            ProblemRecord(
                source="codeforces",
                problem_id=f"cf:201/{index}",
                name=f"Shortest Path Failed {index}",
                url=f"https://codeforces.com/problemset/problem/201/{index}",
                tags=["shortest paths", "graphs"],
                raw_tags=["shortest paths", "graphs"],
                rating=1500,
                difficulty="Medium",
                solved=False,
                attempt_count=3,
                solved_at=None,
                last_attempt_at=_days_ago(30),
            )
        )

    # dynamic programming: 10 solves but all rated 900 (difficulty gap)
    for index in range(10):
        records.append(
            ProblemRecord(
                source="codeforces",
                problem_id=f"cf:300/{index}",
                name=f"Easy DP {index}",
                url=f"https://codeforces.com/problemset/problem/300/{index}",
                tags=["dynamic programming"],
                raw_tags=["dp"],
                rating=900,
                difficulty="Easy",
                solved=True,
                attempt_count=1,
                solved_at=_days_ago(10 + index),
                last_attempt_at=_days_ago(10 + index),
            )
        )

    # geometry: a single recent solve (coverage gap)
    records.append(
        ProblemRecord(
            source="leetcode",
            problem_id="lc:max-points-on-a-line",
            name="Max Points on a Line",
            url="https://leetcode.com/problems/max-points-on-a-line/",
            tags=["geometry", "math"],
            raw_tags=["Geometry", "Math"],
            rating=None,
            difficulty="Hard",
            solved=True,
            attempt_count=1,
            solved_at=_days_ago(3),
            last_attempt_at=_days_ago(3),
        )
    )

    return records


@pytest.fixture
def sample_candidates() -> list[CandidateProblem]:
    """A miniature problemset spanning several ratings and topics."""
    return [
        CandidateProblem(
            problem_id="cf:100/0",  # deliberately already solved in sample_records
            contest_id=100,
            index="0",
            name="Greedy Problem 0",
            url="https://codeforces.com/problemset/problem/100/0",
            rating=1300,
            tags=["greedy"],
            raw_tags=["greedy"],
            solved_count=5000,
        ),
        CandidateProblem(
            problem_id="cf:900/A",
            contest_id=900,
            index="A",
            name="Dijkstra Practice",
            url="https://codeforces.com/problemset/problem/900/A",
            rating=1400,
            tags=["shortest paths", "graphs"],
            raw_tags=["shortest paths", "graphs"],
            solved_count=12000,
        ),
        CandidateProblem(
            problem_id="cf:900/B",
            contest_id=900,
            index="B",
            name="Another Shortest Path",
            url="https://codeforces.com/problemset/problem/900/B",
            rating=1450,
            tags=["shortest paths"],
            raw_tags=["shortest paths"],
            solved_count=8000,
        ),
        CandidateProblem(
            problem_id="cf:900/C",
            contest_id=900,
            index="C",
            name="Third Shortest Path",
            url="https://codeforces.com/problemset/problem/900/C",
            rating=1350,
            tags=["shortest paths"],
            raw_tags=["shortest paths"],
            solved_count=7000,
        ),
        CandidateProblem(
            problem_id="cf:901/A",
            contest_id=901,
            index="A",
            name="Hard DP",
            url="https://codeforces.com/problemset/problem/901/A",
            rating=1500,
            tags=["dynamic programming"],
            raw_tags=["dp"],
            solved_count=3000,
        ),
        CandidateProblem(
            problem_id="cf:902/A",
            contest_id=902,
            index="A",
            name="Way Too Hard",
            url="https://codeforces.com/problemset/problem/902/A",
            rating=2900,
            tags=["shortest paths"],
            raw_tags=["shortest paths"],
            solved_count=400,
        ),
        CandidateProblem(
            problem_id="cf:903/A",
            contest_id=903,
            index="A",
            name="Way Too Easy",
            url="https://codeforces.com/problemset/problem/903/A",
            rating=800,
            tags=["shortest paths"],
            raw_tags=["shortest paths"],
            solved_count=90000,
        ),
        CandidateProblem(
            problem_id="cf:904/A",
            contest_id=904,
            index="A",
            name="Unrated Mystery",
            url="https://codeforces.com/problemset/problem/904/A",
            rating=None,
            tags=["shortest paths"],
            raw_tags=["shortest paths"],
            solved_count=100,
        ),
    ]


@pytest.fixture
def codeforces_status_payload() -> dict:
    """A trimmed but structurally faithful ``user.status`` response."""
    return {
        "status": "OK",
        "result": [
            {
                "id": 3,
                "contestId": 4,
                "creationTimeSeconds": 1_700_000_300,
                "problem": {
                    "contestId": 4,
                    "index": "A",
                    "name": "Watermelon",
                    "rating": 800,
                    "tags": ["brute force", "math"],
                },
                "verdict": "OK",
            },
            {
                "id": 2,
                "contestId": 4,
                "creationTimeSeconds": 1_700_000_200,
                "problem": {
                    "contestId": 4,
                    "index": "A",
                    "name": "Watermelon",
                    "rating": 800,
                    "tags": ["brute force", "math"],
                },
                "verdict": "WRONG_ANSWER",
            },
            {
                "id": 1,
                "contestId": 20,
                "creationTimeSeconds": 1_700_000_100,
                "problem": {
                    "contestId": 20,
                    "index": "C",
                    "name": "Dijkstra?",
                    "rating": 1900,
                    "tags": ["graphs", "shortest paths"],
                },
                "verdict": "TIME_LIMIT_EXCEEDED",
            },
            {
                "id": 0,
                "contestId": 20,
                "creationTimeSeconds": 1_700_000_050,
                "problem": {
                    "contestId": 20,
                    "index": "C",
                    "name": "Dijkstra?",
                    "rating": 1900,
                    "tags": ["graphs", "shortest paths"],
                },
                "verdict": None,
            },
        ],
    }


@pytest.fixture
def codeforces_problemset_payload() -> dict:
    return {
        "status": "OK",
        "result": {
            "problems": [
                {
                    "contestId": 4,
                    "index": "A",
                    "name": "Watermelon",
                    "rating": 800,
                    "tags": ["brute force", "math"],
                },
                {
                    "contestId": 20,
                    "index": "C",
                    "name": "Dijkstra?",
                    "rating": 1900,
                    "tags": ["graphs", "shortest paths"],
                },
            ],
            "problemStatistics": [
                {"contestId": 4, "index": "A", "solvedCount": 250000},
                {"contestId": 20, "index": "C", "solvedCount": 48000},
            ],
        },
    }


@pytest.fixture
def leetcode_recent_payload() -> dict:
    return {
        "data": {
            "recentAcSubmissionList": [
                {"id": "1", "title": "Two Sum", "titleSlug": "two-sum", "timestamp": "1700000000"},
                {
                    "id": "2",
                    "title": "Course Schedule",
                    "titleSlug": "course-schedule",
                    "timestamp": "1700000500",
                },
                # duplicate slug: the client must not emit it twice
                {"id": "3", "title": "Two Sum", "titleSlug": "two-sum", "timestamp": "1700000900"},
            ]
        }
    }


@pytest.fixture
def leetcode_question_payload() -> dict:
    return {
        "data": {
            "question": {
                "questionFrontendId": "207",
                "title": "Course Schedule",
                "titleSlug": "course-schedule",
                "difficulty": "Medium",
                "topicTags": [
                    {"name": "Depth-First Search", "slug": "depth-first-search"},
                    {"name": "Graph", "slug": "graph"},
                    {"name": "Topological Sort", "slug": "topological-sort"},
                ],
            }
        }
    }


@pytest.fixture
def leetcode_profile_payload() -> dict:
    return {
        "data": {
            "matchedUser": {
                "username": "testuser",
                "profile": {"ranking": 123456, "reputation": 0},
                "submitStatsGlobal": {
                    "acSubmissionNum": [
                        {"difficulty": "All", "count": 140},
                        {"difficulty": "Easy", "count": 80},
                        {"difficulty": "Medium", "count": 50},
                        {"difficulty": "Hard", "count": 10},
                    ],
                    "totalSubmissionNum": [{"difficulty": "All", "count": 400}],
                },
            }
        }
    }


@pytest.fixture
def leetcode_solved_list_payload() -> dict:
    return {
        "data": {
            "problemsetQuestionList": {
                "total": 2,
                "questions": [
                    {
                        "questionFrontendId": "1",
                        "title": "Two Sum",
                        "titleSlug": "two-sum",
                        "difficulty": "Easy",
                        "status": "ac",
                        "topicTags": [
                            {"name": "Array", "slug": "array"},
                            {"name": "Hash Table", "slug": "hash-table"},
                        ],
                    },
                    {
                        "questionFrontendId": "4",
                        "title": "Median of Two Sorted Arrays",
                        "titleSlug": "median-of-two-sorted-arrays",
                        "difficulty": "Hard",
                        # not accepted: must be filtered out
                        "status": "notac",
                        "topicTags": [{"name": "Binary Search", "slug": "binary-search"}],
                    },
                ],
            }
        }
    }
