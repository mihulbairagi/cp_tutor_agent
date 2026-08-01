"""A fixed synthetic profile, plus stand-ins for every external service.

The harness must be **reproducible**: running it twice, on any machine, with or
without network access, has to give the same table. So the user's history, the
Codeforces problemset and the vector index are all fixtures here.

Timestamps are expressed as offsets from the moment the harness starts rather
than as absolute dates. A hard-coded date would slowly rot — every topic would
drift towards "stale" as the calendar advanced, and the staleness signal would
stop discriminating. Relative offsets keep the intended shape of the profile
fixed forever.

Everything above these stubs — the graph, the intent classifier, the weakness
scorer, the ranker, the templates — is the real code.

The profile is shaped so each scoring signal has an unambiguous owner:

===================  ==========================================================
``greedy``           well practised, recent, high rated. The control: it must
                     never appear as a weakness.
``shortest paths``   2 solves against 6 unsolved attempts, last solve 240 days
                     ago. The headline weakness.
``dynamic programming``  10 solves, all rated 900. Pure difficulty gap.
``geometry``         one solve. Pure coverage gap.
===================  ==========================================================
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.models import CandidateProblem, ProblemRecord, UserRating

#: Anchor for every timestamp below. Read once so a single run is self-consistent.
NOW = datetime.now(UTC)

USER_RATING = 1400


def _days_ago(days: int) -> datetime:
    return NOW - timedelta(days=days)


def build_records() -> list[ProblemRecord]:
    records: list[ProblemRecord] = []

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
                last_attempt_at=_days_ago(30),
            )
        )

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

    records.append(
        ProblemRecord(
            source="leetcode",
            problem_id="lc:max-points-on-a-line",
            name="Max Points on a Line",
            url="https://leetcode.com/problems/max-points-on-a-line/",
            tags=["geometry", "math"],
            raw_tags=["Geometry", "Math"],
            difficulty="Hard",
            solved=True,
            attempt_count=1,
            solved_at=_days_ago(3),
            last_attempt_at=_days_ago(3),
        )
    )
    return records


def build_problemset() -> list[CandidateProblem]:
    """A miniature Codeforces problemset spanning several topics and ratings.

    Includes ``cf:100/0``, which the user has already solved, so the "never
    recommend a solved problem" invariant is actually exercised rather than
    trivially satisfied.
    """
    specs = [
        ("cf:100/0", "Greedy Problem 0", 1300, ["greedy"], 5000),
        ("cf:900/A", "Dijkstra Practice", 1400, ["shortest paths", "graphs"], 12000),
        ("cf:900/B", "Another Shortest Path", 1450, ["shortest paths"], 8000),
        ("cf:900/C", "Third Shortest Path", 1350, ["shortest paths"], 7000),
        ("cf:901/A", "Hard DP", 1500, ["dynamic programming"], 3000),
        ("cf:901/B", "Harder DP", 1600, ["dynamic programming"], 2200),
        ("cf:902/A", "Way Too Hard", 2900, ["shortest paths"], 400),
        ("cf:903/A", "Way Too Easy", 800, ["shortest paths"], 90000),
        ("cf:904/A", "Unrated Mystery", None, ["shortest paths"], 100),
        ("cf:905/A", "Segment Tree Basics", 1500, ["segment tree", "data structures"], 4000),
        ("cf:905/B", "Range Sum Queries", 1450, ["segment tree"], 5200),
        ("cf:906/A", "Counting Paths", 1550, ["combinatorics", "math"], 3300),
        ("cf:907/A", "Circle Intersections", 1400, ["geometry"], 1800),
        ("cf:908/A", "Union Find Drill", 1350, ["union find", "graphs"], 6100),
        ("cf:909/A", "Binary Search The Answer", 1500, ["binary search"], 9000),
        ("cf:910/A", "Graph Traversal Warmup", 1400, ["graph traversal", "graphs"], 15000),
    ]
    return [
        CandidateProblem(
            problem_id=problem_id,
            contest_id=int(problem_id.split(":")[1].split("/")[0]),
            index=problem_id.split("/")[1],
            name=name,
            url=f"https://codeforces.com/problemset/problem/{problem_id.split(':')[1]}",
            rating=rating,
            tags=tags,
            raw_tags=tags,
            solved_count=solved_count,
        )
        for problem_id, name, rating, tags, solved_count in specs
    ]


class StubIngestion:
    """The synced profile, without any HTTP or SQLite behind it."""

    def __init__(self) -> None:
        self._records = build_records()

    def has_data(self) -> bool:
        return True

    def get_records(self) -> list[ProblemRecord]:
        return self._records

    def get_user_rating(self) -> UserRating:
        return UserRating(handle="EvalUser", rating=USER_RATING, max_rating=1450, rank="pupil")

    def get_last_sync(self) -> datetime:
        return NOW


class StubCodeforces:
    """Serves the fixture problemset instead of calling the real API."""

    def __init__(self) -> None:
        self._problemset = build_problemset()

    def fetch_problemset(self, force_refresh: bool = False) -> list[CandidateProblem]:
        return self._problemset


class StubVectorStore:
    """Lexical stand-in for Chroma.

    Real embeddings would make the harness non-deterministic across
    sentence-transformers versions and slow to start, so search is scored here by
    token overlap against the same documents the real store would embed. The
    *interface* is identical, so the search node is genuinely exercised.
    """

    def __init__(self) -> None:
        self._records = build_records()

    def count(self) -> int:
        return len(self._records)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        source: str | None = None,
        solved: bool | None = None,
    ) -> list[dict]:
        terms = {token for token in query.lower().split() if len(token) > 2}
        scored: list[tuple[float, ProblemRecord]] = []

        for record in self._records:
            if source and record.source != source:
                continue
            if solved is not None and record.solved != solved:
                continue
            document = record.to_document().lower()
            overlap = sum(1 for term in terms if term in document)
            if overlap:
                scored.append((overlap / max(1, len(terms)), record))

        # Sort by similarity, then problem_id so ties never reorder between runs.
        scored.sort(key=lambda pair: (-pair[0], pair[1].problem_id))
        return [
            {
                "name": record.name,
                "url": record.url,
                "source": record.source,
                "tags": record.tags,
                "rating": record.rating,
                "difficulty": record.difficulty,
                "solved": record.solved,
                "similarity": round(similarity, 4),
            }
            for similarity, record in scored[:limit]
        ]


def solved_ids() -> set[str]:
    return {record.problem_id for record in build_records() if record.solved}
