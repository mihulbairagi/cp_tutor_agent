"""Domain models — the vocabulary the whole application shares.

These are deliberately *not* the same objects as the API schemas in
``app.schemas``. Keeping them separate means the shape of a Codeforces
submission can change without forcing a breaking change on HTTP consumers, and
the API can present a friendlier view than the internal one.

The central type is :class:`ProblemRecord`. It represents one problem the user
has *interacted with*, solved or not — a single type covering both cases,
because "attempted eleven times and never solved" is the strongest weakness
signal we have, and modelling failures as second-class citizens would make it
awkward to use.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["codeforces", "leetcode"]


class ProblemRecord(BaseModel):
    """One problem the user has attempted, on either platform."""

    source: Source
    problem_id: str = Field(
        description="Stable cross-platform id, e.g. 'cf:1352/A' or 'lc:two-sum'"
    )
    name: str
    url: str

    tags: list[str] = Field(default_factory=list, description="Canonical topics (see taxonomy).")
    raw_tags: list[str] = Field(default_factory=list, description="Tags exactly as upstream sent.")

    rating: int | None = Field(default=None, description="Codeforces rating; None for LeetCode.")
    difficulty: str | None = Field(default=None, description="Easy / Medium / Hard.")

    solved: bool = False
    attempt_count: int = Field(default=1, description="Submissions the user made on this problem.")
    solved_at: datetime | None = None
    last_attempt_at: datetime | None = None

    @property
    def failed_attempts(self) -> int:
        """Submissions that did not solve the problem."""
        return max(0, self.attempt_count - (1 if self.solved else 0))

    def to_document(self) -> str:
        """The text embedded into the vector store.

        Name plus topics plus difficulty: enough for a semantic query like
        "shortest path problems I found hard" to match, without diluting the
        embedding with ids and URLs that carry no meaning.
        """
        parts = [self.name]
        if self.tags:
            parts.append("Topics: " + ", ".join(self.tags))
        if self.difficulty:
            parts.append(f"Difficulty: {self.difficulty}")
        if self.rating:
            parts.append(f"Rating: {self.rating}")
        parts.append(f"Platform: {self.source}")
        return ". ".join(parts)


class CandidateProblem(BaseModel):
    """An unsolved Codeforces problem eligible to be recommended."""

    problem_id: str
    contest_id: int | None = None
    index: str = ""
    name: str
    url: str
    rating: int | None = None
    tags: list[str] = Field(default_factory=list)
    raw_tags: list[str] = Field(default_factory=list)
    solved_count: int = Field(
        default=0, description="How many users solved it; a popularity proxy."
    )


class Recommendation(BaseModel):
    """A candidate problem plus the justification for surfacing it."""

    problem: CandidateProblem
    score: float
    matched_topics: list[str] = Field(default_factory=list)
    reason: str


class TopicStat(BaseModel):
    """Aggregated activity for a single canonical topic."""

    topic: str
    solved: int = 0
    attempted_unsolved: int = 0
    total_failed_submissions: int = 0
    last_solved_at: datetime | None = None
    max_rating: int | None = None
    avg_rating: float | None = None
    sources: list[Source] = Field(default_factory=list)

    @property
    def total_touched(self) -> int:
        return self.solved + self.attempted_unsolved


class WeakTopic(BaseModel):
    """A ranked weakness, with the arithmetic behind it exposed."""

    topic: str
    score: float = Field(description="0-1, higher means weaker.")
    confidence: float = Field(description="0-1, based on how much data backs the score.")
    solved: int
    attempted_unsolved: int
    days_since_last_solve: int | None
    max_rating: int | None
    signals: dict[str, float] = Field(
        default_factory=dict, description="Per-signal contributions before weighting."
    )
    reasons: list[str] = Field(default_factory=list)


class UserRating(BaseModel):
    """Codeforces rating snapshot, used to size the recommendation band."""

    handle: str
    rating: int | None = None
    max_rating: int | None = None
    rank: str | None = None


class SourceSyncResult(BaseModel):
    """Outcome of ingesting one platform."""

    source: Source
    fetched: int = 0
    solved: int = 0
    attempted_unsolved: int = 0
    ok: bool = True
    detail: str = ""


class SyncResult(BaseModel):
    """Outcome of a full sync across every configured platform."""

    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    sources: list[SourceSyncResult] = Field(default_factory=list)
    total_records: int = 0
    indexed: int = 0

    @property
    def ok(self) -> bool:
        return all(source.ok for source in self.sources)
