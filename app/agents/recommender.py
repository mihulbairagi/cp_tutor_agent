"""Recommender agent.

Chains the weak-topic ranking into :mod:`app.services.recommender` and returns
real, unsolved Codeforces problems. Like the weak-topic agent it holds no
scoring logic of its own — it gathers inputs, calls pure functions, and renders.

One piece of light natural-language handling lives here: if the question names
topics explicitly ("give me some graph and DP problems"), those topics override
the automatic weakness ranking. It is a substring scan over the canonical
taxonomy rather than an LLM call — deterministic, instant, and free.
"""

from __future__ import annotations

import logging

from app.clients.codeforces import CodeforcesClient
from app.config import Settings, get_settings
from app.models import Recommendation, WeakTopic
from app.services.ingestion import IngestionService, get_ingestion_service
from app.services.recommender import rank_candidates, solved_problem_ids
from app.services.taxonomy import CANONICAL_TOPICS, normalize_tag

logger = logging.getLogger(__name__)

# Words that name a topic in casual phrasing but are not the canonical spelling.
_SPOKEN_TOPIC_ALIASES = {
    "dp": "dynamic programming",
    "dfs": "graph traversal",
    "bfs": "graph traversal",
    "dsu": "union find",
    "bitmask": "bit manipulation",
    "bitmasks": "bit manipulation",
    "graph": "graphs",
    "string": "strings",
    "tree": "trees",
    "binary search": "binary search",
    "number theory": "number theory",
    "segment tree": "segment tree",
    "two pointer": "two pointers",
    "sliding window": "sliding window",
}


def extract_topics(question: str) -> list[str]:
    """Find canonical topics named in a question.

    Longest-first matching stops "binary search" from being detected as the
    unrelated topic "search" would-be substring of another tag.
    """
    text = (question or "").lower()
    found: list[str] = []

    vocabulary = sorted({*CANONICAL_TOPICS, *_SPOKEN_TOPIC_ALIASES}, key=len, reverse=True)
    for phrase in vocabulary:
        if phrase in text:
            canonical = _SPOKEN_TOPIC_ALIASES.get(phrase) or normalize_tag(phrase)
            if canonical and canonical not in found:
                found.append(canonical)
    return found


class RecommenderAgent:
    """Answers 'what should I practise next'."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        ingestion: IngestionService | None = None,
        codeforces: CodeforcesClient | None = None,
        weak_topic_agent=None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ingestion = ingestion or get_ingestion_service()
        self._codeforces = codeforces or CodeforcesClient(settings=self._settings)

        if weak_topic_agent is None:
            from app.agents.weak_topics import WeakTopicAgent

            weak_topic_agent = WeakTopicAgent(
                settings=self._settings, ingestion=self._ingestion, codeforces=self._codeforces
            )
        self._weak_topics = weak_topic_agent

    def run(
        self,
        *,
        limit: int = 10,
        topics: list[str] | None = None,
        rating_override: int | None = None,
    ) -> tuple[list[Recommendation], list[WeakTopic]]:
        """Return recommendations plus the weaknesses that motivated them."""
        records = self._ingestion.get_records()
        if not records:
            logger.warning("recommendations requested with no synced data")
            return [], []

        # Look at more weak topics than we return problems, so the diversity cap
        # has alternatives to draw from.
        weak = self._weak_topics.run(limit=max(8, limit))

        if topics:
            requested = {normalize_tag(topic) for topic in topics}
            requested.discard(None)
            filtered = [topic for topic in weak if topic.topic in requested]
            if filtered:
                weak = filtered
            else:
                # The user asked for a topic they are not measurably weak at.
                # Honour the request anyway, at neutral weakness.
                weak = [
                    WeakTopic(
                        topic=topic,
                        score=0.5,
                        confidence=0.0,
                        solved=0,
                        attempted_unsolved=0,
                        days_since_last_solve=None,
                        max_rating=None,
                        reasons=["Explicitly requested."],
                    )
                    for topic in requested
                    if topic
                ]

        try:
            candidates = self._codeforces.fetch_problemset()
        except Exception:  # noqa: BLE001 - surfaced to the caller as an empty list
            logger.exception("could not fetch the Codeforces problemset")
            return [], weak

        rating = rating_override or self._ingestion.get_user_rating().rating

        recommendations = rank_candidates(
            candidates,
            weak_topics=weak,
            solved_ids=solved_problem_ids(records),
            user_rating=rating,
            settings=self._settings,
            limit=limit,
        )
        return recommendations, weak

    @staticmethod
    def template_answer(recommendations: list[Recommendation]) -> str:
        if not recommendations:
            return (
                "I could not build a practice list. Run POST /sync first, and check that "
                "your Codeforces handle is set correctly in .env."
            )

        lines = ["Practice these next:"]
        for position, rec in enumerate(recommendations, start=1):
            problem = rec.problem
            lines.append(
                f"{position}. {problem.name} [{problem.rating}] "
                f"({', '.join(rec.matched_topics)}) — {problem.url}"
            )
        return "\n".join(lines)
