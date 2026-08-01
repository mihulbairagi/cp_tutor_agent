"""Weak-topic agent.

A thin orchestration layer over :mod:`app.services.analysis`. The agent's job is
to gather inputs (the synced profile, and the problemset needed for blind-spot
detection) and hand them to the deterministic scorer — not to reason about them
itself.

The template answer here is a first-class code path, not a placeholder: with no
API key configured it is what the user sees, and it is what the evaluation
harness asserts against.
"""

from __future__ import annotations

import logging

from app.clients.codeforces import CodeforcesClient
from app.config import Settings, get_settings
from app.models import WeakTopic
from app.services.analysis import analyse_weak_topics, summarise_profile
from app.services.ingestion import IngestionService, get_ingestion_service

logger = logging.getLogger(__name__)


class WeakTopicAgent:
    """Answers 'what am I weak at' and 'why am I weak at X'."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        ingestion: IngestionService | None = None,
        codeforces: CodeforcesClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ingestion = ingestion or get_ingestion_service()
        self._codeforces = codeforces or CodeforcesClient(settings=self._settings)

    def run(
        self,
        *,
        limit: int = 5,
        include_blind_spots: bool = True,
        topic_filter: str | None = None,
    ) -> list[WeakTopic]:
        """Rank the user's weakest topics.

        ``topic_filter`` supports "why am I weak at DP" — the same ranking is
        computed, then narrowed to the topic asked about, so the score the user
        sees is the score it holds in the full ranking rather than a differently
        scaled one-off number.
        """
        records = self._ingestion.get_records()
        if not records:
            logger.warning("weak-topic analysis requested with no synced data")
            return []

        candidates = None
        if include_blind_spots:
            try:
                candidates = self._codeforces.fetch_problemset()
            except Exception:  # noqa: BLE001 - blind spots are a bonus, not required
                logger.warning("problemset unavailable; skipping blind-spot detection")

        # Score everything, then slice — filtering before scoring would change
        # the overall rating ceiling and distort the difficulty signal.
        ranked = analyse_weak_topics(
            records, settings=self._settings, limit=0, candidates=candidates
        )

        if topic_filter:
            needle = topic_filter.strip().lower()
            matched = [topic for topic in ranked if needle in topic.topic.lower()]
            if matched:
                return matched[:limit]
            logger.info("no topic matched filter", extra={"filter": topic_filter})
            return []

        return ranked[:limit]

    def profile(self) -> dict:
        return summarise_profile(self._ingestion.get_records())

    @staticmethod
    def template_answer(weak_topics: list[WeakTopic], topic_filter: str | None = None) -> str:
        """Render a readable answer without an LLM."""
        if not weak_topics:
            if topic_filter:
                return (
                    f"I have no recorded activity for '{topic_filter}'. "
                    "Either it is not a topic I track, or you have not attempted it yet — "
                    "which would itself make it a blind spot worth starting on."
                )
            return "No analysis available yet. Run POST /sync to pull your submission history."

        lines: list[str] = []
        if topic_filter and len(weak_topics) == 1:
            topic = weak_topics[0]
            lines.append(
                f"You rank {topic.topic} at a weakness score of {topic.score:.2f} "
                f"(confidence {topic.confidence:.2f}). Here is why:"
            )
            lines.extend(f"  - {reason}" for reason in topic.reasons)
            return "\n".join(lines)

        lines.append("Your weakest topics right now:")
        for position, topic in enumerate(weak_topics, start=1):
            headline = topic.reasons[0] if topic.reasons else "Limited practice recorded."
            lines.append(
                f"{position}. {topic.topic} — score {topic.score:.2f}, "
                f"{topic.solved} solved / {topic.attempted_unsolved} unsolved. {headline}"
            )
        return "\n".join(lines)
