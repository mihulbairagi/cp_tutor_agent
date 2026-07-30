"""Weak-topic analysis.

This module is the analytical core of the project, and it contains **no LLM
calls and no I/O**. That is a deliberate architectural decision, and the one
most worth defending in an interview.

Why not just ask the model "what am I weak at"?
-----------------------------------------------
Three reasons:

* **Reproducibility.** The same history must always yield the same ranking. An
  LLM asked to eyeball a thousand submissions will not do that, and a coaching
  tool that changes its mind between refreshes is not trustworthy.
* **Testability.** Because these are pure functions over a list of records, the
  test suite pins each signal independently. You cannot unit-test a vibe.
* **Explainability.** Every score decomposes into four named signals with
  human-readable reasons attached. When the tutor says "you are weak at
  shortest paths", it can show the arithmetic.

The LLM's job, later in the pipeline, is to *route* the question and *phrase*
the answer — never to compute the facts.

The four signals
----------------
=================  ==========================================================
``coverage``       How far below a healthy solve count this topic sits.
``staleness``      How long since the last solve — skills decay.
``difficulty``     How far the topic's hardest solve trails the user's overall
                   ceiling. Twenty easy DP solves is not DP competence.
``struggle``       Share of touched problems that were attempted and never
                   solved. The sharpest signal, and the one most trackers
                   discard by filtering to accepted submissions only.
=================  ==========================================================

Weights live in configuration so the ranking can be retuned without a code
change. The blended score is then damped by a **confidence** term derived from
sample size, so a topic with two data points cannot outrank one with thirty.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

from app.config import Settings, get_settings
from app.models import CandidateProblem, ProblemRecord, TopicStat, WeakTopic

logger = logging.getLogger(__name__)

# Rating span treated as "a full tier of headroom" when normalising the
# difficulty gap. ~800 Codeforces points is roughly the distance from a
# comfortable solve to a genuinely hard one.
RATING_SPAN = 800.0

# Fraction of the raw score retained when confidence is zero. Keeps blind spots
# visible without letting a single data point dominate the ranking.
_MIN_CONFIDENCE_RETENTION = 0.55


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_topic_stats(records: list[ProblemRecord]) -> dict[str, TopicStat]:
    """Aggregate per-problem records into per-topic statistics.

    A problem tagged with three topics contributes to all three. That
    intentionally over-counts totals, but weakness is measured *per topic*, and
    a graph problem genuinely is evidence about both ``graphs`` and ``dfs``.
    """
    stats: dict[str, TopicStat] = {}

    for record in records:
        for topic in record.tags:
            stat = stats.get(topic)
            if stat is None:
                stat = TopicStat(topic=topic)
                stats[topic] = stat

            if record.source not in stat.sources:
                stat.sources.append(record.source)

            if record.solved:
                stat.solved += 1
                if record.solved_at and (
                    stat.last_solved_at is None or record.solved_at > stat.last_solved_at
                ):
                    stat.last_solved_at = record.solved_at
                if record.rating and (stat.max_rating is None or record.rating > stat.max_rating):
                    stat.max_rating = record.rating
            else:
                stat.attempted_unsolved += 1
                stat.total_failed_submissions += record.attempt_count

    # Average rating over solved, rated problems only.
    rating_totals: dict[str, list[int]] = {}
    for record in records:
        if record.solved and record.rating:
            for topic in record.tags:
                rating_totals.setdefault(topic, []).append(record.rating)
    for topic, ratings in rating_totals.items():
        stats[topic].avg_rating = round(sum(ratings) / len(ratings), 1)

    return stats


def topic_frequency(candidates: list[CandidateProblem]) -> Counter[str]:
    """Count how often each topic appears across the Codeforces problemset.

    Used to decide which *untouched* topics are worth flagging as blind spots.
    Without this, an obscure tag applied to four problems would be ranked as
    urgent a weakness as never having written a DP.
    """
    counter: Counter[str] = Counter()
    for candidate in candidates:
        counter.update(candidate.tags)
    return counter


def _overall_ceiling(stats: dict[str, TopicStat]) -> int | None:
    ratings = [stat.max_rating for stat in stats.values() if stat.max_rating]
    return max(ratings) if ratings else None


def _score_one(
    stat: TopicStat,
    *,
    settings: Settings,
    now: datetime,
    ceiling: int | None,
) -> WeakTopic:
    target = max(1, settings.target_solves_per_topic)
    horizon = max(1, settings.staleness_horizon_days)

    # --- signal 1: coverage ------------------------------------------------
    coverage = _clamp((target - stat.solved) / target)

    # --- signal 2: staleness ----------------------------------------------
    days_since: int | None = None
    if stat.last_solved_at is None:
        staleness = 1.0
    else:
        last = stat.last_solved_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        days_since = max(0, (now - last).days)
        staleness = _clamp(days_since / horizon)

    # --- signal 3: difficulty gap -----------------------------------------
    if ceiling is None or stat.max_rating is None:
        # No rated evidence in this topic. Treat as unproven rather than strong,
        # but confidence damping stops that from dominating the ranking.
        difficulty = 1.0 if stat.solved == 0 else 0.6
    else:
        difficulty = _clamp((ceiling - stat.max_rating) / RATING_SPAN)

    # --- signal 4: struggle rate ------------------------------------------
    touched = stat.total_touched
    struggle = _clamp(stat.attempted_unsolved / touched) if touched else 0.0

    weights = settings.weakness_weights
    weight_sum = sum(weights.values()) or 1.0
    signals = {
        "coverage": round(coverage, 4),
        "staleness": round(staleness, 4),
        "difficulty": round(difficulty, 4),
        "struggle": round(struggle, 4),
    }
    raw = (
        weights["coverage"] * coverage
        + weights["staleness"] * staleness
        + weights["difficulty"] * difficulty
        + weights["struggle"] * struggle
    ) / weight_sum

    confidence = _clamp(touched / target)
    damped = raw * (_MIN_CONFIDENCE_RETENTION + (1 - _MIN_CONFIDENCE_RETENTION) * confidence)

    return WeakTopic(
        topic=stat.topic,
        score=round(damped, 4),
        confidence=round(confidence, 4),
        solved=stat.solved,
        attempted_unsolved=stat.attempted_unsolved,
        days_since_last_solve=days_since,
        max_rating=stat.max_rating,
        signals=signals,
        reasons=_build_reasons(stat, signals, days_since, ceiling, settings),
    )


def _build_reasons(
    stat: TopicStat,
    signals: dict[str, float],
    days_since: int | None,
    ceiling: int | None,
    settings: Settings,
) -> list[str]:
    """Turn the numeric signals into sentences a human can act on.

    Ordered by signal strength so the most important explanation is read first.
    """
    reasons: list[tuple[float, str]] = []

    if stat.total_touched == 0:
        return ["No attempts recorded on this topic — a genuine blind spot."]

    if signals["struggle"] > 0:
        rate = round(signals["struggle"] * 100)
        reasons.append(
            (
                signals["struggle"],
                f"{stat.attempted_unsolved} attempted but unsolved vs {stat.solved} solved "
                f"({rate}% struggle rate, {stat.total_failed_submissions} failed submissions).",
            )
        )

    if stat.solved < settings.target_solves_per_topic:
        reasons.append(
            (
                signals["coverage"],
                f"Only {stat.solved} solved, below the {settings.target_solves_per_topic}-problem "
                "target for a topic to count as practised.",
            )
        )

    if days_since is None:
        reasons.append((signals["staleness"], "Never solved a problem on this topic."))
    elif days_since > settings.staleness_horizon_days // 2:
        reasons.append(
            (signals["staleness"], f"Last solved {days_since} days ago — practice has gone stale.")
        )

    if ceiling and stat.max_rating and stat.max_rating < ceiling:
        reasons.append(
            (
                signals["difficulty"],
                f"Hardest solve here is rated {stat.max_rating}, "
                f"{ceiling - stat.max_rating} points below your overall best of {ceiling}.",
            )
        )

    if not reasons:
        return ["Solid coverage — no significant weakness detected."]

    reasons.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in reasons]


def score_topics(
    stats: dict[str, TopicStat],
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
    problemset_frequency: Counter[str] | None = None,
) -> list[WeakTopic]:
    """Rank topics from weakest to strongest.

    ``problemset_frequency`` opts in blind-spot detection: common Codeforces
    topics the user has never touched are injected with zero stats so they can
    be ranked. Rare tags are filtered out via ``min_topic_frequency``.
    """
    settings = settings or get_settings()
    now = now or datetime.now(UTC)
    ceiling = _overall_ceiling(stats)

    working = dict(stats)
    if problemset_frequency:
        for topic, frequency in problemset_frequency.items():
            if topic in working or frequency < settings.min_topic_frequency:
                continue
            working[topic] = TopicStat(topic=topic)

    scored = [_score_one(stat, settings=settings, now=now, ceiling=ceiling) for stat in working.values()]
    # Ties broken by fewest solves, so the emptier topic surfaces first.
    scored.sort(key=lambda topic: (-topic.score, topic.solved, topic.topic))
    return scored


def analyse_weak_topics(
    records: list[ProblemRecord],
    *,
    settings: Settings | None = None,
    limit: int = 5,
    now: datetime | None = None,
    candidates: list[CandidateProblem] | None = None,
) -> list[WeakTopic]:
    """Convenience pipeline: records in, ranked weaknesses out."""
    settings = settings or get_settings()
    stats = build_topic_stats(records)
    frequency = topic_frequency(candidates) if candidates else None
    ranked = score_topics(stats, settings=settings, now=now, problemset_frequency=frequency)
    logger.info(
        "weak-topic analysis complete",
        extra={"topics_scored": len(ranked), "records": len(records)},
    )
    return ranked[:limit] if limit else ranked


def summarise_profile(records: list[ProblemRecord]) -> dict:
    """Headline numbers for the ``/profile`` endpoint."""
    solved = [record for record in records if record.solved]
    rated = [record.rating for record in solved if record.rating]
    by_source: Counter[str] = Counter(record.source for record in solved)
    by_difficulty: Counter[str] = Counter(
        record.difficulty for record in solved if record.difficulty
    )
    last_solved = max((r.solved_at for r in solved if r.solved_at), default=None)

    return {
        "total_problems_touched": len(records),
        "total_solved": len(solved),
        "total_attempted_unsolved": len(records) - len(solved),
        "solved_by_source": dict(by_source),
        "solved_by_difficulty": dict(by_difficulty),
        "distinct_topics": len({topic for record in solved for topic in record.tags}),
        "max_rating_solved": max(rated) if rated else None,
        "avg_rating_solved": round(sum(rated) / len(rated), 1) if rated else None,
        "last_solved_at": last_solved,
    }
