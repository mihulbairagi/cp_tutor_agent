"""Problem recommender.

Turns a weakness ranking into a concrete practice list of **real, unsolved
Codeforces problems**, drawn from the live ``problemset.problems`` archive.

The pipeline is a filter chain followed by a ranking, both deterministic:

1. **Exclude what is already solved.** Recommending a solved problem is the one
   unforgivable bug in a tool like this, so it is the first filter and it has a
   dedicated test.
2. **Keep problems that touch a weak topic.** No weak topic match, no
   recommendation — otherwise this is just a random problem generator.
3. **Keep problems inside a rating band** around the user's current rating
   (default −100 to +300). Practice should be uncomfortable, not impossible:
   below the band it teaches nothing, far above it the user stalls and loses
   the habit. The asymmetry is intentional — the band leans upward.
4. **Rank** by weakness of the matched topics, how well the rating fits, and a
   small popularity term.
5. **Enforce topic diversity**, so a session is a balanced set rather than ten
   near-identical DP problems.

Note that problems the user *attempted and failed* are deliberately left in the
candidate pool. Those are prime practice material, and they score well because
they are tagged with topics the analyser just flagged.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from app.config import Settings, get_settings
from app.models import CandidateProblem, ProblemRecord, Recommendation, WeakTopic

logger = logging.getLogger(__name__)

# Ranking mix. Topic relevance dominates: the point of the tool is to attack
# weaknesses, with rating fit as a comfort filter and popularity only as a
# tie-breaker that favours problems with editorials and discussion.
WEIGHT_TOPIC = 0.60
WEIGHT_RATING_FIT = 0.30
WEIGHT_POPULARITY = 0.10

# Solve counts are heavily skewed, so popularity is compared on a log scale.
# ~e^12 ≈ 160k solves is about the ceiling for a famous Div. 2 A.
_POPULARITY_LOG_CEILING = 12.0


def solved_problem_ids(records: list[ProblemRecord]) -> set[str]:
    """Ids of problems the user has already solved."""
    return {record.problem_id for record in records if record.solved}


def _rating_fit(rating: int, target: int, low: int, high: int) -> float:
    """1.0 at the target rating, decaying to 0.0 at the edges of the band."""
    span = max(target - low, high - target, 1)
    return max(0.0, 1.0 - abs(rating - target) / span)


def _popularity(solved_count: int) -> float:
    if solved_count <= 0:
        return 0.0
    return min(1.0, math.log(solved_count + 1) / _POPULARITY_LOG_CEILING)


def rank_candidates(
    candidates: list[CandidateProblem],
    *,
    weak_topics: list[WeakTopic],
    solved_ids: set[str],
    user_rating: int | None = None,
    settings: Settings | None = None,
    limit: int = 10,
) -> list[Recommendation]:
    """Filter and rank the problemset into a practice list."""
    settings = settings or get_settings()

    if not weak_topics:
        logger.info("no weak topics supplied; nothing to recommend")
        return []

    weakness = {topic.topic: topic.score for topic in weak_topics}

    rating = user_rating or settings.default_rating_when_unrated
    low = rating - settings.rating_band_below
    high = rating + settings.rating_band_above
    # Aim slightly above current rating: that is where learning happens.
    target = min(high, rating + settings.rating_band_below)

    scored: list[Recommendation] = []
    for candidate in candidates:
        if candidate.problem_id in solved_ids:
            continue
        # Unrated problems are excluded: without a rating we cannot honour the
        # difficulty band, and recommending blind defeats the purpose.
        if candidate.rating is None or not (low <= candidate.rating <= high):
            continue

        matched = [tag for tag in candidate.tags if tag in weakness]
        if not matched:
            continue

        # Best matched topic drives relevance; extra matches add a little, so a
        # problem hitting two weak topics beats one hitting a single topic
        # without letting tag-heavy problems dominate outright.
        best = max(weakness[tag] for tag in matched)
        breadth = min(1.0, (len(matched) - 1) * 0.15)
        topic_score = min(1.0, best * (1 + breadth))

        score = (
            WEIGHT_TOPIC * topic_score
            + WEIGHT_RATING_FIT * _rating_fit(candidate.rating, target, low, high)
            + WEIGHT_POPULARITY * _popularity(candidate.solved_count)
        )

        scored.append(
            Recommendation(
                problem=candidate,
                score=round(score, 4),
                matched_topics=matched,
                reason=(
                    f"Targets your weak topic{'s' if len(matched) > 1 else ''} "
                    f"{', '.join(matched)}; rated {candidate.rating}, "
                    f"inside your {low}-{high} practice band."
                ),
            )
        )

    scored.sort(key=lambda rec: (-rec.score, rec.problem.problem_id))
    diversified = _enforce_diversity(scored, settings.max_problems_per_topic, limit)

    logger.info(
        "recommendations ready",
        extra={
            "candidates": len(candidates),
            "eligible": len(scored),
            "returned": len(diversified),
            "band": f"{low}-{high}",
        },
    )
    return diversified


def _enforce_diversity(
    ranked: list[Recommendation], per_topic_cap: int, limit: int
) -> list[Recommendation]:
    """Cap how many problems share a topic, so a session stays varied.

    If the cap leaves fewer than ``limit`` results (a narrow profile with only
    one or two weak topics), the best remaining problems backfill the list —
    returning three problems when ten were asked for would be worse than a
    slightly repetitive set.
    """
    chosen: list[Recommendation] = []
    used: Counter[str] = Counter()

    for recommendation in ranked:
        if len(chosen) >= limit:
            break
        if all(used[topic] >= per_topic_cap for topic in recommendation.matched_topics):
            continue
        chosen.append(recommendation)
        for topic in recommendation.matched_topics:
            used[topic] += 1

    if len(chosen) < limit:
        already = {rec.problem.problem_id for rec in chosen}
        for recommendation in ranked:
            if len(chosen) >= limit:
                break
            if recommendation.problem.problem_id not in already:
                chosen.append(recommendation)

    return chosen[:limit]
