"""Tests for candidate filtering, ranking and diversity."""

from __future__ import annotations

from app.models import WeakTopic
from app.services.recommender import rank_candidates, solved_problem_ids


def _weak(topic: str, score: float) -> WeakTopic:
    return WeakTopic(
        topic=topic,
        score=score,
        confidence=0.8,
        solved=2,
        attempted_unsolved=6,
        days_since_last_solve=200,
        max_rating=1100,
    )


WEAK_TOPICS = [
    _weak("shortest paths", 0.77),
    _weak("graphs", 0.77),
    _weak("dynamic programming", 0.22),
]


def _rank(candidates, records, **kwargs):
    return rank_candidates(
        candidates,
        weak_topics=kwargs.pop("weak_topics", WEAK_TOPICS),
        solved_ids=solved_problem_ids(records),
        user_rating=kwargs.pop("user_rating", 1400),
        **kwargs,
    )


class TestFiltering:
    def test_an_already_solved_problem_is_never_recommended(
        self, sample_candidates, sample_records, settings
    ) -> None:
        """The one unforgivable bug in a tool like this."""
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        solved = solved_problem_ids(sample_records)
        assert all(rec.problem.problem_id not in solved for rec in results)
        assert "cf:100/0" not in {rec.problem.problem_id for rec in results}

    def test_problems_far_above_the_band_are_excluded(
        self, sample_candidates, sample_records, settings
    ) -> None:
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        assert "cf:902/A" not in {rec.problem.problem_id for rec in results}  # rated 2900

    def test_problems_far_below_the_band_are_excluded(
        self, sample_candidates, sample_records, settings
    ) -> None:
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        assert "cf:903/A" not in {rec.problem.problem_id for rec in results}  # rated 800

    def test_unrated_problems_are_excluded(
        self, sample_candidates, sample_records, settings
    ) -> None:
        """Without a rating the difficulty band cannot be honoured."""
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        assert "cf:904/A" not in {rec.problem.problem_id for rec in results}

    def test_every_result_matches_at_least_one_weak_topic(
        self, sample_candidates, sample_records, settings
    ) -> None:
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        assert results
        assert all(rec.matched_topics for rec in results)

    def test_no_weak_topics_means_no_recommendations(
        self, sample_candidates, sample_records, settings
    ) -> None:
        """A random problem generator would be worse than nothing."""
        assert _rank(sample_candidates, sample_records, settings=settings, weak_topics=[]) == []

    def test_the_band_leans_upward(self, sample_candidates, sample_records, settings) -> None:
        settings.rating_band_below = 100
        settings.rating_band_above = 300
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        ratings = [rec.problem.rating for rec in results]
        assert all(1300 <= rating <= 1700 for rating in ratings)


class TestRanking:
    def test_results_are_sorted_by_descending_score(
        self, sample_candidates, sample_records, settings
    ) -> None:
        results = _rank(sample_candidates, sample_records, settings=settings, limit=10)
        assert [rec.score for rec in results] == sorted(
            (rec.score for rec in results), reverse=True
        )

    def test_a_weaker_topic_outranks_a_stronger_one_at_equal_fit(
        self, sample_records, settings
    ) -> None:
        from app.models import CandidateProblem

        candidates = [
            CandidateProblem(
                problem_id="cf:1/A",
                name="Weak topic problem",
                url="u",
                rating=1500,
                tags=["shortest paths"],
                solved_count=1000,
            ),
            CandidateProblem(
                problem_id="cf:2/A",
                name="Strong topic problem",
                url="u",
                rating=1500,
                tags=["dynamic programming"],
                solved_count=1000,
            ),
        ]
        results = _rank(candidates, sample_records, settings=settings, limit=2)
        assert results[0].problem.problem_id == "cf:1/A"

    def test_every_recommendation_explains_itself(
        self, sample_candidates, sample_records, settings
    ) -> None:
        for rec in _rank(sample_candidates, sample_records, settings=settings, limit=10):
            assert rec.reason
            assert str(rec.problem.rating) in rec.reason

    def test_ranking_is_deterministic(self, sample_candidates, sample_records, settings) -> None:
        first = _rank(sample_candidates, sample_records, settings=settings, limit=5)
        second = _rank(sample_candidates, sample_records, settings=settings, limit=5)
        assert [r.problem.problem_id for r in first] == [r.problem.problem_id for r in second]


class TestDiversity:
    def test_a_single_topic_cannot_fill_the_whole_list(
        self, sample_candidates, sample_records, settings
    ) -> None:
        """Ten near-identical DP problems is a bad practice session."""
        settings.max_problems_per_topic = 2
        results = _rank(sample_candidates, sample_records, settings=settings, limit=3)
        shortest_path_count = sum(1 for rec in results if "shortest paths" in rec.matched_topics)
        assert shortest_path_count <= 2
        assert len(results) == 3

    def test_backfill_prefers_repeats_over_returning_too_few(
        self, sample_candidates, sample_records, settings
    ) -> None:
        """A slightly repetitive full list beats a short one."""
        settings.max_problems_per_topic = 1
        results = _rank(sample_candidates, sample_records, settings=settings, limit=4)
        assert len(results) == 4

    def test_limit_is_respected(self, sample_candidates, sample_records, settings) -> None:
        assert len(_rank(sample_candidates, sample_records, settings=settings, limit=1)) == 1


class TestSolvedIds:
    def test_only_solved_problems_are_collected(self, sample_records) -> None:
        solved = solved_problem_ids(sample_records)
        assert "cf:100/0" in solved
        assert "cf:201/0" not in solved  # attempted but never solved

    def test_failed_attempts_stay_recommendable(self, sample_records, settings) -> None:
        """Problems you got stuck on are prime practice material, not exclusions."""
        from app.models import CandidateProblem

        retry = CandidateProblem(
            problem_id="cf:201/0",  # attempted 3 times, never solved
            name="Shortest Path Failed 0",
            url="u",
            rating=1500,
            tags=["shortest paths"],
            solved_count=1000,
        )
        results = _rank([retry], sample_records, settings=settings, limit=5)
        assert [rec.problem.problem_id for rec in results] == ["cf:201/0"]
