"""Tests for the weak-topic scorer.

This is the module the whole product rests on, so each of the four signals is
pinned independently, using the fixed clock from ``conftest.NOW`` so results
never drift with the calendar.

The ``sample_records`` fixture is shaped to isolate the signals:

* ``greedy``               — 12 recent solves up to 1800. The control.
* ``shortest paths``       — 2 stale low-rated solves, 6 failures. The weakness.
* ``dynamic programming``  — 10 recent solves, all rated 900. Difficulty gap only.
* ``geometry``             — 1 recent solve. Coverage gap only.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from app.services.analysis import (
    analyse_weak_topics,
    build_topic_stats,
    score_topics,
    summarise_profile,
    topic_frequency,
)
from tests.conftest import NOW


class TestBuildTopicStats:
    def test_a_multi_tag_problem_counts_towards_every_topic(self, sample_records) -> None:
        stats = build_topic_stats(sample_records)
        # Each shortest-path problem is also tagged `graphs`.
        assert stats["shortest paths"].solved == stats["graphs"].solved == 2
        assert stats["shortest paths"].attempted_unsolved == 6

    def test_solved_and_unsolved_are_tracked_separately(self, sample_records) -> None:
        stats = build_topic_stats(sample_records)
        assert stats["greedy"].solved == 12
        assert stats["greedy"].attempted_unsolved == 0

    def test_failed_submissions_are_summed_not_just_counted(self, sample_records) -> None:
        # 6 unsolved problems x 3 submissions each.
        assert build_topic_stats(sample_records)["shortest paths"].total_failed_submissions == 18

    def test_max_rating_comes_from_solved_problems_only(self, sample_records) -> None:
        stats = build_topic_stats(sample_records)
        # The 1500-rated shortest-path problems were never solved.
        assert stats["shortest paths"].max_rating == 1100

    def test_last_solved_is_the_most_recent_acceptance(self, sample_records) -> None:
        assert build_topic_stats(sample_records)["greedy"].last_solved_at == NOW - timedelta(days=5)

    def test_sources_are_recorded(self, sample_records) -> None:
        assert build_topic_stats(sample_records)["geometry"].sources == ["leetcode"]

    def test_no_records_yields_no_stats(self) -> None:
        assert build_topic_stats([]) == {}


class TestSignals:
    """Each signal is asserted on the topic engineered to isolate it."""

    def _by_topic(self, sample_records, settings):
        ranked = score_topics(build_topic_stats(sample_records), settings=settings, now=NOW)
        return {topic.topic: topic for topic in ranked}

    def test_coverage_reflects_distance_from_the_solve_target(
        self, sample_records, settings
    ) -> None:
        topics = self._by_topic(sample_records, settings)
        # 1 solve against a target of 10.
        assert topics["geometry"].signals["coverage"] == 0.9
        # 12 solves is past the target, so no coverage gap at all.
        assert topics["greedy"].signals["coverage"] == 0.0

    def test_staleness_saturates_past_the_horizon(self, sample_records, settings) -> None:
        topics = self._by_topic(sample_records, settings)
        # Last solved 240 days ago against a 120-day horizon.
        assert topics["shortest paths"].signals["staleness"] == 1.0
        assert topics["shortest paths"].days_since_last_solve == 240
        assert topics["greedy"].signals["staleness"] < 0.1

    def test_difficulty_gap_catches_grinding_easy_problems(self, sample_records, settings) -> None:
        """Ten easy DP solves is not DP competence — the point of this signal."""
        topics = self._by_topic(sample_records, settings)
        assert topics["dynamic programming"].signals["coverage"] == 0.0
        assert topics["dynamic programming"].signals["difficulty"] == 1.0
        assert topics["greedy"].signals["difficulty"] == 0.0

    def test_struggle_rate_is_unsolved_over_touched(self, sample_records, settings) -> None:
        topics = self._by_topic(sample_records, settings)
        assert topics["shortest paths"].signals["struggle"] == 0.75  # 6 of 8
        assert topics["greedy"].signals["struggle"] == 0.0


class TestRanking:
    def test_the_engineered_weakness_ranks_first(self, sample_records, settings) -> None:
        ranked = analyse_weak_topics(sample_records, settings=settings, limit=2, now=NOW)
        assert {topic.topic for topic in ranked} == {"shortest paths", "graphs"}

    def test_the_control_topic_ranks_last(self, sample_records, settings) -> None:
        ranked = analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        assert ranked[-1].topic == "greedy"

    def test_scores_are_bounded(self, sample_records, settings) -> None:
        for topic in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW):
            assert 0.0 <= topic.score <= 1.0
            assert 0.0 <= topic.confidence <= 1.0

    def test_confidence_tracks_sample_size(self, sample_records, settings) -> None:
        ranked = {
            t.topic: t
            for t in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        }
        assert ranked["greedy"].confidence == 1.0  # 12 problems
        assert ranked["geometry"].confidence == 0.1  # 1 problem

    def test_low_confidence_is_damped_below_its_raw_score(self, sample_records, settings) -> None:
        """A topic with one data point must not outrank one with thirty."""
        ranked = {
            t.topic: t
            for t in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        }
        geometry = ranked["geometry"]
        weights = settings.weakness_weights
        raw = sum(weights[name] * value for name, value in geometry.signals.items()) / sum(
            weights.values()
        )
        assert geometry.score < raw

    def test_ranking_is_deterministic(self, sample_records, settings) -> None:
        """The same history must always produce the same ranking."""
        first = analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        second = analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        assert [t.topic for t in first] == [t.topic for t in second]
        assert [t.score for t in first] == [t.score for t in second]

    def test_weights_are_configurable(self, sample_records, settings) -> None:
        """Retuning the ranking must not require a code change."""
        settings.weight_struggle = 1.0
        settings.weight_coverage = 0.0
        settings.weight_staleness = 0.0
        settings.weight_difficulty = 0.0
        ranked = analyse_weak_topics(sample_records, settings=settings, limit=1, now=NOW)
        assert ranked[0].signals["struggle"] == 0.75


class TestReasons:
    def test_every_topic_explains_itself(self, sample_records, settings) -> None:
        for topic in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW):
            assert topic.reasons, f"{topic.topic} produced no explanation"

    def test_reasons_are_ordered_by_what_actually_moved_the_score(
        self, sample_records, settings
    ) -> None:
        """Weighted contribution, not raw signal value.

        Shortest paths has a *saturated* staleness signal (1.0) but staleness
        only carries a 0.25 weight, while coverage sits at 0.8 on a 0.35 weight.
        Coverage moved the score more, so it must be explained first — leading
        with the bigger-looking number would misdirect the reader.
        """
        topic = {
            t.topic: t
            for t in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        }["shortest paths"]

        weights = settings.weakness_weights
        strongest = max(weights, key=lambda name: weights[name] * topic.signals[name])
        assert strongest == "coverage"
        assert "below the 10-problem target" in topic.reasons[0]
        assert topic.signals["staleness"] > topic.signals["coverage"]  # the tempting wrong answer

    def test_reasons_quote_the_real_numbers(self, sample_records, settings) -> None:
        ranked = {
            t.topic: t
            for t in analyse_weak_topics(sample_records, settings=settings, limit=0, now=NOW)
        }
        joined = " ".join(ranked["shortest paths"].reasons)
        assert "6 attempted but unsolved" in joined
        assert "240 days ago" in joined


class TestBlindSpots:
    def test_untouched_common_topics_are_surfaced(self, sample_records, settings) -> None:
        stats = build_topic_stats(sample_records)
        ranked = score_topics(
            stats,
            settings=settings,
            now=NOW,
            problemset_frequency=Counter({"segment tree": 500}),
        )
        blind = next(topic for topic in ranked if topic.topic == "segment tree")
        assert blind.solved == 0
        assert blind.confidence == 0.0
        assert "blind spot" in blind.reasons[0].lower()

    def test_rare_tags_are_not_treated_as_blind_spots(self, sample_records, settings) -> None:
        """An obscure tag on four problems is not an urgent gap."""
        ranked = score_topics(
            build_topic_stats(sample_records),
            settings=settings,
            now=NOW,
            problemset_frequency=Counter({"expression parsing": 4}),
        )
        assert all(topic.topic != "expression parsing" for topic in ranked)

    def test_a_blind_spot_does_not_outrank_a_measured_weakness(
        self, sample_records, settings
    ) -> None:
        ranked = score_topics(
            build_topic_stats(sample_records),
            settings=settings,
            now=NOW,
            problemset_frequency=Counter({"segment tree": 500}),
        )
        by_topic = {topic.topic: topic.score for topic in ranked}
        assert by_topic["shortest paths"] > by_topic["segment tree"]


class TestTopicFrequency:
    def test_topics_are_counted_across_the_problemset(self, sample_candidates) -> None:
        frequency = topic_frequency(sample_candidates)
        assert frequency["shortest paths"] == 6
        assert frequency["greedy"] == 1


class TestSummariseProfile:
    def test_headline_numbers(self, sample_records) -> None:
        summary = summarise_profile(sample_records)
        assert summary["total_problems_touched"] == 31
        assert summary["total_solved"] == 25
        assert summary["total_attempted_unsolved"] == 6

    def test_breakdown_by_source(self, sample_records) -> None:
        assert summarise_profile(sample_records)["solved_by_source"] == {
            "codeforces": 24,
            "leetcode": 1,
        }

    def test_rating_summary_ignores_unrated_solves(self, sample_records) -> None:
        assert summarise_profile(sample_records)["max_rating_solved"] == 1750

    def test_empty_profile_is_survivable(self) -> None:
        summary = summarise_profile([])
        assert summary["total_solved"] == 0
        assert summary["max_rating_solved"] is None
