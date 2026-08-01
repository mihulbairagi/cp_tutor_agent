"""Tests for LeetCode GraphQL parsing and the public/authenticated path switch."""

from __future__ import annotations

import pytest

from app.clients.leetcode import (
    LeetCodeError,
    _unwrap,
    parse_profile_stats,
    parse_question_detail,
    parse_recent_accepted,
    parse_solved_list,
    parse_solved_total,
)
from app.config import Settings


class TestEnvelope:
    def test_data_is_unwrapped(self) -> None:
        assert _unwrap({"data": {"x": 1}}) == {"x": 1}

    def test_graphql_errors_raise_even_on_http_200(self) -> None:
        """GraphQL reports failures with a 200, so the transport layer cannot."""
        with pytest.raises(LeetCodeError, match="that topic does not exist"):
            _unwrap({"errors": [{"message": "that topic does not exist"}], "data": None})

    def test_missing_data_raises(self) -> None:
        with pytest.raises(LeetCodeError):
            _unwrap({})


class TestProfileStats:
    def test_solved_counts_by_difficulty(self, leetcode_profile_payload) -> None:
        stats = parse_profile_stats(leetcode_profile_payload["data"])
        assert stats == {"All": 140, "Easy": 80, "Medium": 50, "Hard": 10}

    def test_unknown_user_raises(self) -> None:
        with pytest.raises(LeetCodeError, match="not found"):
            parse_profile_stats({"matchedUser": None})


class TestRecentAccepted:
    def test_slugs_and_timestamps_are_parsed(self, leetcode_recent_payload) -> None:
        parsed = parse_recent_accepted(leetcode_recent_payload["data"])
        assert [item["titleSlug"] for item in parsed] == [
            "two-sum",
            "course-schedule",
            "two-sum",
        ]
        assert parsed[0]["solved_at"].timestamp() == 1_700_000_000

    def test_malformed_timestamp_degrades_to_none(self) -> None:
        parsed = parse_recent_accepted(
            {"recentAcSubmissionList": [{"titleSlug": "x", "title": "X", "timestamp": "abc"}]}
        )
        assert parsed[0]["solved_at"] is None

    def test_entries_without_a_slug_are_skipped(self) -> None:
        parsed = parse_recent_accepted({"recentAcSubmissionList": [{"title": "X"}]})
        assert parsed == []


class TestQuestionDetail:
    def test_question_object_is_returned(self, leetcode_question_payload) -> None:
        question = parse_question_detail(leetcode_question_payload["data"])
        assert question["difficulty"] == "Medium"
        assert len(question["topicTags"]) == 3

    def test_missing_question_returns_none(self) -> None:
        assert parse_question_detail({"question": None}) is None


class TestSolvedList:
    def test_accepted_questions_become_records(self, leetcode_solved_list_payload) -> None:
        records = parse_solved_list(leetcode_solved_list_payload["data"])
        assert len(records) == 1
        record = records[0]
        assert record.problem_id == "lc:two-sum"
        assert record.solved is True
        assert record.difficulty == "Easy"
        assert record.url == "https://leetcode.com/problems/two-sum/"

    def test_non_accepted_questions_are_filtered_out(self, leetcode_solved_list_payload) -> None:
        """Defensive: the same query is also used for general browsing."""
        slugs = {r.problem_id for r in parse_solved_list(leetcode_solved_list_payload["data"])}
        assert "lc:median-of-two-sorted-arrays" not in slugs

    def test_tags_are_normalised_onto_the_shared_taxonomy(
        self, leetcode_solved_list_payload
    ) -> None:
        record = parse_solved_list(leetcode_solved_list_payload["data"])[0]
        # "Array" -> "arrays", "Hash Table" -> "hashing"
        assert record.tags == ["arrays", "hashing"]

    def test_total_is_read_for_pagination(self, leetcode_solved_list_payload) -> None:
        assert parse_solved_total(leetcode_solved_list_payload["data"]) == 2

    def test_leetcode_records_carry_no_numeric_rating(self, leetcode_solved_list_payload) -> None:
        assert parse_solved_list(leetcode_solved_list_payload["data"])[0].rating is None


class TestAuthenticationSwitch:
    """The public endpoint caps history at ~20 solves; a cookie unlocks it all."""

    def test_absent_cookie_means_public_mode(self) -> None:
        assert Settings(leetcode_session="").leetcode_authenticated is False

    def test_present_cookie_means_authenticated_mode(self) -> None:
        assert Settings(leetcode_session="abc123").leetcode_authenticated is True

    def test_whitespace_only_cookie_is_treated_as_absent(self) -> None:
        assert Settings(leetcode_session="   ").leetcode_authenticated is False
