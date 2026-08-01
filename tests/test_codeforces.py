"""Tests for Codeforces payload parsing.

All of these run against recorded fixtures — no network. The parsing functions
are pure by design precisely so this is possible.
"""

from __future__ import annotations

import pytest

from app.clients.codeforces import (
    CodeforcesError,
    _unwrap,
    parse_problemset,
    parse_submissions,
    parse_user_info,
)


class TestEnvelope:
    def test_ok_envelope_is_unwrapped(self) -> None:
        assert _unwrap({"status": "OK", "result": [1]}) == [1]

    def test_failed_envelope_raises_with_the_upstream_comment(self) -> None:
        """Codeforces reports errors inside a 200 body, so this check matters."""
        with pytest.raises(CodeforcesError, match="handles: User with handle x not found"):
            _unwrap({"status": "FAILED", "comment": "handles: User with handle x not found"})


class TestParseSubmissions:
    def test_submissions_collapse_to_one_record_per_problem(
        self, codeforces_status_payload
    ) -> None:
        records = parse_submissions(codeforces_status_payload["result"])
        assert len(records) == 2

    def test_a_problem_with_any_accepted_submission_is_solved(
        self, codeforces_status_payload
    ) -> None:
        records = {r.problem_id: r for r in parse_submissions(codeforces_status_payload["result"])}
        watermelon = records["cf:4/A"]
        assert watermelon.solved is True
        # One OK and one WRONG_ANSWER were folded together.
        assert watermelon.attempt_count == 2
        assert watermelon.failed_attempts == 1

    def test_failed_attempts_are_kept_not_discarded(self, codeforces_status_payload) -> None:
        """The whole weakness signal depends on not filtering to verdict == OK."""
        records = {r.problem_id: r for r in parse_submissions(codeforces_status_payload["result"])}
        dijkstra = records["cf:20/C"]
        assert dijkstra.solved is False
        assert dijkstra.attempt_count == 2
        assert dijkstra.solved_at is None

    def test_solved_at_uses_the_earliest_acceptance(self) -> None:
        """Re-submitting an old solve must not look like fresh practice."""
        payload = [
            {
                "creationTimeSeconds": 2_000_000_000,
                "verdict": "OK",
                "problem": {"contestId": 1, "index": "A", "name": "P", "rating": 800, "tags": []},
            },
            {
                "creationTimeSeconds": 1_000_000_000,
                "verdict": "OK",
                "problem": {"contestId": 1, "index": "A", "name": "P", "rating": 800, "tags": []},
            },
        ]
        record = parse_submissions(payload)[0]
        assert record.solved_at.timestamp() == 1_000_000_000
        assert record.last_attempt_at.timestamp() == 2_000_000_000

    def test_tags_are_normalised_and_raw_tags_retained(self, codeforces_status_payload) -> None:
        records = {r.problem_id: r for r in parse_submissions(codeforces_status_payload["result"])}
        watermelon = records["cf:4/A"]
        assert watermelon.tags == ["brute force", "math"]
        assert watermelon.raw_tags == ["brute force", "math"]

    def test_null_verdict_counts_as_an_attempt_not_a_solve(self) -> None:
        """In-queue submissions have verdict None."""
        payload = [
            {
                "creationTimeSeconds": 1_000_000_000,
                "verdict": None,
                "problem": {"contestId": 1, "index": "A", "name": "P", "rating": 800, "tags": []},
            }
        ]
        record = parse_submissions(payload)[0]
        assert record.solved is False
        assert record.attempt_count == 1

    def test_problem_url_is_built_from_contest_and_index(self, codeforces_status_payload) -> None:
        records = {r.problem_id: r for r in parse_submissions(codeforces_status_payload["result"])}
        assert records["cf:20/C"].url == "https://codeforces.com/problemset/problem/20/C"

    def test_empty_and_malformed_input_is_survivable(self) -> None:
        assert parse_submissions([]) == []
        assert parse_submissions([{"verdict": "OK"}]) == []  # no `problem` key


class TestParseUserInfo:
    def test_rating_snapshot_is_extracted(self) -> None:
        rating = parse_user_info(
            [{"handle": "TestUser", "rating": 1387, "maxRating": 1421, "rank": "pupil"}],
            "TestUser",
        )
        assert (rating.rating, rating.max_rating, rating.rank) == (1387, 1421, "pupil")

    def test_unrated_user_yields_none_rather_than_zero(self) -> None:
        """Zero would be a valid-looking rating and would skew the band."""
        rating = parse_user_info([{"handle": "TestUser"}], "TestUser")
        assert rating.rating is None

    def test_empty_result_returns_a_bare_snapshot(self) -> None:
        assert parse_user_info([], "TestUser").handle == "TestUser"


class TestParseProblemset:
    def test_problems_join_to_their_statistics(self, codeforces_problemset_payload) -> None:
        candidates = {
            c.problem_id: c for c in parse_problemset(codeforces_problemset_payload["result"])
        }
        assert candidates["cf:4/A"].solved_count == 250000
        assert candidates["cf:20/C"].solved_count == 48000

    def test_tags_are_normalised(self, codeforces_problemset_payload) -> None:
        candidates = {
            c.problem_id: c for c in parse_problemset(codeforces_problemset_payload["result"])
        }
        assert candidates["cf:20/C"].tags == ["graphs", "shortest paths"]

    def test_missing_statistics_default_to_zero(self) -> None:
        result = {
            "problems": [{"contestId": 1, "index": "A", "name": "P", "rating": 800, "tags": []}],
            "problemStatistics": [],
        }
        assert parse_problemset(result)[0].solved_count == 0

    def test_empty_result_is_survivable(self) -> None:
        assert parse_problemset({}) == []
