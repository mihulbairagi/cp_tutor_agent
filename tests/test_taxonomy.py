"""Tests for the cross-platform tag taxonomy."""

from __future__ import annotations

import pytest

from app.services.taxonomy import (
    difficulty_from_rating,
    normalize_tag,
    normalize_tags,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The headline case: the two platforms' names for the same idea must
        # collapse to one topic, or the analyser double-counts.
        ("dp", "dynamic programming"),
        ("Dynamic Programming", "dynamic programming"),
        ("dsu", "union find"),
        ("Union Find", "union find"),
        ("dfs and similar", "graph traversal"),
        ("Depth-First Search", "graph traversal"),
        ("Breadth-First Search", "graph traversal"),
        ("sortings", "sorting"),
        ("Heap (Priority Queue)", "heap"),
        ("Binary Indexed Tree", "segment tree"),
        ("bitmasks", "bit manipulation"),
    ],
)
def test_aliases_collapse_to_canonical_topic(raw: str, expected: str) -> None:
    assert normalize_tag(raw) == expected


def test_codeforces_and_leetcode_agree_after_normalisation() -> None:
    codeforces = normalize_tags(["dp", "dsu", "dfs and similar"])
    leetcode = normalize_tags(["Dynamic Programming", "Union Find", "Depth-First Search"])
    assert codeforces == leetcode


@pytest.mark.parametrize("raw", ["implementation", "Database", "shell", "Concurrency", "design"])
def test_noise_tags_are_dropped(raw: str) -> None:
    assert normalize_tag(raw) is None


def test_unknown_tags_pass_through_normalised() -> None:
    """New upstream tags must keep counting rather than vanishing."""
    assert normalize_tag("  Quantum   Annealing ") == "quantum annealing"


def test_normalize_tags_dedupes_and_preserves_order() -> None:
    # 'dp' and 'Dynamic Programming' collapse to one entry, and the first tag
    # upstream assigned stays first so the embedding document leads with it.
    assert normalize_tags(["graphs", "dp", "Dynamic Programming", "graph"]) == [
        "graphs",
        "dynamic programming",
    ]


def test_normalize_tags_handles_empty_input() -> None:
    assert normalize_tags(None) == []
    assert normalize_tags([]) == []
    assert normalize_tags(["", "  ", "implementation"]) == []


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (None, None),
        (800, "Easy"),
        (1199, "Easy"),
        (1200, "Medium"),
        (1899, "Medium"),
        (1900, "Hard"),
    ],
)
def test_rating_buckets_map_to_leetcode_vocabulary(
    rating: int | None, expected: str | None
) -> None:
    assert difficulty_from_rating(rating) == expected
