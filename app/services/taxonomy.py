"""Cross-platform topic taxonomy.

The problem this solves
-----------------------
Codeforces tags a problem ``dp``. LeetCode tags the same idea
``Dynamic Programming``. Codeforces says ``dsu``; LeetCode says ``Union Find``.
If you aggregate the two feeds without reconciling vocabularies you get a skill
profile that reports "dp: 40 solves" and "dynamic programming: 12 solves" as two
unrelated topics — and the weak-topic ranking becomes noise.

So every tag from either source is mapped through :data:`TAG_ALIASES` onto a
single canonical topic name before it reaches the analyser.

Deliberate choices
------------------
* **Unknown tags pass through** (lowercased and whitespace-normalised) rather
  than being dropped. When LeetCode adds a new tag next month the system keeps
  counting it instead of silently losing data.
* **Some tags are intentionally discarded** — ``Database``, ``Shell`` and
  ``Concurrency`` are LeetCode categories that have nothing to do with
  competitive programming skill, and ``implementation`` is a Codeforces catch-all
  applied to a third of the problemset, so it carries no signal.
* The map is a plain dict of pure data, which makes it trivially unit-testable
  and easy to extend without touching logic.
"""

from __future__ import annotations

# Tags that exist upstream but say nothing useful about algorithmic weakness.
IGNORED_TAGS: frozenset[str] = frozenset(
    {
        "implementation",  # Codeforces applies this to a huge share of problems
        "database",  # LeetCode SQL track
        "shell",  # LeetCode bash track
        "concurrency",  # LeetCode threading track
        "design",  # LeetCode API-design questions
        "*special",  # Codeforces marker for gimmick problems
    }
)

# raw upstream tag (lowercased) -> canonical topic
TAG_ALIASES: dict[str, str] = {
    # --- dynamic programming ------------------------------------------------
    "dp": "dynamic programming",
    "dynamic programming": "dynamic programming",
    "memoization": "dynamic programming",
    # --- greedy -------------------------------------------------------------
    "greedy": "greedy",
    # --- graphs -------------------------------------------------------------
    "graphs": "graphs",
    "graph": "graphs",
    "graph matchings": "graphs",
    "strongly connected component": "graphs",
    "biconnected component": "graphs",
    "eulerian circuit": "graphs",
    "topological sort": "graphs",
    "flows": "graphs",
    "minimum spanning tree": "graphs",
    # --- graph traversal ----------------------------------------------------
    "dfs and similar": "graph traversal",
    "depth-first search": "graph traversal",
    "breadth-first search": "graph traversal",
    # --- shortest paths -----------------------------------------------------
    "shortest paths": "shortest paths",
    "shortest path": "shortest paths",
    # --- trees --------------------------------------------------------------
    "trees": "trees",
    "tree": "trees",
    "binary tree": "trees",
    "binary search tree": "trees",
    # --- searching ----------------------------------------------------------
    "binary search": "binary search",
    "ternary search": "binary search",
    "two pointers": "two pointers",
    "sliding window": "sliding window",
    # --- sorting ------------------------------------------------------------
    "sortings": "sorting",
    "sorting": "sorting",
    "merge sort": "sorting",
    "counting sort": "sorting",
    "radix sort": "sorting",
    "bucket sort": "sorting",
    "quickselect": "sorting",
    # --- strings ------------------------------------------------------------
    "strings": "strings",
    "string": "strings",
    "string matching": "strings",
    "string suffix structures": "strings",
    "suffix array": "strings",
    "expression parsing": "strings",
    "hashing": "hashing",
    "rolling hash": "hashing",
    "hash function": "hashing",
    "hash table": "hashing",
    "trie": "trie",
    # --- math ---------------------------------------------------------------
    "math": "math",
    "number theory": "number theory",
    "chinese remainder theorem": "number theory",
    "combinatorics": "combinatorics",
    "counting": "combinatorics",
    "enumeration": "combinatorics",
    "probabilities": "probability",
    "probability and statistics": "probability",
    "randomized": "probability",
    "reservoir sampling": "probability",
    "rejection sampling": "probability",
    "geometry": "geometry",
    "line sweep": "geometry",
    "matrices": "matrix",
    "matrix": "matrix",
    "fft": "math",
    # --- data structures ----------------------------------------------------
    "data structures": "data structures",
    "segment tree": "segment tree",
    "binary indexed tree": "segment tree",
    "dsu": "union find",
    "union find": "union find",
    "heap (priority queue)": "heap",
    "heap": "heap",
    "priority queue": "heap",
    "stack": "stack",
    "monotonic stack": "stack",
    "queue": "queue",
    "monotonic queue": "queue",
    "deque": "queue",
    "linked list": "linked list",
    "doubly-linked list": "linked list",
    "ordered set": "data structures",
    "data stream": "data structures",
    "iterator": "data structures",
    "prefix sum": "prefix sum",
    "array": "arrays",
    "arrays": "arrays",
    # --- bit tricks ---------------------------------------------------------
    "bitmasks": "bit manipulation",
    "bitmask": "bit manipulation",
    "bit manipulation": "bit manipulation",
    # --- paradigms ----------------------------------------------------------
    "brute force": "brute force",
    "backtracking": "backtracking",
    "recursion": "recursion",
    "divide and conquer": "divide and conquer",
    "meet-in-the-middle": "divide and conquer",
    "constructive algorithms": "constructive",
    "brainteaser": "constructive",
    "simulation": "simulation",
    "schedules": "greedy",
    "games": "game theory",
    "game theory": "game theory",
    "interactive": "interactive",
    "2-sat": "graphs",
}

#: Every canonical topic the system knows about.
CANONICAL_TOPICS: frozenset[str] = frozenset(TAG_ALIASES.values())


def normalize_tag(tag: str) -> str | None:
    """Map one upstream tag onto its canonical topic.

    Returns ``None`` when the tag is blank or explicitly ignored.
    """
    if not tag:
        return None
    cleaned = " ".join(tag.strip().lower().split())
    if not cleaned or cleaned in IGNORED_TAGS:
        return None
    return TAG_ALIASES.get(cleaned, cleaned)


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalise a list of tags, de-duplicating while preserving order.

    Order is preserved (rather than sorting) so that the first tag upstream
    assigned — usually the primary technique — stays first when we build the
    embedding document.
    """
    if not tags:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        canonical = normalize_tag(tag)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def difficulty_from_rating(rating: int | None) -> str | None:
    """Bucket a Codeforces rating into LeetCode-style difficulty labels.

    Lets the API report one difficulty vocabulary across both platforms. The
    cut-offs follow the rough community consensus that a 1400-rated Codeforces
    problem is about a LeetCode Medium.
    """
    if rating is None:
        return None
    if rating < 1200:
        return "Easy"
    if rating < 1900:
        return "Medium"
    return "Hard"
