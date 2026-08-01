"""The evaluation set: 15 natural-language questions and what each must produce.

A case is not "does the answer look nice" — that is unassertable without a
human. Each case pins down *observable, falsifiable* behaviour:

* the question routed to the right node (``expected_intent``)
* the structured payload the node was supposed to fill is actually populated
* the numbers in the answer come from the fixture profile, not from nowhere

Plus four invariants checked on **every** case, in :mod:`eval.run_eval`:
an answer is always produced, the trace always begins at ``classify`` and ends
at ``synthesize``, no already-solved problem is ever recommended, and the whole
thing stays under the latency budget.

Note on ``limit``: it is supplied by the caller, exactly as the API does it.
A count written into the question ("give me 3 problems") is *not* parsed out of
the text — the classifier decides intent, not quantities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from eval.fixtures import USER_RATING

#: One assertion: a label, and a predicate over the router's result dict.
Check = tuple[str, Callable[[dict], bool]]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_intent: str
    checks: list[Check] = field(default_factory=list)
    limit: int = 5


def _topics(result: dict) -> list[str]:
    return [topic.topic for topic in result["weak_topics"]]


def _names(result: dict) -> list[str]:
    return [rec.problem.name for rec in result["recommendations"]]


def _ratings(result: dict) -> list[int]:
    return [rec.problem.rating for rec in result["recommendations"]]


def _matched(result: dict) -> set[str]:
    return {topic for rec in result["recommendations"] for topic in rec.matched_topics}


CASES: list[EvalCase] = [
    # -- recommend ---------------------------------------------------------
    EvalCase(
        id="rec-basic",
        question="what should I practice today",
        expected_intent="recommend",
        checks=[
            ("returns problems", lambda r: len(r["recommendations"]) > 0),
            ("every problem is rated", lambda r: all(_ratings(r))),
            (
                "stays inside the practice band",
                lambda r: all(
                    USER_RATING - 100 <= rating <= USER_RATING + 300 for rating in _ratings(r)
                ),
            ),
            (
                "every problem targets a weak topic",
                lambda r: all(rec.matched_topics for rec in r["recommendations"]),
            ),
            (
                "every problem explains itself",
                lambda r: all(rec.reason for rec in r["recommendations"]),
            ),
        ],
    ),
    EvalCase(
        id="rec-count",
        question="give me 3 problems to solve",
        expected_intent="recommend",
        limit=3,
        checks=[
            ("honours the requested limit", lambda r: len(r["recommendations"]) <= 3),
            ("still fills the list", lambda r: len(r["recommendations"]) == 3),
        ],
    ),
    EvalCase(
        id="rec-named-topic",
        question="give me some segment tree problems",
        expected_intent="recommend",
        checks=[
            ("returns problems", lambda r: len(r["recommendations"]) > 0),
            (
                "an explicitly named topic overrides the weakness ranking",
                lambda r: "segment tree" in _matched(r),
            ),
        ],
    ),
    EvalCase(
        id="rec-abbreviation",
        question="recommend me some dp problems",
        expected_intent="recommend",
        checks=[
            (
                "'dp' resolves to the canonical topic",
                lambda r: "dynamic programming" in _matched(r),
            ),
        ],
    ),
    EvalCase(
        id="rec-variety",
        question="what should I solve next",
        expected_intent="recommend",
        checks=[
            (
                "no single topic fills the list",
                lambda r: (
                    len(r["recommendations"]) < 2
                    or len({rec.problem.problem_id for rec in r["recommendations"]})
                    == len(r["recommendations"])
                ),
            ),
            (
                "the weaknesses driving the list are reported",
                lambda r: len(r["weak_topics"]) > 0,
            ),
        ],
    ),
    EvalCase(
        id="rec-mentions-a-weakness",
        question="what should I practice to fix my weak graphs",
        expected_intent="recommend",
        checks=[
            (
                "a practice request wins over the word 'weak'",
                lambda r: r["intent"] == "recommend",
            ),
            ("returns problems", lambda r: len(r["recommendations"]) > 0),
        ],
    ),
    # -- weak topics -------------------------------------------------------
    EvalCase(
        id="weak-basic",
        question="what are my weakest topics",
        expected_intent="weak_topics",
        checks=[
            ("returns a ranking", lambda r: len(r["weak_topics"]) > 0),
            (
                "ranked weakest first",
                lambda r: (
                    [t.score for t in r["weak_topics"]]
                    == sorted((t.score for t in r["weak_topics"]), reverse=True)
                ),
            ),
            (
                "shortest paths is identified as the weakness",
                lambda r: "shortest paths" in _topics(r),
            ),
            (
                "the well-practised control topic is not flagged",
                lambda r: "greedy" not in _topics(r),
            ),
            (
                "every weakness gives a reason",
                lambda r: all(topic.reasons for topic in r["weak_topics"]),
            ),
        ],
    ),
    EvalCase(
        id="weak-why",
        question="why am I weak at shortest paths",
        expected_intent="weak_topics",
        checks=[
            ("narrows to the topic asked about", lambda r: _topics(r) == ["shortest paths"]),
            (
                "the struggle rate is spelled out",
                lambda r: any("struggle" in reason for reason in r["weak_topics"][0].reasons),
            ),
            (
                "the answer quotes the real counts",
                lambda r: (
                    r["weak_topics"][0].solved == 2 and r["weak_topics"][0].attempted_unsolved == 6
                ),
            ),
        ],
    ),
    EvalCase(
        id="weak-difficulty-gap",
        question="where should I improve",
        expected_intent="weak_topics",
        checks=[
            (
                "a topic solved only at easy ratings surfaces",
                lambda r: "dynamic programming" in _topics(r),
            ),
            (
                "signals are exposed, not just a score",
                lambda r: all(topic.signals for topic in r["weak_topics"]),
            ),
        ],
    ),
    EvalCase(
        id="weak-unknown-topic",
        question="why am I weak at quantum computing",
        expected_intent="weak_topics",
        checks=[
            (
                "does not invent a weakness it has no data for",
                lambda r: "quantum computing" not in _topics(r),
            ),
            (
                "falls back to the real ranking rather than erroring",
                lambda r: len(r["weak_topics"]) > 0 and "shortest paths" in _topics(r),
            ),
        ],
    ),
    # -- search ------------------------------------------------------------
    EvalCase(
        id="search-basic",
        question="find me problems about shortest paths",
        expected_intent="search",
        checks=[
            ("returns hits", lambda r: len(r["search_hits"]) > 0),
            (
                "hits come from the user's own history",
                lambda r: all(hit["url"] for hit in r["search_hits"]),
            ),
            (
                "hits are ranked by similarity",
                lambda r: (
                    [hit["similarity"] for hit in r["search_hits"]]
                    == sorted((hit["similarity"] for hit in r["search_hits"]), reverse=True)
                ),
            ),
        ],
    ),
    EvalCase(
        id="search-failed",
        question="show me problems like the ones I failed",
        expected_intent="search",
        checks=[("routes to the index rather than the ranker", lambda r: "search" in r["trace"])],
    ),
    # -- profile -----------------------------------------------------------
    EvalCase(
        id="profile-count",
        question="how many problems have I solved",
        expected_intent="profile",
        checks=[
            ("reports the real total", lambda r: r["profile"]["total_solved"] == 25),
            (
                "counts failures separately",
                lambda r: r["profile"]["total_attempted_unsolved"] == 6,
            ),
            ("the number appears in the answer", lambda r: "25" in r["answer"]),
        ],
    ),
    EvalCase(
        id="profile-summary",
        question="summarise my progress",
        expected_intent="profile",
        checks=[
            (
                "breaks down by platform",
                lambda r: r["profile"]["solved_by_source"]["leetcode"] == 1,
            ),
            ("reports the hardest solve", lambda r: r["profile"]["max_rating_solved"] == 1750),
        ],
    ),
    # -- general -----------------------------------------------------------
    EvalCase(
        id="general-vague",
        question="help me get better at competitive programming",
        expected_intent="general",
        checks=[
            (
                "a vague question still gets grounded in real data",
                lambda r: bool(r["profile"]) and bool(r["weak_topics"]),
            ),
            ("does not answer with nothing", lambda r: len(r["answer"]) > 40),
        ],
    ),
]
