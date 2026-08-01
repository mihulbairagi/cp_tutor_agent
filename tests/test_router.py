"""Tests for intent classification and the LangGraph routing graph.

The graph is exercised for real — only the agents at its leaves are stubbed —
so these cover node dispatch, state accumulation and the trace, not just the
classifier.
"""

from __future__ import annotations

import pytest

from app.agents.llm import keyword_intent
from app.agents.recommender import extract_topics
from app.agents.router import TutorRouter
from app.models import CandidateProblem, Recommendation, WeakTopic

WEAK = WeakTopic(
    topic="shortest paths",
    score=0.77,
    confidence=0.8,
    solved=2,
    attempted_unsolved=6,
    days_since_last_solve=240,
    max_rating=1100,
    reasons=["6 attempted but unsolved vs 2 solved (75% struggle rate)."],
)

RECOMMENDATION = Recommendation(
    problem=CandidateProblem(
        problem_id="cf:20/C",
        name="Dijkstra?",
        url="https://codeforces.com/problemset/problem/20/C",
        rating=1500,
        tags=["shortest paths"],
    ),
    score=0.8,
    matched_topics=["shortest paths"],
    reason="Targets your weak topic shortest paths.",
)


class FakeWeakTopicAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, *, limit=5, include_blind_spots=True, topic_filter=None):
        self.calls.append({"limit": limit, "topic_filter": topic_filter})
        return [WEAK]

    def profile(self):
        return {"total_solved": 25, "distinct_topics": 6, "total_attempted_unsolved": 6}


class FakeRecommenderAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, *, limit=10, topics=None, rating_override=None):
        self.calls.append({"limit": limit, "topics": topics})
        return [RECOMMENDATION], [WEAK]


class FakeVectorStore:
    def __init__(self, hits=None, fail=False) -> None:
        self.hits = hits or [
            {
                "name": "Dijkstra?",
                "url": "https://codeforces.com/problemset/problem/20/C",
                "source": "codeforces",
                "tags": ["shortest paths"],
                "rating": 1900,
                "difficulty": "Hard",
                "solved": False,
                "similarity": 0.81,
            }
        ]
        self.fail = fail

    def search(self, query, *, limit=5, source=None, solved=None):
        if self.fail:
            raise RuntimeError("index unavailable")
        return self.hits


@pytest.fixture
def router(settings):
    return TutorRouter(
        settings=settings,
        weak_topic_agent=FakeWeakTopicAgent(),
        recommender_agent=FakeRecommenderAgent(),
        vector_store=FakeVectorStore(),
    )


class TestKeywordClassifier:
    """The fallback path, which is what runs with no API key — including in CI."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("what should I practice today", "recommend"),
            ("recommend me some problems", "recommend"),
            ("what to solve next", "recommend"),
            ("give me 5 graph problems", "recommend"),
            ("why am I weak at dp", "weak_topics"),
            ("what are my weakest topics", "weak_topics"),
            ("which topics do I struggle with", "weak_topics"),
            ("where should I improve", "weak_topics"),
            ("show me problems like the ones I failed", "search"),
            ("find me problems about trees", "search"),
            ("how many problems have I solved", "profile"),
            ("what is my rating", "profile"),
            ("summarise my progress", "profile"),
            ("hello there", "general"),
            ("", "general"),
        ],
    )
    def test_questions_route_to_the_right_intent(self, question: str, expected: str) -> None:
        assert keyword_intent(question) == expected

    def test_a_practice_request_naming_a_weakness_is_a_recommendation(self) -> None:
        """Specificity ordering: 'practise' wins over 'weak' in the same sentence."""
        assert keyword_intent("what should I practice to fix my weak dp") == "recommend"


class TestTopicExtraction:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("give me some dp problems", "dynamic programming"),
            ("I want graph problems", "graphs"),
            ("practise dsu", "union find"),
            ("more bfs please", "graph traversal"),
            ("segment tree drills", "segment tree"),
        ],
    )
    def test_spoken_topic_names_map_to_canonical_topics(self, question: str, expected: str) -> None:
        assert expected in extract_topics(question)

    def test_longer_phrases_win_over_substrings(self) -> None:
        assert "binary search" in extract_topics("binary search problems")

    def test_a_question_with_no_topic_yields_nothing(self) -> None:
        assert extract_topics("what should I do today") == []


class TestGraph:
    def test_a_recommendation_question_visits_the_recommend_node(self, router) -> None:
        result = router.ask("what should I practice today")
        assert result["intent"] == "recommend"
        assert result["trace"] == ["classify", "recommend", "synthesize"]
        assert result["recommendations"][0].problem.name == "Dijkstra?"

    def test_a_weakness_question_visits_the_weak_topics_node(self, router) -> None:
        result = router.ask("what are my weakest topics")
        assert result["intent"] == "weak_topics"
        assert result["trace"] == ["classify", "weak_topics", "synthesize"]
        assert result["weak_topics"][0].topic == "shortest paths"

    def test_a_search_question_visits_the_search_node(self, router) -> None:
        result = router.ask("show me problems like the ones I failed")
        assert result["intent"] == "search"
        assert result["search_hits"][0]["name"] == "Dijkstra?"

    def test_a_profile_question_visits_the_profile_node(self, router) -> None:
        result = router.ask("how many problems have I solved")
        assert result["intent"] == "profile"
        assert result["profile"]["total_solved"] == 25

    def test_a_vague_question_still_returns_grounded_data(self, router) -> None:
        """'general' must not mean 'improvise' — real numbers are attached."""
        result = router.ask("help me get better")
        assert result["intent"] == "general"
        assert result["profile"]
        assert result["weak_topics"]

    def test_a_named_topic_narrows_the_weakness_lookup(self, settings) -> None:
        agent = FakeWeakTopicAgent()
        router = TutorRouter(
            settings=settings,
            weak_topic_agent=agent,
            recommender_agent=FakeRecommenderAgent(),
            vector_store=FakeVectorStore(),
        )
        router.ask("why am I weak at dp")
        assert agent.calls[0]["topic_filter"] == "dynamic programming"

    def test_the_limit_is_passed_through_to_the_agent(self, settings) -> None:
        agent = FakeRecommenderAgent()
        router = TutorRouter(
            settings=settings,
            weak_topic_agent=FakeWeakTopicAgent(),
            recommender_agent=agent,
            vector_store=FakeVectorStore(),
        )
        router.ask("what should I practice", limit=3)
        assert agent.calls[0]["limit"] == 3

    def test_every_answer_reports_how_it_was_produced(self, router) -> None:
        """Honesty about the fallback: no key means keyword + template."""
        result = router.ask("what should I practice today")
        assert result["intent_method"] == "keyword"
        assert result["answer_method"] == "template"
        assert result["answer"]

    def test_a_broken_index_does_not_crash_the_graph(self, settings) -> None:
        router = TutorRouter(
            settings=settings,
            weak_topic_agent=FakeWeakTopicAgent(),
            recommender_agent=FakeRecommenderAgent(),
            vector_store=FakeVectorStore(fail=True),
        )
        result = router.ask("find me problems about trees")
        assert result["search_hits"] == []
        assert "sync" in result["answer"].lower()


class TestTemplateAnswers:
    def test_recommendations_render_as_a_practice_list(self, router) -> None:
        answer = router.ask("what should I practice today")["answer"]
        assert "Dijkstra?" in answer
        assert "1500" in answer

    def test_weakness_answers_include_the_reasoning(self, router) -> None:
        answer = router.ask("what are my weakest topics")["answer"]
        assert "shortest paths" in answer
        assert "struggle rate" in answer
