"""Tests for the HTTP surface.

The app is exercised through ``TestClient`` with every external dependency
replaced via ``app.dependency_overrides`` — so these assert the *contract*
(status codes, response shape, query-parameter plumbing) without touching
Codeforces, LeetCode, Chroma or an LLM.

The ``lifespan`` is deliberately not run: ``TestClient`` is used without a
context manager, so no scheduler thread is started.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.agents.router import get_router
from app.main import app, get_recommender_agent, get_weak_topic_agent
from app.models import (
    CandidateProblem,
    Recommendation,
    SourceSyncResult,
    SyncResult,
    UserRating,
    WeakTopic,
)
from app.services.ingestion import get_ingestion_service
from app.store.vector_store import get_vector_store

SYNCED_AT = datetime(2026, 7, 30, 9, 14, 22, tzinfo=UTC)

WEAK = WeakTopic(
    topic="shortest paths",
    score=0.74,
    confidence=0.4,
    solved=2,
    attempted_unsolved=6,
    days_since_last_solve=214,
    max_rating=1300,
    signals={"coverage": 0.8, "staleness": 1.0, "difficulty": 0.75, "struggle": 0.75},
    reasons=["6 attempted but unsolved vs 2 solved (75% struggle rate)."],
)

RECOMMENDATION = Recommendation(
    problem=CandidateProblem(
        problem_id="cf:20/C",
        contest_id=20,
        index="C",
        name="Dijkstra?",
        url="https://codeforces.com/problemset/problem/20/C",
        rating=1500,
        tags=["graphs", "shortest paths"],
        solved_count=48000,
    ),
    score=0.79,
    matched_topics=["shortest paths"],
    reason="Targets your weak topic shortest paths; rated 1500.",
)

SEARCH_HIT = {
    "name": "Dijkstra?",
    "url": "https://codeforces.com/problemset/problem/20/C",
    "source": "codeforces",
    "tags": ["graphs", "shortest paths"],
    "rating": 1900,
    "difficulty": "Hard",
    "solved": False,
    "similarity": 0.81,
}


class FakeIngestion:
    """Stands in for ``IngestionService``: no HTTP, no vector store."""

    def __init__(self, records, *, has_data: bool = True) -> None:
        self._records = records
        self._has_data = has_data
        self.sync_calls: list[dict] = []

    def has_data(self) -> bool:
        return self._has_data

    def get_records(self):
        return self._records

    def get_user_rating(self) -> UserRating:
        return UserRating(handle="TestUser", rating=1400, max_rating=1450, rank="pupil")

    def get_last_sync(self):
        return SYNCED_AT

    def sync(self, *, force_refresh=False, reset_index=False) -> SyncResult:
        self.sync_calls.append({"force_refresh": force_refresh, "reset_index": reset_index})
        return SyncResult(
            started_at=SYNCED_AT,
            finished_at=SYNCED_AT,
            duration_seconds=1.5,
            total_records=2,
            indexed=2,
            sources=[
                SourceSyncResult(source="codeforces", fetched=2, solved=1, attempted_unsolved=1),
                SourceSyncResult(
                    source="leetcode", fetched=0, ok=False, detail="upstream unavailable"
                ),
            ],
        )


class FakeWeakTopicAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, *, limit=5, include_blind_spots=True, topic_filter=None):
        self.calls.append(
            {
                "limit": limit,
                "include_blind_spots": include_blind_spots,
                "topic_filter": topic_filter,
            }
        )
        return [WEAK]


class FakeRecommenderAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, *, limit=10, topics=None, rating_override=None):
        self.calls.append({"limit": limit, "topics": topics, "rating_override": rating_override})
        return [RECOMMENDATION], [WEAK]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def count(self) -> int:
        return 412

    def search(self, query, *, limit=5, source=None, solved=None):
        self.calls.append({"query": query, "limit": limit, "source": source, "solved": solved})
        return [SEARCH_HIT]


class FakeRouter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask(self, question: str, *, limit: int = 5) -> dict:
        self.calls.append({"question": question, "limit": limit})
        return {
            "question": question,
            "intent": "recommend",
            "intent_method": "keyword",
            "answer": "Practice these next: Dijkstra? [1500]",
            "answer_method": "template",
            "trace": ["classify", "recommend", "synthesize"],
            "weak_topics": [WEAK],
            "recommendations": [RECOMMENDATION],
            "search_hits": [],
            "profile": {},
        }


@pytest.fixture
def stubs(sample_records):
    """All fake collaborators, wired into the app for the duration of a test."""
    bundle = {
        "ingestion": FakeIngestion(sample_records),
        "weak_topics": FakeWeakTopicAgent(),
        "recommender": FakeRecommenderAgent(),
        "store": FakeVectorStore(),
        "router": FakeRouter(),
    }
    app.dependency_overrides[get_ingestion_service] = lambda: bundle["ingestion"]
    app.dependency_overrides[get_weak_topic_agent] = lambda: bundle["weak_topics"]
    app.dependency_overrides[get_recommender_agent] = lambda: bundle["recommender"]
    app.dependency_overrides[get_vector_store] = lambda: bundle["store"]
    app.dependency_overrides[get_router] = lambda: bundle["router"]
    yield bundle
    app.dependency_overrides.clear()


@pytest.fixture
def client(stubs) -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health_reports_ok_when_data_is_present(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["indexed_problems"] == 412
        assert body["last_sync_at"] is not None

    def test_health_reports_degraded_before_the_first_sync(self, client, stubs) -> None:
        stubs["ingestion"]._has_data = False
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["indexed_problems"] == 0

    def test_health_exposes_the_configuration_that_changes_behaviour(self, client) -> None:
        """No key in the test env, so the honest answer is False."""
        body = client.get("/health").json()
        assert body["llm_enabled"] is False
        assert body["leetcode_full_history"] is False
        assert body["codeforces_handle"] == "TestUser"

    def test_health_is_available_without_any_synced_data(self, client, stubs) -> None:
        stubs["ingestion"]._has_data = False
        assert client.get("/health").status_code == 200


class TestSync:
    def test_sync_returns_per_source_results(self, client) -> None:
        body = client.post("/sync").json()["result"]
        assert body["total_records"] == 2
        assert {source["source"] for source in body["sources"]} == {"codeforces", "leetcode"}

    def test_one_failing_source_does_not_fail_the_request(self, client) -> None:
        """A LeetCode outage must still yield a usable Codeforces sync."""
        response = client.post("/sync")
        assert response.status_code == 200
        sources = {s["source"]: s for s in response.json()["result"]["sources"]}
        assert sources["codeforces"]["ok"] is True
        assert sources["leetcode"]["ok"] is False
        assert sources["leetcode"]["detail"]

    def test_query_flags_reach_the_service(self, client, stubs) -> None:
        client.post("/sync?force=true&reset_index=true")
        assert stubs["ingestion"].sync_calls == [{"force_refresh": True, "reset_index": True}]


class TestGuard:
    """``require_synced_data`` — 409 beats a misleading empty 200."""

    @pytest.mark.parametrize(
        ("method", "url", "body"),
        [
            ("get", "/profile", None),
            ("get", "/weak-topics", None),
            ("get", "/recommendations", None),
            ("post", "/ask", {"question": "what should I practice"}),
        ],
    )
    def test_endpoints_needing_a_profile_return_409(self, client, stubs, method, url, body) -> None:
        stubs["ingestion"]._has_data = False
        response = getattr(client, method)(url, **({"json": body} if body else {}))
        assert response.status_code == 409
        assert "sync" in response.json()["detail"].lower()

    def test_search_works_without_a_profile(self, client, stubs) -> None:
        """Search hits the index directly, so it has nothing to guard."""
        stubs["ingestion"]._has_data = False
        assert client.get("/search?q=graphs").status_code == 200


class TestProfile:
    def test_profile_merges_rating_with_computed_statistics(self, client) -> None:
        body = client.get("/profile").json()
        assert body["current_rating"] == 1400
        assert body["rank"] == "pupil"
        assert body["total_solved"] == 25  # from sample_records
        assert body["total_attempted_unsolved"] == 6
        assert body["solved_by_source"] == {"codeforces": 24, "leetcode": 1}


class TestWeakTopics:
    def test_weak_topics_returns_ranked_topics_with_reasoning(self, client) -> None:
        body = client.get("/weak-topics").json()
        assert body["topics"][0]["topic"] == "shortest paths"
        assert body["topics"][0]["reasons"]
        assert body["topics"][0]["signals"]["struggle"] == 0.75

    def test_analysed_records_reports_the_sample_size(self, client) -> None:
        body = client.get("/weak-topics").json()
        assert body["analysed_records"] == 31

    def test_query_parameters_reach_the_agent(self, client, stubs) -> None:
        client.get("/weak-topics?limit=3&include_blind_spots=false&topic=graphs")
        assert stubs["weak_topics"].calls == [
            {"limit": 3, "include_blind_spots": False, "topic_filter": "graphs"}
        ]

    @pytest.mark.parametrize("limit", [0, 51, "abc"])
    def test_an_out_of_range_limit_is_rejected(self, client, limit) -> None:
        assert client.get(f"/weak-topics?limit={limit}").status_code == 422


class TestRecommendations:
    def test_recommendations_are_returned_with_their_justification(self, client) -> None:
        body = client.get("/recommendations").json()
        first = body["recommendations"][0]
        assert first["problem"]["name"] == "Dijkstra?"
        assert first["reason"]
        assert body["based_on_topics"] == ["shortest paths"]

    def test_the_rating_band_is_derived_from_the_users_rating(self, client) -> None:
        """1400 with the default -100/+300 band."""
        assert client.get("/recommendations").json()["rating_band"] == "1300-1700"

    def test_an_overridden_rating_moves_the_band(self, client) -> None:
        assert client.get("/recommendations?rating=1600").json()["rating_band"] == "1500-1900"

    def test_forced_topics_reach_the_agent(self, client, stubs) -> None:
        client.get("/recommendations?limit=4&topics=graphs&topics=segment tree")
        assert stubs["recommender"].calls == [
            {"limit": 4, "topics": ["graphs", "segment tree"], "rating_override": None}
        ]


class TestSearch:
    def test_search_returns_hits_with_similarity(self, client) -> None:
        body = client.get("/search?q=graph problems I failed").json()
        assert body["query"] == "graph problems I failed"
        assert body["hits"][0]["similarity"] == 0.81

    def test_filters_reach_the_store(self, client, stubs) -> None:
        client.get("/search?q=trees&limit=3&source=codeforces&solved=false")
        assert stubs["store"].calls == [
            {"query": "trees", "limit": 3, "source": "codeforces", "solved": False}
        ]

    def test_an_empty_query_is_rejected(self, client) -> None:
        assert client.get("/search?q=").status_code == 422


class TestAsk:
    def test_ask_returns_prose_and_the_structured_data_behind_it(self, client) -> None:
        body = client.post("/ask", json={"question": "what should I practice", "limit": 3}).json()
        assert body["answer"]
        assert body["intent"] == "recommend"
        assert body["recommendations"][0]["problem"]["name"] == "Dijkstra?"

    def test_the_response_reports_which_path_produced_it(self, client) -> None:
        body = client.post("/ask", json={"question": "what should I practice"}).json()
        assert body["intent_method"] == "keyword"
        assert body["answer_method"] == "template"
        assert body["trace"] == ["classify", "recommend", "synthesize"]

    def test_the_limit_reaches_the_router(self, client, stubs) -> None:
        client.post("/ask", json={"question": "why am I weak at dp", "limit": 2})
        assert stubs["router"].calls == [{"question": "why am I weak at dp", "limit": 2}]

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "x", "limit": 0},
            {"question": "x", "limit": 26},
        ],
    )
    def test_invalid_payloads_are_rejected(self, client, payload) -> None:
        assert client.post("/ask", json=payload).status_code == 422


class TestDocs:
    def test_the_openapi_schema_is_generated(self, client) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "CP Tutor Agent"
        assert "/ask" in schema["paths"]

    def test_every_route_is_documented_and_tagged(self, client) -> None:
        """A wall of untyped 200s is not documentation."""
        schema = client.get("/openapi.json").json()
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                assert operation.get("summary"), f"{method.upper()} {path} has no summary"
                assert operation.get("description"), f"{method.upper()} {path} has no description"
                assert operation.get("tags"), f"{method.upper()} {path} has no tag"

    def test_the_root_redirects_to_the_docs(self, client) -> None:
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (307, 308)
        assert response.headers["location"] == "/docs"
