"""LangGraph router: natural language in, the right specialist agent out.

Why a graph instead of an if/else or a single mega-prompt
--------------------------------------------------------
The obvious alternatives both have real problems:

* **One big prompt with all the data.** Does not scale — the full problem
  history is far too large for a context window, and the model would be doing
  arithmetic it gets wrong.
* **A hand-rolled if/else.** Works, but the control flow ends up tangled with
  the business logic, and adding a node means editing the dispatcher, the
  state handling and the error paths at once.

LangGraph gives an explicit state machine: nodes declare what they read and
write, edges declare the routing, and the graph is a value you can inspect,
draw and extend. Concretely it buys three things here:

1. **Typed shared state.** Each node contributes its slice of ``TutorState``;
   nothing passes data by side effect.
2. **Observability.** Every node appends to ``trace``, and the API returns it,
   so a misrouted question is diagnosable from the response alone.
3. **Room to grow.** Adding a "explain this editorial" agent is one node plus
   one edge, with no changes to the existing ones.

The flow::

    question -> classify -> [weak_topics | recommend | search | profile | general]
                                     \\-> synthesize -> answer
"""

from __future__ import annotations

import json
import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.llm import classify_intent, synthesize
from app.agents.recommender import RecommenderAgent, extract_topics
from app.agents.weak_topics import WeakTopicAgent
from app.config import Settings, get_settings
from app.models import Recommendation, WeakTopic
from app.store.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class TutorState(TypedDict, total=False):
    """Shared state threaded through the graph.

    ``trace`` uses ``operator.add`` so each node *appends* rather than
    overwriting — that is what makes the execution path visible in the response.
    """

    question: str
    limit: int
    topics: list[str]

    intent: str
    intent_method: str
    trace: Annotated[list[str], operator.add]

    weak_topics: list[WeakTopic]
    recommendations: list[Recommendation]
    search_hits: list[dict]
    profile: dict[str, Any]

    answer: str
    answer_method: str


class TutorRouter:
    """Compiles and runs the routing graph."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        weak_topic_agent: WeakTopicAgent | None = None,
        recommender_agent: RecommenderAgent | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._weak_topics = weak_topic_agent or WeakTopicAgent(settings=self._settings)
        self._recommender = recommender_agent or RecommenderAgent(
            settings=self._settings, weak_topic_agent=self._weak_topics
        )
        self._vector_store = vector_store or get_vector_store()
        self._graph = self._build_graph()

    # -- graph construction ------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(TutorState)

        graph.add_node("classify", self._classify_node)
        graph.add_node("weak_topics", self._weak_topics_node)
        graph.add_node("recommend", self._recommend_node)
        graph.add_node("search", self._search_node)
        graph.add_node("profile", self._profile_node)
        graph.add_node("general", self._general_node)
        graph.add_node("synthesize", self._synthesize_node)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: state.get("intent", "general"),
            {
                "weak_topics": "weak_topics",
                "recommend": "recommend",
                "search": "search",
                "profile": "profile",
                "general": "general",
            },
        )
        for node in ("weak_topics", "recommend", "search", "profile", "general"):
            graph.add_edge(node, "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    # -- nodes -------------------------------------------------------------
    def _classify_node(self, state: TutorState) -> dict:
        question = state.get("question", "")
        intent, method = classify_intent(question)
        logger.info("routed question", extra={"intent": intent, "method": method})
        return {"intent": intent, "intent_method": method, "trace": ["classify"]}

    def _weak_topics_node(self, state: TutorState) -> dict:
        question = state.get("question", "")
        # "why am I weak at DP" names a topic; narrow the answer to it.
        mentioned = extract_topics(question)
        topic_filter = mentioned[0] if mentioned else None

        weak = self._weak_topics.run(
            limit=state.get("limit") or 5, topic_filter=topic_filter
        )
        return {"weak_topics": weak, "topics": mentioned, "trace": ["weak_topics"]}

    def _recommend_node(self, state: TutorState) -> dict:
        recommendations, weak = self._recommender.run(
            limit=state.get("limit") or 5,
            topics=extract_topics(state.get("question", "")) or None,
        )
        return {
            "recommendations": recommendations,
            "weak_topics": weak[:5],
            "trace": ["recommend"],
        }

    def _search_node(self, state: TutorState) -> dict:
        question = state.get("question", "")
        try:
            hits = self._vector_store.search(question, limit=state.get("limit") or 5)
        except Exception:  # noqa: BLE001 - an unbuilt index should not 500
            logger.exception("semantic search failed")
            hits = []
        return {"search_hits": hits, "trace": ["search"]}

    def _profile_node(self, state: TutorState) -> dict:
        return {"profile": self._weak_topics.profile(), "trace": ["profile"]}

    def _general_node(self, state: TutorState) -> dict:
        """Fall back to a profile-plus-weakness snapshot.

        A vague question ("help me get better") still deserves grounded advice,
        so we attach real data rather than letting the model improvise.
        """
        return {
            "profile": self._weak_topics.profile(),
            "weak_topics": self._weak_topics.run(limit=3),
            "trace": ["general"],
        }

    def _synthesize_node(self, state: TutorState) -> dict:
        """Phrase the answer, preferring the LLM and falling back to templates."""
        analysis = _analysis_payload(state)
        answer, method = synthesize(state.get("question", ""), analysis)
        if answer is None:
            answer = _template_answer(state)
            method = method if method.startswith("template") else "template"
        return {"answer": answer, "answer_method": method, "trace": ["synthesize"]}

    # -- entry point -------------------------------------------------------
    def ask(self, question: str, *, limit: int = 5) -> dict:
        """Run the graph over one question and return the final state."""
        final: TutorState = self._graph.invoke(
            {"question": question, "limit": limit, "trace": []}
        )
        return {
            "question": question,
            "intent": final.get("intent", "general"),
            "intent_method": final.get("intent_method", "keyword"),
            "answer": final.get("answer", ""),
            "answer_method": final.get("answer_method", "template"),
            "weak_topics": final.get("weak_topics", []),
            "recommendations": final.get("recommendations", []),
            "search_hits": final.get("search_hits", []),
            "profile": final.get("profile", {}),
            "trace": final.get("trace", []),
        }


def _analysis_payload(state: TutorState) -> str:
    """Serialise the computed facts for the synthesis prompt.

    Only what the nodes actually produced is included, keeping the prompt small
    and making it impossible for the model to cite a section that was never
    computed.
    """
    payload: dict[str, Any] = {"intent": state.get("intent")}

    weak = state.get("weak_topics") or []
    if weak:
        payload["weak_topics"] = [
            {
                "topic": topic.topic,
                "score": topic.score,
                "confidence": topic.confidence,
                "solved": topic.solved,
                "attempted_unsolved": topic.attempted_unsolved,
                "days_since_last_solve": topic.days_since_last_solve,
                "reasons": topic.reasons,
            }
            for topic in weak
        ]

    recommendations = state.get("recommendations") or []
    if recommendations:
        payload["recommendations"] = [
            {
                "name": rec.problem.name,
                "rating": rec.problem.rating,
                "url": rec.problem.url,
                "topics": rec.matched_topics,
                "why": rec.reason,
            }
            for rec in recommendations
        ]

    if state.get("search_hits"):
        payload["search_hits"] = state["search_hits"]
    if state.get("profile"):
        payload["profile"] = state["profile"]

    return json.dumps(payload, indent=2, default=str)


def _template_answer(state: TutorState) -> str:
    """Deterministic phrasing used whenever the LLM is unavailable."""
    intent = state.get("intent", "general")

    if intent == "recommend":
        return RecommenderAgent.template_answer(state.get("recommendations") or [])

    if intent == "weak_topics":
        topics = state.get("topics") or []
        return WeakTopicAgent.template_answer(
            state.get("weak_topics") or [], topic_filter=topics[0] if topics else None
        )

    if intent == "search":
        hits = state.get("search_hits") or []
        if not hits:
            return "No matching problems in your history yet. Run POST /sync to index it."
        lines = ["Closest matches in your solved history:"]
        lines.extend(
            f"{position}. {hit['name']} ({', '.join(hit['tags']) or 'untagged'}) "
            f"— similarity {hit['similarity']} — {hit['url']}"
            for position, hit in enumerate(hits, start=1)
        )
        return "\n".join(lines)

    profile = state.get("profile") or {}
    if not profile:
        return "No profile data yet. Run POST /sync to pull your submission history."

    lines = [
        f"You have solved {profile.get('total_solved', 0)} problems across "
        f"{profile.get('distinct_topics', 0)} topics "
        f"({profile.get('total_attempted_unsolved', 0)} attempted but unsolved).",
    ]
    if profile.get("max_rating_solved"):
        lines.append(
            f"Your hardest solve is rated {profile['max_rating_solved']}, "
            f"average {profile.get('avg_rating_solved')}."
        )
    weak = state.get("weak_topics") or []
    if weak:
        lines.append("Weakest topics: " + ", ".join(topic.topic for topic in weak) + ".")
    return " ".join(lines)


_router_singleton: TutorRouter | None = None


def get_router() -> TutorRouter:
    """Process-wide router. Compiling the graph once keeps request latency down."""
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = TutorRouter()
    return _router_singleton
