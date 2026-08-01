"""FastAPI application: HTTP surface for the tutor.

Route design notes
------------------
There are two ways in, on purpose:

* **Structured endpoints** (``/weak-topics``, ``/recommendations``, ``/search``,
  ``/profile``) return typed JSON and skip the LLM entirely. They are the
  contract a frontend or a cron job should build against — fast, deterministic,
  and free.
* **``POST /ask``** is the natural-language front door. It runs the LangGraph
  router, and its response embeds the *same* structured objects the endpoints
  above return, alongside the prose. Nothing is only available through the model.

Every route declares a ``response_model``, a summary, a description and an
OpenAPI tag, so the generated ``/docs`` is genuinely usable rather than a wall
of untyped 200s.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from app.agents.recommender import RecommenderAgent
from app.agents.router import TutorRouter, get_router
from app.agents.weak_topics import WeakTopicAgent
from app.config import Settings, get_settings
from app.logging_config import setup_logging
from app.models import WeakTopic
from app.scheduler import SyncScheduler
from app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    ProfileResponse,
    RecommendationsResponse,
    SearchResponse,
    SyncResponse,
    WeakTopicsResponse,
)
from app.services.analysis import summarise_profile
from app.services.ingestion import IngestionService, get_ingestion_service
from app.store.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

DESCRIPTION = """\
A multi-agent coach for competitive programming.

It ingests your **Codeforces** and **LeetCode** history, scores your weakest
topics with a deterministic, explainable model, and recommends real unsolved
problems to practise next.

### How it works

1. `POST /sync` pulls your submission history from both platforms and indexes it.
2. `GET /weak-topics` ranks your weaknesses from four signals — coverage,
   staleness, difficulty gap and struggle rate — and shows the reasoning.
3. `GET /recommendations` filters the live Codeforces problemset down to
   unsolved problems that hit those weak topics inside a sensible rating band.
4. `POST /ask` routes a plain-English question to whichever of those agents fits.

### A note on the LLM

Scores and problem selection are computed in code, never by the model, so
results are reproducible and unit-tested. The model only classifies the intent
of your question and phrases the reply. Without an `ANTHROPIC_API_KEY` the API
still works end to end using a keyword router and template answers — the
`intent_method` and `answer_method` fields tell you which path ran.
"""

TAGS_METADATA = [
    {"name": "System", "description": "Health and readiness."},
    {"name": "Data", "description": "Ingesting and inspecting your problem history."},
    {"name": "Coaching", "description": "Analysis, recommendations and search."},
    {"name": "Agent", "description": "The natural-language entry point."},
]

scheduler = SyncScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, prepare storage, and run the background scheduler."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    settings.ensure_data_dirs()

    logger.info(
        "starting %s",
        settings.app_name,
        extra={
            "version": settings.app_version,
            "codeforces_handle": settings.codeforces_handle or "<unset>",
            "leetcode_username": settings.leetcode_username or "<unset>",
            "llm_enabled": settings.llm_enabled,
            "chroma_mode": settings.chroma_mode,
        },
    )
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown()
        logger.info("shutdown complete")


app = FastAPI(
    title="CP Tutor Agent",
    description=DESCRIPTION,
    version=get_settings().app_version,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    contact={"name": "CP Tutor Agent", "url": "https://github.com/mihulbairagi/cp-tutor-agent"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_weak_topic_agent() -> WeakTopicAgent:
    return WeakTopicAgent()


def get_recommender_agent() -> RecommenderAgent:
    return RecommenderAgent()


def require_synced_data(
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> IngestionService:
    """Guard endpoints that are meaningless without a synced profile.

    Returns 409 rather than an empty 200, so a client can tell "you have no data"
    apart from "you have no weaknesses".
    """
    if not ingestion.has_data():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No profile data yet. Call POST /sync first.",
        )
    return ingestion


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service health and configuration",
    description=(
        "Reports liveness plus the configuration that actually affects behaviour: "
        "which handles are being tracked, whether an LLM key is present, whether "
        "LeetCode full history is unlocked, and how many problems are indexed.\n\n"
        "Returns `degraded` when the service is up but has no synced data yet."
    ),
)
async def health(
    settings: Settings = Depends(get_settings),
    ingestion: IngestionService = Depends(get_ingestion_service),
    store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    has_data = ingestion.has_data()
    return HealthResponse(
        status="ok" if has_data else "degraded",
        version=settings.app_version,
        codeforces_handle=settings.codeforces_handle,
        leetcode_username=settings.leetcode_username,
        llm_enabled=settings.llm_enabled,
        leetcode_full_history=settings.leetcode_authenticated,
        indexed_problems=store.count() if has_data else 0,
        last_sync_at=ingestion.get_last_sync(),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@app.post(
    "/sync",
    response_model=SyncResponse,
    tags=["Data"],
    summary="Pull and index your problem history",
    description=(
        "Fetches submissions from Codeforces and LeetCode, merges them into a single "
        "profile, and indexes them for semantic search.\n\n"
        "Responses are cached, so calling this repeatedly is cheap; pass `force=true` to "
        "bypass the cache and hit the upstream APIs directly. Each source reports its own "
        "status, so a LeetCode outage still yields a successful Codeforces sync.\n\n"
        "This also runs automatically on the interval set by `SYNC_INTERVAL_MINUTES`."
    ),
)
async def sync(
    force: bool = Query(default=False, description="Bypass the cache and refetch from upstream."),
    reset_index: bool = Query(default=False, description="Drop the vector index and rebuild it."),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> SyncResponse:
    result = ingestion.sync(force_refresh=force, reset_index=reset_index)
    return SyncResponse(result=result)


@app.get(
    "/profile",
    response_model=ProfileResponse,
    tags=["Data"],
    summary="Aggregate statistics across both platforms",
    description=(
        "Totals, per-source and per-difficulty breakdowns, rating summary, and when you "
        "last solved anything. Computed from the synced profile — no upstream calls."
    ),
    responses={409: {"model": ErrorResponse, "description": "Nothing synced yet."}},
)
async def profile(
    settings: Settings = Depends(get_settings),
    ingestion: IngestionService = Depends(require_synced_data),
) -> ProfileResponse:
    records = ingestion.get_records()
    rating = ingestion.get_user_rating()
    return ProfileResponse(
        codeforces_handle=settings.codeforces_handle,
        leetcode_username=settings.leetcode_username,
        current_rating=rating.rating,
        max_rating=rating.max_rating,
        rank=rating.rank,
        last_sync_at=ingestion.get_last_sync(),
        **summarise_profile(records),
    )


# ---------------------------------------------------------------------------
# Coaching
# ---------------------------------------------------------------------------
@app.get(
    "/weak-topics",
    response_model=WeakTopicsResponse,
    tags=["Coaching"],
    summary="Rank your weakest topics, with reasoning",
    description=(
        "Scores every topic you have touched on four signals and returns them weakest "
        "first:\n\n"
        "- **coverage** — how far below a healthy solve count the topic sits\n"
        "- **staleness** — how long since you last solved one\n"
        "- **difficulty** — how far the topic's hardest solve trails your overall ceiling\n"
        "- **struggle** — share of problems you attempted and never solved\n\n"
        "Each score is damped by a confidence term derived from sample size, so a topic "
        "with two data points cannot outrank one with thirty. Every result carries the "
        "raw signal values and plain-English `reasons`.\n\n"
        "Set `include_blind_spots=true` to also surface common topics you have **never** "
        "attempted, which by definition have no submission history to score."
    ),
    responses={409: {"model": ErrorResponse, "description": "Nothing synced yet."}},
)
async def weak_topics(
    limit: int = Query(default=5, ge=1, le=50),
    include_blind_spots: bool = Query(
        default=True, description="Include common topics you have never attempted."
    ),
    topic: str | None = Query(
        default=None, description="Narrow to one topic, e.g. `dynamic programming`."
    ),
    ingestion: IngestionService = Depends(require_synced_data),
    agent: WeakTopicAgent = Depends(get_weak_topic_agent),
) -> WeakTopicsResponse:
    topics: list[WeakTopic] = agent.run(
        limit=limit, include_blind_spots=include_blind_spots, topic_filter=topic
    )
    return WeakTopicsResponse(topics=topics, analysed_records=len(ingestion.get_records()))


@app.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    tags=["Coaching"],
    summary="Real unsolved problems to practise next",
    description=(
        "Filters the live Codeforces problemset down to problems you have **not** solved, "
        "that carry at least one of your weak topics, and that fall inside a rating band "
        "around your current rating (default −100 to +300 — practice should be "
        "uncomfortable, not impossible).\n\n"
        "Results are capped per topic so a session stays varied instead of returning ten "
        "near-identical DP problems. Problems you attempted and failed stay eligible; "
        "they are prime practice material.\n\n"
        "Pass `topics` to override the automatic weakness ranking, e.g. "
        "`?topics=graphs&topics=segment tree`."
    ),
    responses={409: {"model": ErrorResponse, "description": "Nothing synced yet."}},
)
async def recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    topics: list[str] | None = Query(default=None, description="Force specific topics."),
    rating: int | None = Query(
        default=None, description="Override your Codeforces rating when sizing the band."
    ),
    settings: Settings = Depends(get_settings),
    agent: RecommenderAgent = Depends(get_recommender_agent),
    ingestion: IngestionService = Depends(require_synced_data),
) -> RecommendationsResponse:
    results, weak = agent.run(limit=limit, topics=topics, rating_override=rating)

    effective = rating or ingestion.get_user_rating().rating
    effective = effective or settings.default_rating_when_unrated
    band = f"{effective - settings.rating_band_below}-{effective + settings.rating_band_above}"

    return RecommendationsResponse(
        recommendations=results,
        based_on_topics=[topic.topic for topic in weak],
        rating_band=band,
    )


@app.get(
    "/search",
    response_model=SearchResponse,
    tags=["Coaching"],
    summary="Semantic search over your own history",
    description=(
        "Vector search across the problems you have attempted, embedded as "
        "`name + topics + difficulty`.\n\n"
        'This matches on *meaning*, not keywords — a query for "graph traversal" will '
        "surface problems tagged `dfs and similar` and `trees` even though the words do "
        "not appear. Filter with `solved=false` to look specifically at what you got stuck on."
    ),
)
async def search(
    q: str = Query(min_length=1, description="Free-text query."),
    limit: int = Query(default=5, ge=1, le=25),
    source: str | None = Query(default=None, description="`codeforces` or `leetcode`."),
    solved: bool | None = Query(default=None, description="Filter by solved status."),
    store: VectorStore = Depends(get_vector_store),
) -> SearchResponse:
    hits = store.search(q, limit=limit, source=source, solved=solved)
    return SearchResponse(query=q, hits=hits)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Agent"],
    summary="Ask the tutor anything, in plain English",
    description=(
        "Routes your question through a LangGraph state machine to the right specialist "
        "agent, then answers.\n\n"
        "Recognised intents: `weak_topics`, `recommend`, `search`, `profile`, `general`.\n\n"
        "The response includes the same structured objects the dedicated endpoints return, "
        "so a client never has to parse the prose. `trace` lists the graph nodes that ran, "
        "and `intent_method` / `answer_method` tell you whether the LLM or the "
        "deterministic fallback produced each part."
    ),
)
async def ask(
    request: AskRequest,
    router: TutorRouter = Depends(get_router),
    _: IngestionService = Depends(require_synced_data),
) -> AskResponse:
    result = router.ask(request.question, limit=request.limit)
    return AskResponse(**result)
