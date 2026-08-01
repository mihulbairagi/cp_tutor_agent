"""LLM access, and the deterministic fallback for when there is no API key.

Design rule for the whole project: **the LLM classifies and phrases; it never
computes.** Weakness scores, ratings and problem selection are all produced by
``app.services``. That keeps the numbers reproducible and testable, and confines
the model to the two jobs it is genuinely better at than code — understanding a
vague question, and writing a readable answer.

Everything here degrades gracefully. With no ``ANTHROPIC_API_KEY``:

* intent classification falls back to a keyword classifier
* answer synthesis falls back to templates over the same structured data

The result is that CI, the evaluation harness and offline demos all run with no
key and no network, while the hosted app gets natural language. The fallback is
not a stub — it is a real, tested code path.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

#: Every intent the router can dispatch to.
INTENTS = ("weak_topics", "recommend", "search", "profile", "general")

CLASSIFIER_SYSTEM_PROMPT = """\
You classify a competitive-programming student's question into exactly one intent.

Intents:
- weak_topics: asking which topics they are weak at, or WHY they are weak at something.
  ("why am I bad at DP", "what are my weakest areas", "where should I improve")
- recommend: asking what to solve or practise next.
  ("what should I practice today", "give me 5 graph problems", "next problem")
- search: asking to find or look up problems similar to something in their history.
  ("show me problems like the tree ones I failed", "find my shortest path solves")
- profile: asking about their stats, totals, rating or progress.
  ("how many problems have I solved", "what's my rating", "summarise my progress")
- general: anything else, including advice questions with no data lookup.

Reply with the intent name and nothing else.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a competitive-programming coach. You will be given a student's question
and a JSON block of ANALYSIS RESULTS computed by a deterministic scoring engine.

Rules:
- Use ONLY the numbers in the analysis. Never invent problems, ratings or counts.
- Be direct and specific. Reference the actual topic names and problem names given.
- Explain the WHY using the reasons provided, in plain language.
- Keep it under 200 words. Use short paragraphs or a tight list.
- Address the student as "you".
"""

# Ordered most-specific first: a question like "what should I practise to fix my
# weak DP" is a recommendation request, not a weakness lookup, so `recommend`
# is checked before `weak_topics`.
_KEYWORD_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "recommend",
        re.compile(
            # `problems?` rather than `problem`: the group is closed by a `\b`,
            # and "problem" followed by "s" is not a word boundary, so a bare
            # `problem` here would silently fail to match "give me 3 problems".
            r"\b(recommend|suggest|what should i (do|solve|practi[cs]e)|practi[cs]e today|"
            r"next problem|give me .*problems?|problems? to solve|work on next|todo|"
            r"what to solve)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "search",
        re.compile(
            r"\b(similar to|problems like|like the .*(i|I) (solved|failed|attempted)|"
            r"find (me )?problems|search|show me .*problems (about|on|with))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weak_topics",
        re.compile(
            r"\b(weak|weakest|bad at|worst|struggl\w*|why am i|where (should|do) i improve|"
            r"improve|blind spot|gaps?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "profile",
        re.compile(
            r"\b(how many|my (rating|stats|progress|profile|history)|total solved|"
            r"summar(y|ise|ize)|overview|statistics)\b",
            re.IGNORECASE,
        ),
    ),
]


def keyword_intent(question: str) -> str:
    """Classify intent with regexes. Fast, free, deterministic, and tested."""
    text = question or ""
    for intent, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            return intent
    return "general"


@lru_cache(maxsize=1)
def get_llm() -> Any | None:
    """Return a configured chat model, or ``None`` when no key is available.

    Cached because constructing the client sets up an HTTP pool we want to reuse
    across requests.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        logger.info("no ANTHROPIC_API_KEY set; using deterministic router and templates")
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        logger.warning("langchain-anthropic is not installed; falling back to templates")
        return None

    logger.info("LLM enabled", extra={"model": settings.llm_model})
    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key,
    )


def classify_intent(question: str) -> tuple[str, str]:
    """Classify a question. Returns ``(intent, method)``.

    ``method`` is reported in the API response so it is always obvious whether an
    answer came from the model or the fallback — useful when debugging a
    misrouted question, and honest about what the system is doing.
    """
    llm = get_llm()
    if llm is None:
        return keyword_intent(question), "keyword"

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        raw = str(getattr(response, "content", "")).strip().lower()
        # Validate against the allowed set rather than trusting free text: a
        # hallucinated intent would crash the graph's conditional edge.
        for intent in INTENTS:
            if intent in raw:
                return intent, "llm"
        logger.warning("unrecognised intent from LLM; using keywords", extra={"raw": raw[:80]})
    except Exception:  # noqa: BLE001 - a model outage must not break routing
        logger.exception("intent classification failed; using keywords")

    return keyword_intent(question), "keyword-fallback"


def synthesize(question: str, analysis: str) -> tuple[str | None, str]:
    """Ask the model to phrase an answer over the computed analysis.

    Returns ``(answer, method)``; ``answer`` is ``None`` when the caller should
    use its template instead.
    """
    llm = get_llm()
    if llm is None:
        return None, "template"

    try:
        response = llm.invoke(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nANALYSIS RESULTS:\n{analysis}",
                },
            ]
        )
        text = str(getattr(response, "content", "")).strip()
        return (text, "llm") if text else (None, "template")
    except Exception:  # noqa: BLE001 - degrade to the template, never 500
        logger.exception("answer synthesis failed; using template")
        return None, "template-fallback"
