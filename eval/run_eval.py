"""Run the evaluation set and print a pass/fail table.

    python -m eval.run_eval
    python -m eval.run_eval --verbose      # show the answer text for each case
    python -m eval.run_eval --case rec-basic

Exits non-zero if anything fails, so it can be wired into CI as a behavioural
regression gate alongside the unit tests. The unit tests assert that each piece
works; this asserts that the *assembled system* answers real questions correctly.

Nothing here touches the network: the leaf services are the stubs in
:mod:`eval.fixtures`, and with no API key the router uses its deterministic
classifier and templates. Which means the harness is also a demo you can run
with nothing installed but the requirements.
"""

from __future__ import annotations

import argparse
import sys
import time

from app.agents.recommender import RecommenderAgent
from app.agents.router import TutorRouter
from app.agents.weak_topics import WeakTopicAgent
from app.config import get_settings
from eval.cases import CASES, EvalCase
from eval.fixtures import StubCodeforces, StubIngestion, StubVectorStore, solved_ids

#: A question should be answered well inside a second on the deterministic path.
LATENCY_BUDGET_MS = 1000.0

PASS = "PASS"
FAIL = "FAIL"


def build_router() -> TutorRouter:
    """Assemble the real graph over stubbed leaf services."""
    settings = get_settings()
    ingestion = StubIngestion()
    codeforces = StubCodeforces()

    weak_topics = WeakTopicAgent(settings=settings, ingestion=ingestion, codeforces=codeforces)
    recommender = RecommenderAgent(
        settings=settings,
        ingestion=ingestion,
        codeforces=codeforces,
        weak_topic_agent=weak_topics,
    )
    return TutorRouter(
        settings=settings,
        weak_topic_agent=weak_topics,
        recommender_agent=recommender,
        vector_store=StubVectorStore(),
    )


def _invariants(case: EvalCase, result: dict, elapsed_ms: float) -> list[tuple[str, bool]]:
    """Checks applied to every case, whatever it asked for."""
    trace = result.get("trace", [])
    recommended = {rec.problem.problem_id for rec in result.get("recommendations", [])}
    return [
        (f"intent is '{case.expected_intent}'", result.get("intent") == case.expected_intent),
        ("produced an answer", bool(result.get("answer", "").strip())),
        (
            "graph ran classify -> ... -> synthesize",
            len(trace) == 3 and trace[0] == "classify" and trace[-1] == "synthesize",
        ),
        ("never recommends a solved problem", not (recommended & solved_ids())),
        (f"under {LATENCY_BUDGET_MS:.0f}ms", elapsed_ms < LATENCY_BUDGET_MS),
    ]


def run_case(router: TutorRouter, case: EvalCase) -> dict:
    started = time.perf_counter()
    try:
        result = router.ask(case.question, limit=case.limit)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, not a crashed run
        return {
            "case": case,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "failures": [f"raised {type(exc).__name__}: {exc}"],
            "result": {},
        }

    elapsed_ms = (time.perf_counter() - started) * 1000
    failures = [label for label, ok in _invariants(case, result, elapsed_ms) if not ok]

    for label, predicate in case.checks:
        try:
            if not predicate(result):
                failures.append(label)
        except Exception as exc:  # noqa: BLE001 - a check that explodes has failed
            failures.append(f"{label} (raised {type(exc).__name__})")

    return {"case": case, "elapsed_ms": elapsed_ms, "failures": failures, "result": result}


def _print_table(outcomes: list[dict], verbose: bool) -> None:
    id_width = max(len(outcome["case"].id) for outcome in outcomes)
    question_width = min(46, max(len(outcome["case"].question) for outcome in outcomes))

    header = (
        f"{'CASE':<{id_width}}  {'QUESTION':<{question_width}}  {'INTENT':<12}  {'ms':>7}  RESULT"
    )
    print(header)
    print("-" * len(header))

    for outcome in outcomes:
        case = outcome["case"]
        question = case.question
        if len(question) > question_width:
            question = question[: question_width - 1] + "…"
        status = FAIL if outcome["failures"] else PASS
        intent = outcome["result"].get("intent", "-")
        print(
            f"{case.id:<{id_width}}  {question:<{question_width}}  "
            f"{intent:<12}  {outcome['elapsed_ms']:>7.1f}  {status}"
        )
        for failure in outcome["failures"]:
            print(f"{'':<{id_width}}  ↳ {failure}")
        if verbose and outcome["result"].get("answer"):
            for line in outcome["result"]["answer"].splitlines():
                print(f"{'':<{id_width}}  │ {line}")


def main(argv: list[str] | None = None) -> int:
    # The templates and this table both use box-drawing characters and em
    # dashes; a Windows console defaults to cp1252 and would raise on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate the tutor's question answering.")
    parser.add_argument("--case", help="Run a single case by id.")
    parser.add_argument("--verbose", action="store_true", help="Print each generated answer.")
    args = parser.parse_args(argv)

    cases = CASES
    if args.case:
        cases = [case for case in CASES if case.id == args.case]
        if not cases:
            print(f"No such case: {args.case}", file=sys.stderr)
            print("Available: " + ", ".join(case.id for case in CASES), file=sys.stderr)
            return 2

    settings = get_settings()
    print(
        f"CP Tutor Agent — evaluation ({len(cases)} cases)\n"
        f"intent routing: {'LLM' if settings.llm_enabled else 'keyword (no API key)'}\n"
    )

    router = build_router()
    outcomes = [run_case(router, case) for case in cases]
    _print_table(outcomes, args.verbose)

    passed = sum(1 for outcome in outcomes if not outcome["failures"])
    total = len(outcomes)
    latencies = sorted(outcome["elapsed_ms"] for outcome in outcomes)
    median = latencies[len(latencies) // 2]

    print()
    print(f"{passed}/{total} passed · median {median:.1f}ms · slowest {latencies[-1]:.1f}ms")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
