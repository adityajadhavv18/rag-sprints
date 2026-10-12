"""Evaluate Contextual RAG on the medical golden set (tracing on, LLM judge on), with deltas vs the naive baseline.

Two runs: contextual + hybrid search (arch "contextual") and contextual + dense search only ("contextual_dense").

Run from rag_sprints/:
    uv run week1/03_contextual_rag/run_eval.py            # both
    uv run week1/03_contextual_rag/run_eval.py hybrid     # one of: hybrid, dense
"""

import sys

from contextualize import GOLDEN_SET
from eval_harness import evaluate, settings
from pipeline import contextual_pipeline, dense_search

RUNS = {
    "hybrid": ("contextual", lambda q: contextual_pipeline(q)),
    "dense": ("contextual_dense", lambda q: contextual_pipeline(q, search=dense_search)),
}

for name in sys.argv[1:] or RUNS:
    arch, fn = RUNS[name]
    print(f"\n=== {arch}: 370-token chunks + notes, {name} search, top 5 original chunks to the LLM ===")
    run = evaluate(fn, arch=arch, golden_set=GOLDEN_SET)
    print(f"Compare in detail:  uv run compare-runs {arch}_{run.started_at:%Y%m%d-%H%M%S}.json")
    print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{arch}")
