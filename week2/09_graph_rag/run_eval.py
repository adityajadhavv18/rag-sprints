"""Evaluate GraphRAG lite on the medical golden set (LLM judge on), two runs:

  graph_rag   Contextual hybrid search + graph walk, graph chunks first (the harness entry point)
  graph_only  the walk alone: a question naming no entity the graph knows gets no chunks and is refused in code

Both compared with Contextual (sprint 03). Compare with the rest afterwards:
  uv run compare-runs <graph_rag run> agentic_20261017-194940.json   (or adaptive_..., corrective_...)
Indexing cost (one-off, graph extraction) is in index_cost.json, kept apart from the per-query cost.

Run from rag_sprints/:  uv run week2/09_graph_rag/run_eval.py   (~$0.06 incl. judge)
"""

import json
from pathlib import Path

from eval_harness import evaluate, settings
from pipeline import graph_only_pipeline, graph_rag_pipeline

GOLDEN_SET = "golden_medical_v1.jsonl"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "eval_harness" / "results"
CONTEXTUAL_RUN = RESULTS / "contextual_20261012-211033.json"

runs = {}
import sys
ONLY = sys.argv[1:]  # e.g. `run_eval.py graph_only` re-runs just that one
for arch, fn in (("graph_rag", graph_rag_pipeline), ("graph_only", graph_only_pipeline)):
    if ONLY and arch not in ONLY:
        continue
    print(f"\n=== {arch} ===")
    runs[arch] = evaluate(fn, arch=arch, golden_set=GOLDEN_SET, compare_to=CONTEXTUAL_RUN)

index = json.loads((HERE / "index_cost.json").read_text())
print(f"\nIndexing (one-off): {index['llm_calls']} LLM calls, ${index['cost_usd']}, "
      f"{index['nodes']} nodes, {index['edges']} edges")
for arch, run in runs.items():
    print(f"Compare in detail:  uv run compare-runs {arch}_{run.started_at:%Y%m%d-%H%M%S}.json "
          f"{CONTEXTUAL_RUN.name}")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:graph_rag / arch:graph_only")
