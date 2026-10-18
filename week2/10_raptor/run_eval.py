"""Evaluate RAPTOR lite on the medical golden set (LLM judge on), compared with Contextual (sprint 03).

The golden set is fact lookups: this run answers "does the tree HURT them?". Whether it helps on broad questions is
thematic.py's job (questions_thematic.jsonl). Compare with GraphRAG afterwards:
  uv run compare-runs <raptor run> graph_rag_20261018-144833.json
Indexing cost (one-off, 22 summary calls) is in index_cost.json, kept apart from the per-query cost.

Run from rag_sprints/:  uv run week2/10_raptor/run_eval.py   (~$0.06 incl. judge)
"""

import json
from pathlib import Path

from eval_harness import evaluate, settings
from pipeline import raptor_pipeline

GOLDEN_SET = "golden_medical_v1.jsonl"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "eval_harness" / "results"
CONTEXTUAL_RUN = RESULTS / "contextual_20261012-211033.json"

run = evaluate(raptor_pipeline, arch="raptor", golden_set=GOLDEN_SET, compare_to=CONTEXTUAL_RUN)

index = json.loads((HERE / "index_cost.json").read_text())
print(f"\nIndexing (one-off): {index['llm_calls']} summary calls, ${index['cost_usd']}, "
      f"{index['level1_summaries']} + {index['level2_summaries']} summaries over {index['leaves']} leaves")
print(f"Compare in detail:  uv run compare-runs raptor_{run.started_at:%Y%m%d-%H%M%S}.json {CONTEXTUAL_RUN.name}")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:raptor")
