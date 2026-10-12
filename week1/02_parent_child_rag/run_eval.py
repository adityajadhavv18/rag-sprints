"""Evaluate Parent-Child RAG on the medical golden set (tracing on, LLM judge on), with deltas vs the naive baseline.

Run from rag_sprints/:
    uv run week1/02_parent_child_rag/run_eval.py
"""

from eval_harness import evaluate, settings
from ingest import chunking
from pipeline import CHILD_DEPTH, N_PARENTS, parent_child_pipeline

print(f"Parent-Child config: dense search over {CHILD_DEPTH} children -> top {N_PARENTS} distinct parents to the LLM")
run = evaluate(parent_child_pipeline, arch="parent_child", golden_set=chunking.GOLDEN_SET)
print(f"Compare in detail:  uv run compare-runs parent_child_{run.started_at:%Y%m%d-%H%M%S}.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:parent_child")
