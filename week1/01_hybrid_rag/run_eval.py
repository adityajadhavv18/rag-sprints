"""Evaluate Hybrid RAG on the full golden set (tracing on, LLM judge on), with deltas vs the naive baseline.

Run from rag_sprints/:
    RAG_CORPUS=medical uv run week1/01_hybrid_rag/run_eval.py   # medical corpus + golden_medical_v1.jsonl
    uv run week1/01_hybrid_rag/run_eval.py                      # 10-K corpus + golden_v1.jsonl
"""

from eval_harness import evaluate, settings
from ingest import GOLDEN_SET
from pipeline import CANDIDATES, RRF_K, TOP_K, hybrid_pipeline

print(f"Hybrid config: top {TOP_K} to the LLM, {CANDIDATES} candidates per path, RRF k={RRF_K}, equal weights")
run = evaluate(hybrid_pipeline, arch="hybrid", golden_set=GOLDEN_SET)
print(f"Compare in detail:  uv run compare-runs hybrid_{run.started_at:%Y%m%d-%H%M%S}.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:hybrid")
