"""Evaluate RAG Fusion on the medical golden set (tracing on, LLM judge on), with deltas vs the naive baseline.

One run, arch "rag_fusion": live rewrite (4 rephrasings) -> 10 lists on the contextual index -> RRF -> top 5
original chunks -> naive's generate(). Latency and cost include the rewrite call.

Run from rag_sprints/:  uv run week1/04_rag_fusion/run_eval.py
"""

from eval_harness import evaluate, settings
from pipeline import fusion_pipeline
from rewrite import GOLDEN_SET

ARCH = "rag_fusion"
print(f"\n=== {ARCH}: question + 4 rephrasings, dense + BM25 each on the contextual index, RRF, top 5 to the LLM ===")
run = evaluate(fusion_pipeline, arch=ARCH, golden_set=GOLDEN_SET)
print(f"Compare in detail:  uv run compare-runs {ARCH}_{run.started_at:%Y%m%d-%H%M%S}.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{ARCH}")
