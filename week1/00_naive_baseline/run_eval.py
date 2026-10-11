"""Evaluate the naive baseline on the full golden set (tracing on, LLM judge on).

Run from rag_sprints/:
    uv run week1/00_naive_baseline/run_eval.py                  # evaluate + save results/naive_<time>.json
    uv run week1/00_naive_baseline/run_eval.py --set-baseline   # same, then promote it to baseline_v1.json
    RAG_CORPUS=medical uv run week1/00_naive_baseline/run_eval.py   # medical corpus + golden_medical_v1.jsonl
"""

import sys

from eval_harness import evaluate, set_baseline, settings
from ingest import CORPUS, GOLDEN_SET
from pipeline import naive_pipeline

run = evaluate(naive_pipeline, arch="naive", golden_set=GOLDEN_SET)
result_file = f"naive_{run.started_at:%Y%m%d-%H%M%S}.json"
prefix = f"RAG_CORPUS={CORPUS} " if CORPUS != "10k" else ""

if "--set-baseline" in sys.argv:
    set_baseline(result_file)
else:
    print(f"To make this run the baseline:  {prefix}uv run week1/00_naive_baseline/run_eval.py --set-baseline")
    print(f"(or promote this exact run: set_baseline('{result_file}'))")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:naive")
