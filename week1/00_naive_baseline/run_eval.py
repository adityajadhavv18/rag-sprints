"""Evaluate the naive baseline on the full golden set (tracing on, LLM judge on).

Run from rag_sprints/:
    uv run week1/00_naive_baseline/run_eval.py                  # evaluate + save results/naive_<time>.json
    uv run week1/00_naive_baseline/run_eval.py --set-baseline   # same, then promote it to baseline_v1.json
"""

import sys

from eval_harness import evaluate, set_baseline, settings
from pipeline import naive_pipeline

run = evaluate(naive_pipeline, arch="naive")
result_file = f"naive_{run.started_at:%Y%m%d-%H%M%S}.json"

if "--set-baseline" in sys.argv:
    set_baseline(result_file)
else:
    print(f"To make this run the baseline:  uv run week1/00_naive_baseline/run_eval.py --set-baseline")
    print(f"(or promote this exact run: set_baseline('{result_file}'))")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:naive")
