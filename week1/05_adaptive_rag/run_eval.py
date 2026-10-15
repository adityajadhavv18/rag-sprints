"""Evaluate Adaptive RAG: the judged eval on the medical golden set, then the 10 extra questions end to end.

Part 1, arch "adaptive": router -> no_rag / single / multi on the 30 golden questions (LLM judge on, ~$0.08).
        The golden set has no route labels, so this checks that routing loses nothing against Contextual and Fusion.
Part 2, the 10 extra questions (5 no-search, 5 chains): route, answer and the expected answer, printed for review.
        Not judged: they are not in the golden set, and the harness's contract stays untouched.

Run from rag_sprints/:  uv run week1/05_adaptive_rag/run_eval.py
"""

import json
import time
from collections import Counter
from pathlib import Path

from eval_harness import evaluate, settings
from pipeline import ROUTE_LOG, adaptive_pipeline

ARCH = "adaptive"
print(f"\n=== {ARCH}: gpt-4o-mini router -> no_rag | single (contextual hybrid) | multi (search-read loop) ===")
run = evaluate(adaptive_pipeline, arch=ARCH, golden_set="golden_medical_v1.jsonl")
print(f"Routes taken on the golden set: {dict(Counter(ROUTE_LOG.values()))}")
print(f"Compare in detail:  uv run compare-runs {ARCH}_{run.started_at:%Y%m%d-%H%M%S}.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{ARCH}")

print(f"\n=== The 10 extra questions, end to end ===")
for line in Path(__file__).with_name("questions_extra.jsonl").read_text().splitlines():
    q = json.loads(line)
    t0 = time.perf_counter()
    out = adaptive_pipeline(q["question"])
    print(f"\n{q['id']}  expected {q['route']}, took {ROUTE_LOG[q['question']]}  "
          f"({time.perf_counter() - t0:.1f}s, {len(out.contexts)} chunks)  {q['question']}")
    print(f"  ANSWER  {out.answer}")
    print(f"  TRUTH   {q['ground_truth']}")
