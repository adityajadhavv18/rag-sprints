"""Evaluate Corrective RAG on the medical golden set (tracing on, LLM judge on), compared with Contextual hybrid.

One run, arch "corrective": search -> grade -> refine and retry (max 3 rounds) -> answer from the kept chunks.
Latency and cost include every grader and refine call. Web-search events go to web_search_events.jsonl.

Run from rag_sprints/:  uv run week2/06_corrective_rag/run_eval.py   (~$0.10 incl. judge)
"""

from collections import Counter
from pathlib import Path

from eval_harness import evaluate, settings
from pipeline import LOOP_LOG, corrective_pipeline

ARCH = "corrective"
GOLDEN_SET = "golden_medical_v1.jsonl"
CONTEXTUAL_RUN = Path(__file__).resolve().parents[3] / "eval_harness" / "results" / "contextual_20261012-211033.json"

print(f"\n=== {ARCH}: contextual hybrid -> grader -> refine + retry (max 3 rounds) -> answer from kept chunks ===")
run = evaluate(corrective_pipeline, arch=ARCH, golden_set=GOLDEN_SET, compare_to=CONTEXTUAL_RUN)
rounds = Counter(r for r, _ in LOOP_LOG.values())
print(f"Rounds used: {dict(sorted(rounds.items()))} | web-search events: {sum(w for _, w in LOOP_LOG.values())}")
print(f"Compare in detail:  uv run compare-runs {ARCH}_{run.started_at:%Y%m%d-%H%M%S}.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{ARCH}")
