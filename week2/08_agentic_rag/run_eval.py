"""Evaluate Agentic RAG: the judged eval on the medical golden set, then the 5 chain questions end to end.

Part 1, arch "agentic": the agent loop on the 30 golden questions (LLM judge on), compared with Corrective.
        Compare with Adaptive too:  uv run compare-runs <this run> adaptive_20261015-134945.json
Part 2, the 5 chain questions from sprint 05: answer, rounds, and which hops' evidence reached the LLM.
        Not judged: they are not in the golden set, and the harness's contract stays untouched.

Run from rag_sprints/:  uv run week2/08_agentic_rag/run_eval.py   (~$0.08 incl. judge)
"""

import json
import time
from collections import Counter
from pathlib import Path

from agent import EXTRA, hop_found
from eval_harness import evaluate, settings
from pipeline import LOOP_LOG, agentic_pipeline

ARCH = "agentic"
GOLDEN_SET = "golden_medical_v1.jsonl"
RESULTS = Path(__file__).resolve().parents[3] / "eval_harness" / "results"
CORRECTIVE_RUN = RESULTS / "corrective_20261015-202424.json"

print(f"\n=== {ARCH}: gpt-4o-mini + search / lookup_entry tools, max 5 rounds, the agent writes the answer ===")
run = evaluate(agentic_pipeline, arch=ARCH, golden_set=GOLDEN_SET, compare_to=CORRECTIVE_RUN)
rounds = Counter(r for r, _, _ in LOOP_LOG.values())
tools = sum((t for _, t, _ in LOOP_LOG.values()), Counter())
print(f"Rounds used: {dict(sorted(rounds.items()))} | tool calls: {dict(tools)} | "
      f"hit the cap: {sum(c for _, _, c in LOOP_LOG.values())}")
print(f"Compare in detail:  uv run compare-runs {ARCH}_{run.started_at:%Y%m%d-%H%M%S}.json adaptive_20261015-134945.json")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{ARCH}")

print("\n=== The 5 chain questions, end to end ===")
for line in EXTRA.read_text().splitlines():
    q = json.loads(line)
    if q["route"] != "multi":
        continue
    t0 = time.perf_counter()
    out = agentic_pipeline(q["question"])
    secs = time.perf_counter() - t0
    r, t, cap = LOOP_LOG[q["question"]]
    hops = ["#" + str(p) if (p := hop_found(h["evidence"], out.contexts)) else "MISSED" for h in q["hops"]]
    print(f"\n{q['id']}  ({secs:.1f}s, {r} rounds, {dict(t)}{', CAP' if cap else ''}, {len(out.contexts)} items, "
          f"hops: {' '.join(hops)})  {q['question']}")
    print(f"  ANSWER  {out.answer}")
    print(f"  TRUTH   {q['ground_truth']}")
