"""Evaluate Speculative RAG on the medical golden set (LLM judge on), two runs:

  contextual_4o  the baseline to race: the same 9 chunks, gpt-4o reads them all and answers (compared with Contextual)
  speculative    3 gpt-4o-mini children draft, gpt-4o parent picks one (compared with contextual_4o)

The harness prices every token at gpt-4o-mini's rate, which is wrong for both runs. The real cost, each call priced
at its own model, is printed and saved to cost_real.json. Also printed for speculative: on how many answerable
questions the evidence was somewhere in the 9 chunks, vs in the winner's 3 (harness evidence recall).

Run from rag_sprints/:  uv run week2/12_speculative_rag/run_eval.py   (~$0.45 incl. judge)
"""

import json
import sys
from pathlib import Path

from eval_harness import evaluate, settings
from eval_harness.golden import load_golden
from pipeline import ALL_HITS, USAGE, contextual_4o_pipeline, real_cost, search, speculative_pipeline

sys.path.append(str(Path(__file__).resolve().parents[1] / "06_corrective_rag"))
from grader import holds_evidence  # noqa: E402

GOLDEN_SET = "golden_medical_v1.jsonl"
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "eval_harness" / "results"
CONTEXTUAL_RUN = RESULTS / "contextual_20261012-211033.json"

golden = load_golden(GOLDEN_SET)
search("warm up the embedder")  # so the first question's latency isn't model loading

costs, files = {}, {}
for arch, fn in (("contextual_4o", contextual_4o_pipeline), ("speculative", speculative_pipeline)):
    print(f"\n=== {arch} ===")
    start = len(USAGE)
    run = evaluate(fn, arch=arch, golden_set=GOLDEN_SET,
                   compare_to=files.get("contextual_4o", CONTEXTUAL_RUN))
    files[arch] = RESULTS / f"{arch}_{run.started_at:%Y%m%d-%H%M%S}.json"
    calls = USAGE[start:]
    by_model = {}
    for _, model, i, o in calls:
        m = by_model.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        m["calls"] += 1
        m["tokens_in"] += i
        m["tokens_out"] += o
    for m, row in by_model.items():
        row["cost_usd"] = round(real_cost([(None, m, row["tokens_in"], row["tokens_out"])]), 5)
    costs[arch] = {"run": files[arch].name, "by_model": by_model,
                   "cost_per_query": round(real_cost(calls) / len(golden), 6),
                   "harness_cost_per_query": run.summary.get("cost_per_query_usd")}

answerable = [g for g in golden if g.question_type != "no_answer"]
in_nine = sum(any(holds_evidence(h.text, g.evidence) for h in ALL_HITS[g.question]) for g in answerable)
costs["speculative"]["evidence_in_any_of_9"] = f"{in_nine}/{len(answerable)}"
(HERE / "cost_real.json").write_text(json.dumps(costs, indent=2) + "\n")

print("\nReal cost per query (each model at its own price; the harness's cost column assumes gpt-4o-mini):")
for arch, c in costs.items():
    parts = ", ".join(f"{m} {r['calls']} calls ${r['cost_usd']}" for m, r in c["by_model"].items())
    print(f"  {arch:14s} ${c['cost_per_query']:.5f}/query  ({parts})")
print(f"Speculative: evidence somewhere in the 9 chunks on {in_nine}/{len(answerable)} answerable questions "
      f"(harness evidence recall counts only the winner's 3)")
print(f"Compare in detail:  uv run compare-runs {files['speculative'].name} {files['contextual_4o'].name}")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:speculative / arch:contextual_4o")
