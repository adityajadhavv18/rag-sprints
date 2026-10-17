"""Evaluate Self-RAG on the medical golden set (tracing on, LLM judge on), compared with Corrective and Contextual.

One run, arch "self_rag": search -> [IsRel] -> answer -> [IsSup + IsUse] -> wider search / rewrite -> answer.
Latency and cost include every grader, checker and rewrite call.

Faithfulness is Self-RAG's target metric and the judge's noisiest one, so this also:
  - prints faithfulness per question for Contextual, Corrective and Self-RAG
  - re-judges Corrective's and Self-RAG's saved answers once more (same answers, same chunks): any change in a
    score is judge noise, which shows how big a difference has to be before it means anything

Run from rag_sprints/:  uv run week2/07_self_rag/run_eval.py   (~$0.10 incl. both judge passes)
"""

from pathlib import Path

from eval_harness import evaluate, settings
from eval_harness.golden import load_golden
from eval_harness.metrics import is_refusal, ragas_scores
from eval_harness.report import load_run
from pipeline import PATH_LOG, self_rag_pipeline

ARCH = "self_rag"
GOLDEN_SET = "golden_medical_v1.jsonl"
RESULTS = Path(__file__).resolve().parents[3] / "eval_harness" / "results"
CORRECTIVE_RUN = RESULTS / "corrective_20261015-202424.json"
CONTEXTUAL_RUN = RESULTS / "contextual_20261012-211033.json"


def faith(run) -> dict[str, float | None]:
    return {q.golden_id: q.scores.get("faithfulness") for q in run.questions}


def rejudge(run, golden) -> dict[str, float | None]:
    """Second judge pass on a saved run's answers. Refusals are not scored, as in the harness."""
    answered = [q for q in run.questions if q.output is not None and not is_refusal(q.output.answer)]
    scores, _ = ragas_scores(golden, answered)
    return {gid: s.get("faithfulness") for gid, s in scores.items()}


def fmt(x: float | None) -> str:
    return " –  " if x is None else f"{x:.2f}"


print(f"\n=== {ARCH}: contextual hybrid -> [IsRel] -> answer -> [IsSup+IsUse] -> widen once / rewrite x3 ===")
run = evaluate(self_rag_pipeline, arch=ARCH, golden_set=GOLDEN_SET, compare_to=CORRECTIVE_RUN)

golden = load_golden(GOLDEN_SET)
questions = {g.question: g for g in golden}
paths = {questions[q].id: p for q, p in PATH_LOG.items()}
print(f"\nPaths: {sum(p['widened'] for p in paths.values())} widened, "
      f"{sum(p['rewrites'] > 0 for p in paths.values())} rewritten ({sum(p['rewrites'] for p in paths.values())} "
      f"rewrites), {sum(p['flag'] is not None for p in paths.values())} flagged, "
      f"{sum(not p['widened'] and not p['rewrites'] for p in paths.values())} straight through")
for gid, p in sorted(paths.items()):
    if p["widened"] or p["rewrites"] or p["flag"]:
        print(f"  {gid}: widened={p['widened']} rewrites={p['rewrites']} flag={p['flag']}")
        for i, a in enumerate(p["answers"], start=1):
            print(f"      answer {i}: {a[:140]}")

print("\n▶ second judge pass on Corrective's and Self-RAG's answers (noise check)")
corrective = load_run(CORRECTIVE_RUN)
contextual = load_run(CONTEXTUAL_RUN)
cols = {"ctx": faith(contextual), "crag": faith(corrective), "crag2": rejudge(corrective, golden),
        "self": faith(run), "self2": rejudge(run, golden)}

print(f"\nFaithfulness per question (– = refusal or no-answer question, not scored)")
print(f"{'id':4} {'type':10} {'Ctx':>5} {'Corr':>5} {'re':>5} {'Self':>5} {'re':>5}  path")
for g in golden:
    p = paths.get(g.id, {})
    path = ("widen " if p.get("widened") else "") + (f"rewrite×{p['rewrites']} " if p.get("rewrites") else "") + \
           (f"FLAG:{p['flag']}" if p.get("flag") else "")
    print(f"{g.id:4} {g.question_type:10} " + " ".join(f"{fmt(cols[c].get(g.id)):>5}" for c in cols) + f"  {path}")


def mean(col: dict) -> str:
    vals = [v for v in col.values() if v is not None]
    return f"{sum(vals) / len(vals):.3f} (n={len(vals)})"


def noise(a: dict, b: dict) -> str:
    diffs = [abs(a[k] - b[k]) for k in a if a.get(k) is not None and b.get(k) is not None]
    return f"mean |change| {sum(diffs) / len(diffs):.2f}, {sum(d >= 0.2 for d in diffs)} of {len(diffs)} moved ≥ 0.2"


print(f"\nMean faithfulness: Contextual {mean(cols['ctx'])} | Corrective {mean(cols['crag'])} -> re-judged "
      f"{mean(cols['crag2'])} | Self-RAG {mean(cols['self'])} -> re-judged {mean(cols['self2'])}")
print(f"Judge noise (same answers, judged twice): Corrective {noise(cols['crag'], cols['crag2'])}; "
      f"Self-RAG {noise(cols['self'], cols['self2'])}")
print(f"\nCompare in detail:  uv run compare-runs {ARCH}_{run.started_at:%Y%m%d-%H%M%S}.json {CORRECTIVE_RUN.name}")
print(f"Traces: LangSmith project '{settings.langsmith_project}', filter tag arch:{ARCH}")
