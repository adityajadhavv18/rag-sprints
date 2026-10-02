"""Phase 6 check: traces reach LangSmith, nested per question, tagged by arch; judge cost is counted.

Run from rag_sprints/:  uv run check_tracing.py     (5 questions, costs < $0.02)
"""

import re
import time
import warnings
from datetime import datetime
from pathlib import Path

from langsmith import Client

from eval_harness import (PipelineOutput, flush, load_golden, run_pipeline, score_run, settings, traceable,
                          traced_openai)

DATA_DIR = Path(__file__).resolve().parent / "data"
PARAGRAPHS = [(p.stem, para) for p in DATA_DIR.glob("*.txt") for para in p.read_text().split("\n\n") if len(para) > 80]
client = traced_openai()


@traceable(run_type="retriever", name="keyword_retrieve")
def retrieve(question: str, k: int = 3) -> list[tuple[str, str]]:
    """Toy retriever: rank paragraphs by word overlap with the question. (Real retrieval arrives in Phase 8.)"""
    words = set(re.findall(r"[a-z0-9]{4,}", question.lower()))
    scored = sorted(PARAGRAPHS, key=lambda dp: -len(words & set(re.findall(r"[a-z0-9]{4,}", dp[1].lower()))))
    return scored[:k]


@traceable(run_type="chain", name="toy_pipeline")
def toy_pipeline(question: str) -> PipelineOutput:
    if "shareholders of record" in question:
        raise RuntimeError("simulated crash: should show as a red trace in LangSmith")
    hits = retrieve(question)
    context = "\n\n".join(text for _, text in hits)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": f"Answer using only this context. If it isn't there, say so.\n\n"
                                              f"{context}\n\nQuestion: {question}"}],
    )
    return PipelineOutput(answer=resp.choices[0].message.content, contexts=[t for _, t in hits],
                          context_ids=[d for d, _ in hits], tokens_in=resp.usage.prompt_tokens,
                          tokens_out=resp.usage.completion_tokens)


golden = [g for g in load_golden() if g.id in {"q01", "q04", "q11", "q20", "q30"}]
run_label = f"tracing-check-{datetime.now():%Y%m%d-%H%M%S}"
results = run_pipeline(toy_pipeline, golden, arch="toy", run_label=run_label)
summary = score_run(golden, results)
flush()

print(f"\npipeline cost: ${summary.get('cost_total_usd', 0):.5f}   judge cost: ${summary['judge_cost_usd']:.5f}")

# Verify in LangSmith itself (ingestion can lag a few seconds).
# list_runs() is deprecated in favour of client.runs.query() (removal after Jan 2027); fine for this check.
warnings.filterwarnings("ignore", message=r"list_runs\(\) is deprecated")
ls = Client()
for attempt in range(6):
    roots = list(ls.list_runs(project_name=settings.langsmith_project, is_root=True,
                              filter=f'has(tags, "{run_label}")'))
    if len(roots) >= len(golden):
        break
    time.sleep(5)

print(f"\nLangSmith project '{settings.langsmith_project}': {len(roots)}/{len(golden)} question traces")
for root in sorted(roots, key=lambda r: r.name):
    children = list(ls.list_runs(project_name=settings.langsmith_project, trace_id=root.trace_id, is_root=False))
    steps = sorted({c.name for c in children})
    print(f"  {'❌' if root.error else '✅'} {root.name:<12} steps={steps}")
judge_runs = list(ls.list_runs(project_name=f"{settings.langsmith_project}-judge", is_root=True, limit=5))
print(f"LangSmith project '{settings.langsmith_project}-judge': {'✅ judge traces present' if judge_runs else '❌ none'}")
