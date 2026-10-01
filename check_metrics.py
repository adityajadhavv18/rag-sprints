"""Phase 5 check: metrics must clearly separate a perfect pipeline from a bad one.

Run from rag_sprints/:  uv run check_metrics.py     (~6 questions x 2 pipelines, costs < $0.05)
"""

import re
from pathlib import Path

from openai import OpenAI

from eval_harness import PipelineOutput, load_golden, run_pipeline, score_run, settings

DATA_DIR = Path(__file__).resolve().parent / "data"
corpus = {p.stem: p.read_text() for p in DATA_DIR.glob("*.txt")}
golden = [g for g in load_golden() if g.id in {"q01", "q06", "q08", "q15", "q24", "q30"}]
by_question = {g.question: g for g in golden}
client = OpenAI()


def passage_around(doc_id: str, quote: str, width: int = 600) -> str:
    text = corpus[doc_id]
    i = text.find(quote)
    return text[max(0, i - width): i + len(quote) + width]


def oracle_pipeline(question: str) -> PipelineOutput:
    """Cheats: reads the answer key. Retrieves the exact evidence passage and returns ground_truth."""
    g = by_question[question]
    if g.question_type == "no_answer":
        return PipelineOutput(answer="The documents do not contain this information.",
                              contexts=[corpus["aapl_10k_2025"][5000:6200]], context_ids=["aapl_10k_2025"])
    fragments = re.split(r" \.\.\. | / ", g.evidence)
    contexts = [passage_around(g.source_doc, f) for f in fragments]
    return PipelineOutput(answer=g.ground_truth, contexts=contexts, context_ids=[g.source_doc] * len(contexts))


def bad_pipeline(question: str) -> PipelineOutput:
    """Retrieves an irrelevant passage from the wrong company and lets the LLM guess."""
    g = by_question[question]
    wrong_doc = "tsla_10k_2025" if g.source_doc != "tsla_10k_2025" else "aapl_10k_2025"
    resp = client.chat.completions.create(
        model=settings.llm_model, messages=[{"role": "user", "content": f"Answer in one sentence: {question}"}])
    return PipelineOutput(answer=resp.choices[0].message.content,
                          contexts=[corpus[wrong_doc][20000:21200]], context_ids=[wrong_doc],
                          tokens_in=resp.usage.prompt_tokens, tokens_out=resp.usage.completion_tokens)


summaries = {}
for name, fn in [("oracle", oracle_pipeline), ("bad", bad_pipeline)]:
    print(f"\n=== {name} pipeline")
    results = run_pipeline(fn, golden, verbose=False)
    summaries[name] = score_run(golden, results)

keys = ["evidence_recall", "evidence_mrr", "doc_hit", "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "abstention", "latency_p50_ms", "cost_total_usd"]
print(f"\n{'metric':<20}{'oracle':>10}{'bad':>10}")
for k in keys:
    o, b = summaries["oracle"].get(k), summaries["bad"].get(k)
    fmt = lambda v: "-" if v is None else f"{v:.4f}" if k.startswith("cost") else f"{v:.2f}"
    print(f"{k:<20}{fmt(o):>10}{fmt(b):>10}")
