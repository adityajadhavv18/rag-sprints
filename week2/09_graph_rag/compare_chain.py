"""GraphRAG lite on the 5 chain questions: Contextual search vs graph only vs graph + search, real answers.

For each question and pipeline: how many chunks the LLM read, which hops' evidence was among them (position),
the answer, and the ground truth. Agentic RAG's results (sprint 08) are in its README, for comparison.

Run from rag_sprints/:  uv run week2/09_graph_rag/compare_chain.py   (15 generate() calls, < $0.01)
"""

import json

from build_graph import ROOT
from eval_harness.metrics import _norm
from pipeline import _ctx, graph_only_pipeline, graph_rag_pipeline

PIPELINES = {"contextual": _ctx.contextual_pipeline, "graph only": graph_only_pipeline,
             "graph + search": graph_rag_pipeline}
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens


def hop_position(evidence: str, contexts: list[str]) -> str:
    for i, text in enumerate(contexts, 1):
        if _norm(evidence) in _norm(text):
            return f"#{i}"
    return "MISSED"


def main():
    chains = [json.loads(line) for line in (ROOT / "week1" / "05_adaptive_rag" / "questions_extra.jsonl").open()
              if '"route": "multi"' in line]
    cost = 0.0
    for q in chains:
        print(f"\n{'=' * 100}\n{q['id']}: {q['question']}\n  TRUTH  {q['ground_truth']}")
        for name, pipe in PIPELINES.items():
            out = pipe(q["question"])
            cost += (out.tokens_in * PRICE_IN + out.tokens_out * PRICE_OUT) / 1e6
            hops = " ".join(hop_position(h["evidence"], out.contexts) for h in q["hops"])
            print(f"\n  [{name}]  {len(out.contexts)} chunks, {out.tokens_in} tokens in, hops: {hops}")
            print(f"    {out.answer}")
    print(f"\n{'=' * 100}\ncost: ${cost:.4f}")


if __name__ == "__main__":
    main()
