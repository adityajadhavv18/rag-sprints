"""Score each retriever (dense alone, BM25 alone, fused hybrid) without the LLM. Free: retrieval metrics only,
nothing saved.

  1. evaluate() each path with use_llm_judge=False -> official metrics, deltas vs the naive baseline
  2. per question: where does the evidence rank in each path's top 20?

Run from rag_sprints/:  RAG_CORPUS=medical uv run week1/01_hybrid_rag/compare_paths.py
"""

import re

from eval_harness import PipelineOutput, evaluate, load_golden
from ingest import GOLDEN_SET
from pipeline import TOP_K, bm25_search, dense_search, hybrid_search

DEPTH = 20  # how far down each list we look for the evidence
PATHS = {"dense": dense_search, "bm25": bm25_search, "hybrid": hybrid_search}


def retrieval_only(search):
    """Wrap a retriever as a harness pipeline. answer is empty: with the judge off, only retrieval is scored."""
    def pipeline(question: str) -> PipelineOutput:
        hits = search(question, k=TOP_K)
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits])
    return pipeline


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_rank(hits, fragments) -> int | None:
    """Rank of the first chunk holding any evidence fragment (same matching rule as the harness)."""
    return next((i for i, h in enumerate(hits, start=1) if any(f in _norm(h.text) for f in fragments)), None)


def main():
    runs = {name: evaluate(retrieval_only(fn), arch=f"retrieval-{name}", golden_set=GOLDEN_SET,
                           use_llm_judge=False, save=False)
            for name, fn in PATHS.items()}

    print(f"Evidence rank per question in each path's top {DEPTH}  (- = not in top {DEPTH}; ✅ = inside top {TOP_K})")
    print(f"   {'id':<4} {'type':<11} {'dense':>6} {'bm25':>6} {'hybrid':>6}   single paths")
    wins = {"dense": 0, "bm25": 0}
    for item in load_golden(GOLDEN_SET):
        if item.question_type == "no_answer":
            continue
        fragments = [_norm(f) for f in re.split(r" \.\.\. | / ", item.evidence) if f.strip()]
        ranks = {name: evidence_rank(fn(item.question, k=DEPTH), fragments) for name, fn in PATHS.items()}
        inside = {name: r is not None and r <= TOP_K for name, r in ranks.items()}
        winner = ""
        if inside["dense"] != inside["bm25"]:
            winner = "dense only" if inside["dense"] else "bm25 only"
            wins["dense" if inside["dense"] else "bm25"] += 1
        elif not inside["dense"]:
            winner = "both miss"
        cell = lambda r: f"{r if r else '-':>4}{' ✅' if r and r <= TOP_K else '  '}"
        print(f"   {item.id:<4} {item.question_type:<11} {cell(ranks['dense'])} {cell(ranks['bm25'])} "
              f"{cell(ranks['hybrid'])}   {winner}")
    print(f"\nQuestions only one path gets into the top {TOP_K}: dense {wins['dense']}, bm25 {wins['bm25']}")
    for name, run in runs.items():
        s = run.summary
        print(f"   {name:<6} evidence recall {s['evidence_recall']:.2f}   MRR {s['evidence_mrr']:.2f}")


if __name__ == "__main__":
    main()
