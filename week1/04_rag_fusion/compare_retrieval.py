"""Phase 2: does asking 5 times beat asking once? Free: retrieval only, no LLM call, nothing saved.

Six setups, all handing 5 chunks to the LLM:
  hybrid / ctx-hyb      1 query, dense + BM25          (sprints 01 and 03, the baselines)
  fus-hyb / fus-ctx     5 queries, 10 lists            (this sprint, on each index)
  fus-ctx-2             original + 2 rephrasings       (do we need all 4?)
  fus-ctx-w2            original question counts x2    (does guarding the original help?)

Run from rag_sprints/:  uv run week1/04_rag_fusion/compare_retrieval.py
"""

import contextlib
import io
import statistics

from eval_harness import PipelineOutput, evaluate, load_golden
from eval_harness.metrics import _evidence_fragments, _norm
from pipeline import CANDIDATES, HYBRID, TOP_K, ctx, fusion_search
from rewrite import GOLDEN_SET, rephrase

DEPTH = 20  # how far down each ranking we look for the evidence
SETUPS = {
    "hybrid": lambda q, k=TOP_K: ctx.hybrid_search(q, k, collection=HYBRID),
    "ctx-hyb": lambda q, k=TOP_K: ctx.hybrid_search(q, k),
    "fus-hyb": lambda q, k=TOP_K: fusion_search(q, k, collection=HYBRID),
    "fus-ctx": lambda q, k=TOP_K: fusion_search(q, k),
    "fus-ctx-2": lambda q, k=TOP_K: fusion_search(q, k, n_rephrasings=2),
    "fus-ctx-w2": lambda q, k=TOP_K: fusion_search(q, k, original_weight=2.0),
}


def retrieval_only(search):
    """Wrap a retriever as a harness pipeline with no LLM call: only retrieval gets scored."""
    def pipeline(question: str) -> PipelineOutput:
        hits = search(question)
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits])
    return pipeline


def evidence_rank(hits, fragments) -> int | None:
    """Rank of the first chunk holding any evidence fragment (same matching rule as the harness)."""
    return next((i for i, h in enumerate(hits, start=1) if any(f in _norm(h.text) for f in fragments)), None)


def distinct_chunks(question: str) -> int:
    """How many different chunks the 10 lists of fus-ctx hold between them (100 slots, many repeats)."""
    seen = set()
    for q in [question] + rephrase(question):
        seen.update(h.chunk_id for h in ctx.dense_search(q, CANDIDATES))
        seen.update(h.chunk_id for h in ctx.bm25_search(q, CANDIDATES))
    return len(seen)


def main():
    golden = load_golden(GOLDEN_SET)
    types = {g.id: g.question_type for g in golden}

    print("1. scores (26 answerable questions, 5 chunks each)")
    print(f"     {'':11} {'evidence recall':>15} {'MRR':>6} {'simple':>8} {'exact':>7} {'multi':>7}")
    for name, search in SETUPS.items():
        with contextlib.redirect_stdout(io.StringIO()):  # evaluate() prints a full report; we only want numbers
            run = evaluate(retrieval_only(search), arch=name, golden_set=GOLDEN_SET,
                           use_llm_judge=False, save=False, compare_to=None)
        by_type = {t: statistics.mean(q.scores["evidence_recall"] for q in run.questions if types[q.golden_id] == t)
                   for t in ("simple", "exact_term", "multi_part")}
        print(f"     {name:11} {run.summary['evidence_recall']:>15.2f} {run.summary['evidence_mrr']:>6.2f} "
              f"{by_type['simple']:>8.2f} {by_type['exact_term']:>7.2f} {by_type['multi_part']:>7.2f}")

    print(f"\n2. evidence rank per question in the top {DEPTH}  (- = not found; ✅ = in the top {TOP_K})")
    print(f"     {'id':<4} {'type':<11}" + "".join(f"{name:>12}" for name in SETUPS))
    for item in golden:
        if item.question_type == "no_answer":
            continue
        fragments = _evidence_fragments(item)
        cells = []
        for search in SETUPS.values():
            r = evidence_rank(search(item.question, k=DEPTH), fragments)
            cells.append(f"{r if r else '-':>9}{' ✅' if r and r <= TOP_K else '  '} ")
        print(f"     {item.id:<4} {item.question_type:<11}" + "".join(cells))

    counts = [distinct_chunks(g.question) for g in golden]
    print(f"\n3. distinct chunks in fus-ctx's 10 lists (100 slots): median {statistics.median(counts):.0f}, "
          f"min {min(counts)}, max {max(counts)}  (of 180 in the index)")


if __name__ == "__main__":
    main()
