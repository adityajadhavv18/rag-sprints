"""Phase 3: do the notes help retrieval? Free: retrieval metrics only, no LLM, nothing saved.

Six setups, all handing 5 chunks to the LLM:
  naive / hybrid          450-token chunks, no notes        (sprints 00 and 01)
  control dense / hybrid  370-token chunks, no notes        (isolates the chunk-size change)
  contextual dense / hyb  370-token chunks + notes          (this sprint)

Run from rag_sprints/:  uv run week1/03_contextual_rag/compare_retrieval.py
"""

import contextlib
import io

from eval_harness import PipelineOutput, evaluate, load_golden
from eval_harness.metrics import _evidence_fragments, _norm
from ingest import CONTROL_COLLECTION
from contextualize import GOLDEN_SET
from pipeline import TOP_K, dense_search, hybrid, hybrid_search

DEPTH = 20  # how far down each list we look for the evidence
SETUPS = {
    "naive": lambda q, k=TOP_K: hybrid.dense_search(q, k=k),  # hybrid_medical's dense vector = naive's index
    "hybrid": lambda q, k=TOP_K: hybrid.hybrid_search(q, k=k),
    "ctrl-dense": lambda q, k=TOP_K: dense_search(q, k, collection=CONTROL_COLLECTION),
    "ctrl-hyb": lambda q, k=TOP_K: hybrid_search(q, k, collection=CONTROL_COLLECTION),
    "ctx-dense": lambda q, k=TOP_K: dense_search(q, k),
    "ctx-hyb": lambda q, k=TOP_K: hybrid_search(q, k),
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


def main():
    print("1. scores (26 answerable questions, 5 chunks each)")
    print(f"     {'':11} {'evidence recall':>15} {'MRR':>6}")
    for name, search in SETUPS.items():
        with contextlib.redirect_stdout(io.StringIO()):  # evaluate() prints a full report; we only want numbers
            run = evaluate(retrieval_only(search), arch=name, golden_set=GOLDEN_SET,
                           use_llm_judge=False, save=False, compare_to=None)
        print(f"     {name:11} {run.summary['evidence_recall']:>15.2f} {run.summary['evidence_mrr']:>6.2f}")

    print(f"\n2. evidence rank per question in the top {DEPTH}  (- = not found; ✅ = in the top {TOP_K})")
    print(f"     {'id':<4} {'type':<11}" + "".join(f"{name:>12}" for name in SETUPS))
    for item in load_golden(GOLDEN_SET):
        if item.question_type == "no_answer":
            continue
        fragments = _evidence_fragments(item)
        cells = []
        for search in SETUPS.values():
            r = evidence_rank(search(item.question, k=DEPTH), fragments)
            cells.append(f"{r if r else '-':>9}{' ✅' if r and r <= TOP_K else '  '} ")
        print(f"     {item.id:<4} {item.question_type:<11}" + "".join(cells))


if __name__ == "__main__":
    main()
