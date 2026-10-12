"""Phase 2: search the children for every golden question and see where the evidence lands. Free, no LLM.

For each question, two ranks in the top DEPTH children:
  child rank  = first child whose OWN text holds the evidence    (the search found the exact line)
  parent rank = first child whose PARENT holds the evidence      (the search found any line of the right entry)
The gap between them is what Parent-Child gains: a near-miss line still brings the right entry along.
Naive's rank (450-token chunks, dense) is shown for comparison.

Run from rag_sprints/:  uv run week1/02_parent_child_rag/child_retrieval.py
"""

from collections import Counter

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from eval_harness.golden import load_golden
from eval_harness.metrics import _evidence_fragments, _norm
from ingest import EMBED_MODEL, QDRANT_URL, chunking
from pipeline import dense_child_search, hybrid_child_search

DEPTH = 30  # how far down the child list we look
TOP = 5  # "made it": inside the first 5 children
NAIVE_COLLECTION = "naive_medical"


def first_rank(texts, fragments) -> int | None:
    """Rank of the first text holding any evidence fragment (same matching rule as the harness)."""
    return next((i for i, t in enumerate(texts, start=1) if any(f in _norm(t) for f in fragments)), None)


def naive_texts(question: str, k: int) -> list[str]:
    vector = next(TextEmbedding(EMBED_MODEL).query_embed(question)).tolist()
    points = QdrantClient(url=QDRANT_URL).query_points(NAIVE_COLLECTION, query=vector, limit=k, with_payload=True).points
    return [p.payload["text"] for p in points]


def cell(rank) -> str:
    return f"{rank if rank else '-':>3}{' ✅' if rank and rank <= TOP else '  '}"


def main():
    golden = [g for g in load_golden(chunking.GOLDEN_SET) if g.question_type != "no_answer"]
    made_it = Counter()
    print(f"Evidence rank in the top {DEPTH}   (- = not found, ✅ = inside top {TOP})")
    print(f"{'':16} {'naive':>6} | {'dense children':^15} | {'hybrid children':^15}")
    print(f"{'id   type':16} {'chunk':>6} | {'child':>6} {'parent':>7} | {'child':>6} {'parent':>7}")
    for g in golden:
        fragments = _evidence_fragments(g)
        dense, hybrid = dense_child_search(g.question, DEPTH), hybrid_child_search(g.question, DEPTH)
        ranks = {
            "naive": first_rank(naive_texts(g.question, DEPTH), fragments),
            "dense child": first_rank([h.text for h in dense], fragments),
            "dense parent": first_rank([h.parent_text for h in dense], fragments),
            "hybrid child": first_rank([h.text for h in hybrid], fragments),
            "hybrid parent": first_rank([h.parent_text for h in hybrid], fragments),
        }
        made_it.update(name for name, r in ranks.items() if r and r <= TOP)
        print(f"{g.id:<4} {g.question_type:<11} {cell(ranks['naive'])} | {cell(ranks['dense child'])} "
              f"{cell(ranks['dense parent'])} | {cell(ranks['hybrid child'])} {cell(ranks['hybrid parent'])}")

    n = len(golden)
    print(f"\nEvidence inside the top {TOP}, out of {n} questions:")
    for name in ("naive", "dense child", "dense parent", "hybrid child", "hybrid parent"):
        print(f"   {name:<14} {made_it[name]:>2}/{n}")

    # How many different entries do the top children actually cover? (duplicates = many lines from one entry,
    # or identical lines like 'Renal adjustment: No adjustment.' from many entries)
    distinct = [len({h.parent_id for h in dense_child_search(g.question, TOP)}) for g in golden]
    print(f"\nDense: the top {TOP} children come from {sum(distinct) / n:.1f} different parents on average")


if __name__ == "__main__":
    main()
