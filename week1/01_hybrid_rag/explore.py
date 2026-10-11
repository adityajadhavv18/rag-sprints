"""Stage 3 demo: watch RRF fuse two lists, on the questions where each path alone fails.

For each question: the fused top 5 with each chunk's rank in dense and in bm25 and the RRF points that put
it there; the evidence chunk is marked. Then a cross-check of our hand-written RRF against Qdrant's built-in one.

Run from rag_sprints/:  RAG_CORPUS=medical uv run week1/01_hybrid_rag/explore.py
"""

import re

from qdrant_client import models

from eval_harness import load_golden
from ingest import COLLECTION, DENSE, GOLDEN_SET, SPARSE
from pipeline import CANDIDATES, RRF_K, TOP_K, _bm25, _dense, _qdrant, bm25_search, dense_search, rrf_fuse

DEMO = {
    "m01": "dense fails, bm25 wins (exact code)",
    "m02": "bm25 fails, dense wins (code shared by 3 manuals)",
    "m11": "both miss the top 5 alone; does agreement rescue it?",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def show(item, why: str):
    fragments = [_norm(f) for f in re.split(r" \.\.\. | / ", item.evidence) if f.strip()]
    is_evidence = lambda hit: any(f in _norm(hit.text) for f in fragments)
    lists = {"dense": dense_search(item.question, k=CANDIDATES), "bm25": bm25_search(item.question, k=CANDIDATES)}
    fused = rrf_fuse(lists)

    print(f"\n{'─' * 96}\n{item.id} ({why})\nQ: {item.question}")
    for path, hits in lists.items():
        rank = next((i for i, h in enumerate(hits, start=1) if is_evidence(h)), None)
        print(f"   {path:<6} alone: evidence at rank {rank or f'>{CANDIDATES}'}")
    rank = next((i for i, h in enumerate(fused, start=1) if is_evidence(h)), None)
    print(f"   fused       : evidence at rank {rank or '-'}  {'✅ reaches the LLM' if rank and rank <= TOP_K else '❌'}")

    print(f"\n   fused top {TOP_K + 2}   {'dense':>5} {'bm25':>5}   RRF points")
    for i, hit in enumerate(fused[:TOP_K + 2], start=1):
        d, b = hit.ranks.get("dense"), hit.ranks.get("bm25")
        parts = " + ".join(f"1/{RRF_K + r}" for r in (d, b) if r)
        cut = "   ← cut: only the top 5 reach the LLM" if i == TOP_K + 1 else ""
        print(f"   {i}. {'★' if is_evidence(hit) else ' '} {hit.chunk_id:<28} {d or '-':>5} {b or '-':>5}   "
              f"{parts} = {hit.score:.4f}{cut}")


def qdrant_native(question: str) -> list[str]:
    """The same fusion done server-side by Qdrant: two prefetches, fused with RRF (k=60), one round trip."""
    sparse = next(_bm25().query_embed(question))
    points = _qdrant().query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=next(_dense().query_embed(question)).tolist(), using=DENSE, limit=CANDIDATES),
            models.Prefetch(query=models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                            using=SPARSE, limit=CANDIDATES),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
        limit=TOP_K,
        with_payload=True,
    ).points
    return [p.payload["chunk_id"] for p in points]


def main():
    golden = {g.id: g for g in load_golden(GOLDEN_SET)}
    for gid, why in DEMO.items():
        show(golden[gid], why)

    print(f"\n{'─' * 96}\nCross-check: our rrf_fuse vs Qdrant's built-in RRF, top {TOP_K} chunk sets")
    answerable = [g for g in golden.values() if g.question_type != "no_answer"]
    same = 0
    for g in answerable:
        ours = [h.chunk_id for h in rrf_fuse({"dense": dense_search(g.question, k=CANDIDATES),
                                              "bm25": bm25_search(g.question, k=CANDIDATES)})[:TOP_K]]
        theirs = qdrant_native(g.question)
        if set(ours) == set(theirs):
            same += 1
        else:
            print(f"   {g.id}: differs  ours-only {sorted(set(ours) - set(theirs))}  "
                  f"qdrant-only {sorted(set(theirs) - set(ours))}")
    print(f"   identical top-{TOP_K} sets on {same}/{len(answerable)} questions")


if __name__ == "__main__":
    main()
