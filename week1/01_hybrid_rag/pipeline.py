"""Hybrid RAG, query path.

  dense_search   question -> bge vector    -> nearest chunks by cosine   ("the meaning librarian")
  bm25_search    question -> its BM25 words -> chunks sharing rare words  ("the keyword librarian")
  hybrid_search  ask both for CANDIDATES chunks, merge with Reciprocal Rank Fusion, keep the top TOP_K
  hybrid_pipeline  hybrid_search -> naive's exact generate() step, so retrieval is the only thing that differs

Requires the index built by ingest.py and Qdrant running (docker start qdrant).
"""

import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

import ingest as _hybrid_ingest
from eval_harness import PipelineOutput, traceable
from ingest import BM25_MODEL, COLLECTION, DENSE, EMBED_MODEL, QDRANT_URL, SPARSE

# Reuse naive's generate() (prompt, model, temperature, chunk labels) instead of copying it. Naive's pipeline.py
# does `from ingest import ...`, which would find OUR ingest; point that name at naive's ingest while it loads.
_spec = importlib.util.spec_from_file_location(
    "naive_pipeline", Path(__file__).resolve().parents[1] / "00_naive_baseline" / "pipeline.py")
naive_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["ingest"] = _hybrid_ingest.naive
try:
    _spec.loader.exec_module(naive_pipeline)
finally:
    sys.modules["ingest"] = _hybrid_ingest

TOP_K = 5  # chunks passed to the LLM, same as naive
# Tuned in tune_fusion.py (was 20 / 60, the paper defaults): best on the medical set (evidence recall 0.81 -> 0.88)
# and still +0.02 on the held-out 10-K set. Path weights were left equal: favouring either path helped one corpus
# and hurt the other.
CANDIDATES = 10  # how deep each path's list goes into fusion
RRF_K = 10  # damping in 1 / (k + rank); the RRF paper (Cormack et al., 2009) uses 60


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # dense: cosine similarity (0..1); bm25: summed IDF x TF (unbounded); hybrid: RRF points
    ranks: dict[str, int] = field(default_factory=dict)  # hybrid only: rank in each path that found it


@cache  # load models and connect once, not on every question
def _dense() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


@cache
def _bm25() -> SparseTextEmbedding:
    # Query side needs no avg_len: BM25 only applies length normalisation to documents
    return SparseTextEmbedding(BM25_MODEL)


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _hits(points) -> list[Hit]:
    return [Hit(p.payload["chunk_id"], p.payload["doc_id"], p.payload["text"], p.score) for p in points]


@traceable(run_type="retriever", name="dense_search")
def dense_search(question: str, k: int = TOP_K) -> list[Hit]:
    """Same model and chunks as naive, so this alone should reproduce naive's retrieval exactly."""
    query_vector = next(_dense().query_embed(question)).tolist()
    return _hits(_qdrant().query_points(COLLECTION, query=query_vector, using=DENSE, limit=k,
                                        with_payload=True).points)


@traceable(run_type="retriever", name="bm25_search")
def bm25_search(question: str, k: int = TOP_K) -> list[Hit]:
    """Every question word gets weight 1; Qdrant multiplies in each chunk's TF weight and the word's IDF."""
    sparse = next(_bm25().query_embed(question))
    query = models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
    return _hits(_qdrant().query_points(COLLECTION, query=query, using=SPARSE, limit=k, with_payload=True).points)


def rrf_fuse(ranked_lists: dict[str, list[Hit]], k: int = RRF_K, weights: dict[str, float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion: each list gives a chunk weight / (k + rank) points; add them up, sort.

    Only RANKS are used, never the raw scores, because cosine (0..1) and BM25 (0..30ish) live on different
    scales. A chunk ranked well by both lists beats one ranked #1 by a single list (k=10):
        #2 in both: 1/12 + 1/12 = 0.167     #1 in one only: 1/11 = 0.091
    Smaller k widens the gap between rank 1 and rank 10 (k=10: 1.8x, k=60: 1.15x), so a confident single-list #1
    survives more often. Weights let one path count for more (default 1 each).
    """
    weights = weights or {}
    points: dict[str, float] = defaultdict(float)
    by_id: dict[str, Hit] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for path, hits in ranked_lists.items():
        for rank, hit in enumerate(hits, start=1):
            points[hit.chunk_id] += weights.get(path, 1.0) / (k + rank)
            ranks[hit.chunk_id][path] = rank
            by_id.setdefault(hit.chunk_id, hit)
    # Ties (e.g. #3 in dense only vs #3 in bm25 only) go to the better single rank, then to chunk_id for stability
    order = sorted(points, key=lambda cid: (-points[cid], min(ranks[cid].values()), cid))
    return [Hit(by_id[cid].chunk_id, by_id[cid].doc_id, by_id[cid].text, points[cid], ranks[cid]) for cid in order]


@traceable(run_type="retriever", name="hybrid_search")
def hybrid_search(question: str, k: int = TOP_K, candidates: int = CANDIDATES) -> list[Hit]:
    """Ask both librarians for `candidates` chunks each, fuse, keep the top k."""
    lists = {"dense": dense_search(question, k=candidates), "bm25": bm25_search(question, k=candidates)}
    return rrf_fuse(lists)[:k]


@traceable(run_type="chain", name="hybrid_pipeline")
def hybrid_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls: question in, PipelineOutput out."""
    hits = hybrid_search(question)
    resp = naive_pipeline.generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )
