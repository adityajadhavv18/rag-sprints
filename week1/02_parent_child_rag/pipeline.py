"""Parent-Child RAG, query path.

  dense_child_search   question -> bge vector     -> nearest CHILDREN by cosine
  bm25_child_search    question -> its BM25 words -> children sharing rare words
  hybrid_child_search  both, merged with Hybrid's Reciprocal Rank Fusion
  parent_search        search CHILD_DEPTH children -> swap each for its parent -> drop repeats -> keep N_PARENTS
  parent_child_pipeline  parent_search -> naive's exact generate() step on the PARENTS

Every hit carries its parent's text (stored in the payload at ingest), so finding a child also hands us its parent.

Requires the index built by ingest.py and Qdrant running (docker start qdrant).
"""

import sys
import importlib.util
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

import ingest as _pc_ingest
from eval_harness import PipelineOutput, traceable
from ingest import COLLECTION, DENSE, EMBED_MODEL, QDRANT_URL, SPARSE

# Reuse Hybrid's rrf_fuse. Hybrid's pipeline.py does `import ingest`, which would find OUR ingest;
# point that name at Hybrid's ingest while it loads.
_spec = importlib.util.spec_from_file_location(
    "hybrid_pipeline", Path(__file__).resolve().parents[1] / "01_hybrid_rag" / "pipeline.py")
hybrid_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["ingest"] = _pc_ingest.hybrid
try:
    _spec.loader.exec_module(hybrid_pipeline)
finally:
    sys.modules["ingest"] = _pc_ingest

RRF_K = hybrid_pipeline.RRF_K  # 10, tuned in sprint 01
CHILD_DEPTH = 30  # children searched before swapping to parents
# Parents handed to the LLM. Picked in tune_parents.py: 10 beats naive on evidence recall (0.77 vs 0.75) and MRR
# (0.59 vs 0.54) while the LLM reads about half the text (1,167 vs 2,120 tokens). 5 parents: 0.73 / 0.58, 608 tokens.
N_PARENTS = 10


@dataclass
class ChildHit:
    chunk_id: str  # the child's id; named chunk_id so Hybrid's rrf_fuse can use it unchanged
    parent_id: str
    doc_id: str
    text: str  # the child line that matched
    parent_text: str  # the whole entry it belongs to
    score: float


@cache
def _dense() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


@cache
def _bm25() -> SparseTextEmbedding:
    return SparseTextEmbedding(_pc_ingest.hybrid.BM25_MODEL)


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def _hits(points) -> list[ChildHit]:
    return [ChildHit(p.payload["child_id"], p.payload["parent_id"], p.payload["doc_id"], p.payload["text"],
                     p.payload["parent_text"], p.score) for p in points]


@traceable(run_type="retriever", name="dense_child_search")
def dense_child_search(question: str, k: int) -> list[ChildHit]:
    query_vector = next(_dense().query_embed(question)).tolist()
    return _hits(_qdrant().query_points(COLLECTION, query=query_vector, using=DENSE, limit=k,
                                        with_payload=True).points)


@traceable(run_type="retriever", name="bm25_child_search")
def bm25_child_search(question: str, k: int) -> list[ChildHit]:
    sparse = next(_bm25().query_embed(question))
    query = models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
    return _hits(_qdrant().query_points(COLLECTION, query=query, using=SPARSE, limit=k, with_payload=True).points)


@traceable(run_type="retriever", name="hybrid_child_search")
def hybrid_child_search(question: str, k: int) -> list[ChildHit]:
    """Ask both searches for k children each, fuse with RRF, keep the top k."""
    dense, bm25 = dense_child_search(question, k), bm25_child_search(question, k)
    by_id = {h.chunk_id: h for h in dense + bm25}
    fused = hybrid_pipeline.rrf_fuse({"dense": dense, "bm25": bm25}, k=RRF_K)[:k]
    return [ChildHit(**{**vars(by_id[f.chunk_id]), "score": f.score}) for f in fused]


@dataclass
class ParentHit:
    parent_id: str
    doc_id: str
    text: str  # the whole entry: what the LLM reads
    child_rank: int  # rank of the best child that pointed here
    child_text: str  # that child's line


def to_parents(children: list[ChildHit], n: int) -> list[ParentHit]:
    """Walk the children best-first, swap each for its parent, skip parents already taken, stop at n.
    A parent ranks where its BEST child ranked."""
    parents, seen = [], set()
    for rank, child in enumerate(children, start=1):
        if child.parent_id not in seen:
            seen.add(child.parent_id)
            parents.append(ParentHit(child.parent_id, child.doc_id, child.parent_text, rank, child.text))
            if len(parents) == n:
                break
    return parents


@traceable(run_type="retriever", name="parent_search")
def parent_search(question: str, n: int = N_PARENTS, depth: int = CHILD_DEPTH,
                  search=dense_child_search) -> list[ParentHit]:
    return to_parents(search(question, depth), n)


@traceable(run_type="chain", name="parent_child_pipeline")
def parent_child_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls. contexts = the PARENTS, i.e. exactly what the LLM saw."""
    parents = parent_search(question)
    resp = hybrid_pipeline.naive_pipeline.generate(question, parents)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[p.text for p in parents],
        context_ids=[p.doc_id for p in parents],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )
