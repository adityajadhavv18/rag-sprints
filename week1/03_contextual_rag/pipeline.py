"""Contextual RAG, query path. Search matches note + chunk; the LLM reads the original chunk only.

  dense_search    question -> bge vector     -> nearest (note + chunk) by cosine
  bm25_search     question -> its BM25 words -> (note + chunk) sharing rare words
  hybrid_search   both, fused with Hybrid's rrf_fuse (same k and depth as sprint 01)
  contextual_pipeline  search -> naive's exact generate() step

`collection` lets the same functions search the control (370-token chunks, no notes).
Requires the collections built by ingest.py and Qdrant running (docker start qdrant).
"""

import importlib.util
import sys
from pathlib import Path

from qdrant_client import models

import ingest as _ctx_ingest
from eval_harness import PipelineOutput, traceable
from ingest import COLLECTION, DENSE, SPARSE

# Reuse Hybrid's pipeline (rrf_fuse, cached models, naive's generate). Hybrid's pipeline.py does `import ingest`,
# which would find OUR ingest.py; point that name at Hybrid's ingest while it loads, then put ours back.
_spec = importlib.util.spec_from_file_location(
    "hybrid_pipeline", Path(__file__).resolve().parents[1] / "01_hybrid_rag" / "pipeline.py")
hybrid = importlib.util.module_from_spec(_spec)
sys.modules["ingest"] = _ctx_ingest.hybrid
try:
    _spec.loader.exec_module(hybrid)
finally:
    sys.modules["ingest"] = _ctx_ingest

TOP_K, CANDIDATES = hybrid.TOP_K, hybrid.CANDIDATES  # 5 chunks to the LLM, 10 per path into fusion


@traceable(run_type="retriever", name="contextual_dense_search")
def dense_search(question: str, k: int = TOP_K, collection: str = COLLECTION) -> list:
    query_vector = next(hybrid._dense().query_embed(question)).tolist()
    return hybrid._hits(hybrid._qdrant().query_points(collection, query=query_vector, using=DENSE, limit=k,
                                                      with_payload=True).points)


@traceable(run_type="retriever", name="contextual_bm25_search")
def bm25_search(question: str, k: int = TOP_K, collection: str = COLLECTION) -> list:
    sparse = next(hybrid._bm25().query_embed(question))
    query = models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
    return hybrid._hits(hybrid._qdrant().query_points(collection, query=query, using=SPARSE, limit=k,
                                                      with_payload=True).points)


@traceable(run_type="retriever", name="contextual_hybrid_search")
def hybrid_search(question: str, k: int = TOP_K, candidates: int = CANDIDATES, collection: str = COLLECTION) -> list:
    lists = {"dense": dense_search(question, candidates, collection),
             "bm25": bm25_search(question, candidates, collection)}
    return hybrid.rrf_fuse(lists)[:k]


@traceable(run_type="chain", name="contextual_pipeline")
def contextual_pipeline(question: str, search=hybrid_search) -> PipelineOutput:
    """The function the harness calls. contexts = the original chunks, exactly what the LLM saw."""
    hits = search(question)
    resp = hybrid.naive_pipeline.generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )
