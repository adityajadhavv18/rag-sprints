"""RAG Fusion, query path. Search with 5 versions of the question; the LLM answers the original one.

  queries        the original question + 4 cached rephrasings from rewrite.py
  fusion_search  each query -> dense top 10 + BM25 top 10 -> 10 ranked lists -> Hybrid's rrf_fuse -> top 5
  fusion_pipeline  fusion_search -> naive's exact generate() with the ORIGINAL question

No new index: it searches Contextual's collection (default) or Hybrid's, both built in earlier sprints.
Requires Qdrant running (docker start qdrant). Medical corpus only.
"""

import importlib.util
import sys
from functools import cache
from pathlib import Path

from openai import OpenAI

from eval_harness import PipelineOutput, traceable
from rewrite import N_REPHRASINGS, ask_model, rephrase

# Reuse Contextual's search functions (they take a `collection`, so they also search Hybrid's index). This folder
# has no ingest.py, so putting Contextual's folder on the path lets its `import ingest` find Contextual's own.
# Its pipeline.py is loaded by path under another name, because this file is also called pipeline.py.
_CTX_DIR = Path(__file__).resolve().parents[1] / "03_contextual_rag"
sys.path.append(str(_CTX_DIR))
_spec = importlib.util.spec_from_file_location("contextual_pipeline", _CTX_DIR / "pipeline.py")
ctx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx)
hybrid = ctx.hybrid  # sprint 01's pipeline: rrf_fuse, RRF_K=10, naive's generate()

TOP_K, CANDIDATES = ctx.TOP_K, ctx.CANDIDATES  # 5 chunks to the LLM, 10 per list into fusion
CONTEXTUAL, HYBRID = ctx.COLLECTION, "hybrid_medical"  # notes + 370-token chunks / plain 450-token chunks


@traceable(run_type="retriever", name="fusion_search")
def fusion_search(question: str, k: int = TOP_K, n_rephrasings: int = N_REPHRASINGS,
                  original_weight: float = 1.0, collection: str = CONTEXTUAL, rephrasings: list[str] | None = None):
    """5 queries x (dense + BM25) = 10 lists, merged by RRF. A chunk earns 1/(10 + rank) in every list it is in.

    original_weight > 1 makes the user's own question count for more than each rephrasing, as a guard against
    rephrasings that drift away from what was asked. Rephrasings come from the cache unless they are passed in.
    """
    rephrasings = rephrase(question) if rephrasings is None else rephrasings
    queries = [question] + rephrasings[:n_rephrasings]
    lists, weights = {}, {}
    for i, q in enumerate(queries):
        lists[f"q{i}_dense"] = ctx.dense_search(q, CANDIDATES, collection)
        lists[f"q{i}_bm25"] = ctx.bm25_search(q, CANDIDATES, collection)
    weights["q0_dense"] = weights["q0_bm25"] = original_weight  # q0 = the original question
    return hybrid.rrf_fuse(lists, weights=weights)[:k]


@traceable(run_type="llm", name="rewrite_question")
def _rewrite_live(question: str):
    return ask_model(_openai(), question)


@cache
def _openai() -> OpenAI:
    return OpenAI()


@traceable(run_type="chain", name="fusion_pipeline")
def fusion_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls. The LLM sees the original question and the original chunk texts.

    The rewrite runs LIVE here (no cache), as it would in production, so latency includes it. Its tokens are added
    to tokens_in/out: both calls are gpt-4o-mini, so the harness prices the sum correctly.
    """
    rephrasings, rewrite_usage = _rewrite_live(question)
    hits = fusion_search(question, rephrasings=rephrasings)
    resp = hybrid.naive_pipeline.generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens + rewrite_usage["in"],
        tokens_out=resp.usage.completion_tokens + rewrite_usage["out"],
    )
