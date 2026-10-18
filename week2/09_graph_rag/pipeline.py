"""GraphRAG lite, the pipeline: Contextual hybrid search's top 5 + up to 4 chunks from the graph walk -> naive's generate().

  question ─┬─ find entities -> guided walk (retrieve.py) -> walked links' chunks (max 4) ─┐
            └─ sprint 03 hybrid search ────────────────────────────────── top 5 chunks ──┴─ graph first, no repeats,
                                                                                            max 9 -> generate()

The graph only ADDS chunks. A question that names nothing the graph knows ("my sugar crashed...") gets exactly
sprint 03's answer. The walk is free (no LLM), so the per-query cost is one generate() call, like Contextual.

graph_only_pipeline: the walk's chunks and nothing else (the chain comparison, and a judged run to see the graph alone).
"""

from dataclasses import dataclass

from build_graph import ROOT, _load
from eval_harness import PipelineOutput, traceable
from retrieve import chunk_text, walk

SEARCH_K = 5   # sprint 03's top 5, unchanged
GRAPH_K = 4    # extra chunks from the walk: c01's and c05's evidence sit within the walk's first 4 chunks


@dataclass
class Chunk:
    """What generate() reads: a doc id (shown as the source label) and the text."""
    chunk_id: str
    doc_id: str
    text: str


_ctx = _load("contextual_pipeline", ROOT / "week1" / "03_contextual_rag" / "pipeline.py")
generate = _ctx.hybrid.naive_pipeline.generate


def walked_chunks(question: str, skip: set[str] = frozenset(), k: int = GRAPH_K) -> list[Chunk]:
    """The walk's chunks in walk order (hop 1 first), minus the ones search already found."""
    ids = [c for c in walk(question).chunk_ids if c not in skip][:k]
    return [Chunk(c, c.split("::")[0], chunk_text()[c]) for c in ids]


def _output(question: str, chunks: list) -> PipelineOutput:
    resp = generate(question, chunks)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[c.text for c in chunks],
        context_ids=[c.doc_id for c in chunks],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )


@traceable(run_type="chain", name="graph_rag_pipeline")
def graph_rag_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls. contexts = the graph's chunks first, then search's (no repeats).

    Graph first: when the walk found something, its chunks are the named entity's own entry. With search first,
    c01 read 9 chunks and took "INR above 5.0" from a neighbouring lab entry (2 runs out of 2); graph first
    gave "above 100 seconds" (2 out of 2). No entity -> no graph chunks -> exactly sprint 03's answer."""
    graph = walked_chunks(question)
    hits = [h for h in _ctx.hybrid_search(question, k=SEARCH_K) if h.chunk_id not in {c.chunk_id for c in graph}]
    return _output(question, graph + hits)


@traceable(run_type="chain", name="graph_only_pipeline")
def graph_only_pipeline(question: str) -> PipelineOutput:
    """The walk alone (up to 5 chunks, the same count as search).

    No entity found -> refuse in code, no LLM call. Given zero chunks, generate() still answered 8 of 15 golden
    questions from its own knowledge, some made up ("E-12 means replace the IV site every 72 hours")."""
    chunks = walked_chunks(question, k=SEARCH_K)
    if not chunks:
        return PipelineOutput(answer=_ctx.hybrid.naive_pipeline.NO_ANSWER, contexts=[], context_ids=[],
                              tokens_in=0, tokens_out=0)
    return _output(question, chunks)
