"""Naive baseline, query path: embed question -> top-k vector search -> gpt-4o-mini answers from those chunks.

Requires the index built by ingest.py and Qdrant running (docker start qdrant).
"""

from dataclasses import dataclass
from functools import cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from eval_harness import PipelineOutput, settings, traceable, traced_openai
from ingest import COLLECTION, EMBED_MODEL, QDRANT_URL

TOP_K = 5
NO_ANSWER = "I don't know based on the provided documents."

PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "{no_answer}"
Be concise and include the specific figures from the context.

Context:
{context}

Question: {question}"""


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity: closer to 1 = closer in meaning


@cache  # load the model and connect once, not on every question
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


@traceable(run_type="retriever", name="vector_search")
def retrieve(question: str, k: int = TOP_K) -> list[Hit]:
    """Pure vector search: no keywords, no query rewriting, no reranking. That's what makes it naive."""
    query_vector = next(_embedder().query_embed(question)).tolist()
    points = _qdrant().query_points(COLLECTION, query=query_vector, limit=k, with_payload=True).points
    return [Hit(p.payload["chunk_id"], p.payload["doc_id"], p.payload["text"], p.score) for p in points]


@cache
def _llm():
    return traced_openai()


@traceable(run_type="chain", name="generate")
def generate(question: str, hits: list[Hit]):
    # Label each chunk with its source document: the one piece of help the naive baseline gives the model,
    # since many chunks only say "the Company"
    context = "\n\n".join(f"[{i}] ({h.doc_id})\n{h.text}" for i, h in enumerate(hits, start=1))
    return _llm().chat.completions.create(
        model=settings.llm_model,
        temperature=0,  # same question -> (almost) same answer, so runs are comparable
        messages=[{"role": "user", "content": PROMPT.format(no_answer=NO_ANSWER, context=context, question=question)}],
    )


@traceable(run_type="chain", name="naive_pipeline")
def naive_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls: question in, PipelineOutput out."""
    hits = retrieve(question)
    resp = generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )
