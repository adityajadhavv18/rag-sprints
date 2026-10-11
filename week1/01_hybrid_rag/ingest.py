"""Hybrid RAG, indexing path: the SAME chunks as naive, stored with two vectors each.

  dense  "meaning" vector  (bge-small, 384 numbers)    -> finds paraphrases, misses exact codes
  bm25   "keyword" vector  (sparse: word id -> weight) -> finds exact codes, misses paraphrases

Reusing naive's chunks means retrieval is the only thing that changes, so any score delta is Hybrid's doing.

Run from rag_sprints/:
    RAG_CORPUS=medical uv run week1/01_hybrid_rag/ingest.py   # -> Qdrant collection hybrid_medical
    uv run week1/01_hybrid_rag/ingest.py                      # 10-K corpus -> hybrid_10k
"""

import importlib.util
import statistics
import time
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.utils import remove_non_alphanumeric
from qdrant_client import QdrantClient, models

# Both sprints have a file called ingest.py, so load naive's by path under another name
_spec = importlib.util.spec_from_file_location(
    "naive_ingest", Path(__file__).resolve().parents[1] / "00_naive_baseline" / "ingest.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)

CORPUS, GOLDEN_SET, EMBED_MODEL, QDRANT_URL = naive.CORPUS, naive.GOLDEN_SET, naive.EMBED_MODEL, naive.QDRANT_URL
COLLECTION = f"hybrid_{CORPUS}"
BM25_MODEL = "Qdrant/bm25"
DENSE, SPARSE = "dense", "bm25"  # names of the two vectors stored on every point


def bm25_tokens(model, text: str) -> list[str]:
    """The exact words BM25 indexes: punctuation split, lowercased, stopwords dropped, stemmed."""
    return model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text)))


def build_bm25(chunks: list[naive.Chunk]):
    """BM25 shrinks the weight of words in long chunks relative to the AVERAGE chunk length (avg_len).
    FastEmbed assumes 256 words; measure ours so the length correction is centred on our real chunks."""
    probe = SparseTextEmbedding(BM25_MODEL)
    avg_len = statistics.mean(len(bm25_tokens(probe.model, c.text)) for c in chunks)
    return SparseTextEmbedding(BM25_MODEL, avg_len=avg_len), avg_len


def store(chunks, dense_vectors, sparse_vectors) -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=len(dense_vectors[0]), distance=models.Distance.COSINE)},
        # FastEmbed's BM25 vectors hold only the term-frequency half of the formula. Modifier.IDF makes Qdrant
        # supply the other half (how RARE each word is across the collection) at query time.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upload_points(
        COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector={DENSE: dense, SPARSE: models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())},
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text},
            )
            for c, dense, sp in zip(chunks, dense_vectors, sparse_vectors)
        ],
        batch_size=64,
        wait=True,  # block until indexed, so the count printed below is final
    )
    return client


def show_sparse_vector(bm25, chunk: naive.Chunk, top: int = 12):
    """Print one chunk the way BM25 sees it: word -> weight (Qdrant only stores the hashed word ids)."""
    sp = next(bm25.embed([chunk.text]))
    weight_by_id = dict(zip(sp.indices.tolist(), sp.values.tolist()))
    words = dict.fromkeys(bm25_tokens(bm25.model, chunk.text))
    rows = sorted(((weight_by_id[bm25.model.compute_token_id(w)], w) for w in words), reverse=True)
    print(f"\n   one chunk as a BM25 sparse vector ({chunk.chunk_id}): {len(sp.indices)} non-zero entries "
          f"out of ~2 billion possible word ids")
    print(f"   text starts: {chunk.text[:90]!r}")
    for weight, word in rows[:top]:
        print(f"     {word:<16} id {bm25.model.compute_token_id(word):>10}   weight {weight:.2f}")
    print("     ...")


def main():
    chunks = naive.load_chunks()
    print(f"1. chunks: {len(chunks)} from {len({c.doc_id for c in chunks})} documents (same as naive_{CORPUS})")

    start = time.perf_counter()
    dense_vectors = [v.tolist() for v in TextEmbedding(EMBED_MODEL).passage_embed([c.text for c in chunks], batch_size=32)]
    print(f"2. dense:  {len(dense_vectors)} x {len(dense_vectors[0])} numbers in {time.perf_counter() - start:.0f}s")

    bm25, avg_len = build_bm25(chunks)
    sparse_vectors = list(bm25.embed([c.text for c in chunks]))
    nonzero = [len(s.indices) for s in sparse_vectors]
    print(f"3. bm25:   {len(sparse_vectors)} sparse vectors, median {statistics.median(nonzero):.0f} distinct words each "
          f"(avg chunk length after stopwords: {avg_len:.0f} words, FastEmbed default 256)")

    client = store(chunks, dense_vectors, sparse_vectors)
    print(f"4. stored: {client.count(COLLECTION, exact=True).count} points in '{COLLECTION}', "
          f"each with a '{DENSE}' and a '{SPARSE}' vector")

    # Show the chunk holding a code that dense search ranked #74 for m01 (only meaningful on the medical corpus)
    demo = next((c for c in chunks if "HV-RX-2074" in c.text), chunks[0])
    show_sparse_vector(bm25, demo)
    print(f"\n   how codes tokenize:  'HV-RX-2074' -> {bm25_tokens(bm25.model, 'HV-RX-2074')}   "
          f"'E-71' -> {bm25_tokens(bm25.model, 'E-71')}   'CP-09' -> {bm25_tokens(bm25.model, 'CP-09')}")


if __name__ == "__main__":
    main()
