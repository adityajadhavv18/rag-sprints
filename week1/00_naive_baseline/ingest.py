"""Naive baseline, indexing path: chunk -> embed -> store in Qdrant. Run once.

Naive on purpose: fixed-size windows that ignore sections, paragraphs and tables.
Later sprints (Parent-Child, Contextual) fix that, and this baseline measures by how much.

Run from rag_sprints/:  uv run week1/00_naive_baseline/ingest.py
"""

import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "naive_10k"  # one collection per architecture, so sprints never overwrite each other

# Measured in the EMBEDDING model's tokens: bge-small reads at most 512 tokens and silently drops the rest.
# 450 + 2 special tokens ([CLS], [SEP]) can never exceed 512. (500 OpenAI tokens overflowed on 30% of chunks.)
CHUNK_TOKENS = 450
OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    chunk_id: str  # "aapl_10k_2025::0042"
    doc_id: str  # becomes context_ids in PipelineOutput
    text: str
    n_tokens: int


def chunk_document(doc_id: str, text: str, tokenizer: Tokenizer) -> list[Chunk]:
    """Slide a fixed token window over the whole document, cutting wherever the window ends."""
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets  # (start_char, end_char) of each token in the original text
    step = CHUNK_TOKENS - OVERLAP_TOKENS

    chunks = []
    for i, start in enumerate(range(0, len(offsets), step)):
        end = min(start + CHUNK_TOKENS, len(offsets))
        # Slice the ORIGINAL text by character offsets: bge's tokenizer lowercases, so never decode its tokens
        chunk_text = text[offsets[start][0]: offsets[end - 1][1]]
        chunks.append(Chunk(f"{doc_id}::{i:04d}", doc_id, chunk_text, end - start))
        if end == len(offsets):
            break
    return chunks


def load_chunks() -> list[Chunk]:
    tokenizer = Tokenizer.from_pretrained(EMBED_MODEL)
    tokenizer.no_truncation()  # we need every token of the full document, not just the first 512
    return [c for path in sorted(DATA_DIR.glob("*.txt")) for c in chunk_document(path.stem, path.read_text(), tokenizer)]


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    # passage_embed (not embed) so the model applies its document-side formatting; queries use query_embed
    model = TextEmbedding(EMBED_MODEL)
    return [v.tolist() for v in model.passage_embed([c.text for c in chunks], batch_size=32)]


def store(chunks: list[Chunk], vectors: list[list[float]]) -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    # Recreate on every run so re-ingesting never leaves stale or duplicate chunks behind
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upload_points(
        COLLECTION,
        points=[
            models.PointStruct(
                # Qdrant ids must be ints or UUIDs; uuid5 turns the readable chunk_id into a stable UUID
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=vec,
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text},
            )
            for c, vec in zip(chunks, vectors)
        ],
        batch_size=64,
    )
    return client


def main():
    chunks = load_chunks()
    print(f"1. chunked: {len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")
    for doc_id in sorted({c.doc_id for c in chunks}):
        sizes = [c.n_tokens for c in chunks if c.doc_id == doc_id]
        print(f"     {doc_id}: {len(sizes):>4} chunks   tokens median {statistics.median(sizes):.0f}, max {max(sizes)}")

    start = time.perf_counter()
    vectors = embed_chunks(chunks)
    print(f"2. embedded: {len(vectors)} vectors x {len(vectors[0])} dims in {time.perf_counter() - start:.0f}s "
          f"(local {EMBED_MODEL}, free)")

    client = store(chunks, vectors)
    info = client.get_collection(COLLECTION)
    print(f"3. stored: {info.points_count} points in Qdrant collection '{COLLECTION}' "
          f"({info.config.params.vectors.distance.value} distance)")


if __name__ == "__main__":
    main()
