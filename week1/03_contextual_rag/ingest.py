"""Contextual RAG, indexing path: note + chunk -> dense and BM25 vectors -> Qdrant.

  vectors  are made from note + chunk  -> what the search matches
  payload  keeps the original chunk     -> what the LLM reads (and what evidence recall checks)

Same two-vector layout as Hybrid, so the pipeline can search dense only or dense + BM25.
Also builds a CONTROL collection: the same 370-token chunks with no notes. Comparing the two separates
"the notes helped" from "smaller chunks helped". Needs the notes from contextualize.py. Medical corpus only.

Run from rag_sprints/:
    uv run week1/03_contextual_rag/ingest.py   # -> contextual_medical and chunks370_medical
"""

import importlib.util
import statistics
import time
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from contextualize import EMBED_MODEL, contextualized_text, load_chunks, load_notes, naive

# Load hybrid's ingest.py by path under another name (every sprint has a file called ingest.py)
_spec = importlib.util.spec_from_file_location(
    "hybrid_ingest", Path(__file__).resolve().parents[1] / "01_hybrid_rag" / "ingest.py")
hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid)

COLLECTION = f"contextual_{naive.CORPUS}"
CONTROL_COLLECTION = f"chunks370_{naive.CORPUS}"  # same chunks, no notes
DENSE, SPARSE, BM25_MODEL = hybrid.DENSE, hybrid.SPARSE, hybrid.BM25_MODEL


def index(collection: str, chunks: list, texts: list[str], notes: dict | None) -> QdrantClient:
    """Embed `texts` (dense + bm25) and store one point per chunk. The payload always holds the ORIGINAL chunk."""
    start = time.perf_counter()
    dense = [v.tolist() for v in TextEmbedding(EMBED_MODEL).passage_embed(texts, batch_size=32)]
    # BM25 length correction centred on the texts we actually index, as in Hybrid
    probe = SparseTextEmbedding(BM25_MODEL)
    avg_len = statistics.mean(len(hybrid.bm25_tokens(probe.model, t)) for t in texts)
    sparse = list(SparseTextEmbedding(BM25_MODEL, avg_len=avg_len).embed(texts))

    client = QdrantClient(url=naive.QDRANT_URL)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={DENSE: models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upload_points(
        collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector={DENSE: d, SPARSE: models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())},
                # text = the ORIGINAL chunk: the LLM never sees the note
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text}
                        | ({"note": notes[c.chunk_id]["note"]} if notes else {}),
            )
            for c, d, sp in zip(chunks, dense, sparse)
        ],
        batch_size=64,
        wait=True,
    )
    print(f"     '{collection}': {client.get_collection(collection).points_count} points, dense + bm25 "
          f"(avg_len {avg_len:.0f} words), {time.perf_counter() - start:.0f}s")
    return client


def main():
    chunks, notes = load_chunks(), load_notes()
    missing = [c.chunk_id for c in chunks if c.chunk_id not in notes or notes[c.chunk_id]["chunk_start"] != c.text[:80]]
    if missing:
        raise SystemExit(f"{len(missing)} chunks have no up-to-date note (e.g. {missing[0]}): run contextualize.py first")
    print(f"1. loaded: {len(chunks)} chunks (370 tokens) with a note each")

    print("2. indexing")
    client = index(COLLECTION, chunks, [contextualized_text(notes[c.chunk_id]["note"], c.text) for c in chunks], notes)
    index(CONTROL_COLLECTION, chunks, [c.text for c in chunks], None)

    point = client.retrieve(COLLECTION, [str(uuid.uuid5(uuid.NAMESPACE_URL, "hvmc_formulary::0002"))])[0]
    print(f"\n3. one stored point in '{COLLECTION}' (payload):")
    for key, value in point.payload.items():
        print(f"     {key:9s} {value[:110].replace(chr(10), ' | ')}{'…' if len(value) > 110 else ''}")


if __name__ == "__main__":
    main()
