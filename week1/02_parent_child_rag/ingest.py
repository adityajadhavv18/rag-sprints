"""Parent-Child RAG, indexing path: embed the CHILDREN, store each with its parent's text in the payload.

  search runs on   -> child vectors (one line each, sharp match)
  LLM later reads  -> payload["parent_text"] (the whole entry that line came from)

Each child gets both vectors Hybrid uses (dense + bm25), so one collection serves a dense-only
and a hybrid search later. Parents are never embedded.

Run from rag_sprints/:
    uv run week1/02_parent_child_rag/ingest.py   # -> Qdrant collection parent_child_medical
"""

import importlib.util
import json
import statistics
import time
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

import chunking  # sets RAG_CORPUS=medical before naive's module is loaded below

# Reuse Hybrid's BM25 setup (measured avg_len) instead of copying it
_spec = importlib.util.spec_from_file_location(
    "hybrid_ingest", Path(__file__).resolve().parents[1] / "01_hybrid_rag" / "ingest.py")
hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid)

EMBED_MODEL, QDRANT_URL = chunking.EMBED_MODEL, chunking.naive.QDRANT_URL
COLLECTION = "parent_child_medical"
DENSE, SPARSE = hybrid.DENSE, hybrid.SPARSE
EXAMPLE_CHILD = "hvmc_formulary::p006::c04"  # Corvetrol's renal line: m01's evidence


def store(children, parents_by_id, dense_vectors, sparse_vectors) -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=len(dense_vectors[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upload_points(
        COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.child_id)),
                vector={DENSE: dense, SPARSE: models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())},
                payload={
                    "child_id": c.child_id,
                    "parent_id": c.parent_id,
                    "doc_id": c.doc_id,
                    "text": c.text,  # what was embedded
                    "parent_text": parents_by_id[c.parent_id].text,  # what the LLM will read
                },
            )
            for c, dense, sp in zip(children, dense_vectors, sparse_vectors)
        ],
        batch_size=64,
        wait=True,
    )
    return client


def show_point(client: QdrantClient, child_id: str):
    """Print one stored point the way Qdrant returns it (vectors shortened so it fits on screen)."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, child_id))
    point = client.retrieve(COLLECTION, ids=[point_id], with_vectors=True, with_payload=True)[0]
    dense, sparse = point.vector[DENSE], point.vector[SPARSE]
    shown = {
        "collection": COLLECTION,
        "id": point.id,
        "vector": {
            DENSE: [round(x, 4) for x in dense[:6]] + [f"... {len(dense) - 6} more numbers ({len(dense)} total)"],
            SPARSE: {"indices": sparse.indices, "values": [round(v, 3) for v in sparse.values]},
        },
        "payload": point.payload,
    }
    print(json.dumps(shown, indent=2))


def main():
    parents, children = chunking.load_parents_and_children()
    parents_by_id = {p.parent_id: p for p in parents}
    print(f"1. split: {len(children)} children from {len(parents)} parents")

    start = time.perf_counter()
    dense_vectors = [v.tolist() for v in TextEmbedding(EMBED_MODEL).passage_embed([c.text for c in children], batch_size=32)]
    print(f"2. dense:  {len(dense_vectors)} x {len(dense_vectors[0])} numbers in {time.perf_counter() - start:.0f}s")

    bm25, avg_len = hybrid.build_bm25(children)
    sparse_vectors = list(bm25.embed([c.text for c in children]))
    print(f"3. bm25:   {len(sparse_vectors)} sparse vectors, median "
          f"{statistics.median(len(s.indices) for s in sparse_vectors):.0f} distinct words each "
          f"(avg child length after stopwords: {avg_len:.1f} words; naive chunks: 202)")

    client = store(children, parents_by_id, dense_vectors, sparse_vectors)
    print(f"4. stored: {client.count(COLLECTION, exact=True).count} points in '{COLLECTION}' "
          f"(one per child, each with '{DENSE}' + '{SPARSE}' vectors and its parent's text)\n")

    show_point(client, EXAMPLE_CHILD)


if __name__ == "__main__":
    main()
