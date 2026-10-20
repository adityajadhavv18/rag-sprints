"""Phase 2: embed the 6 images with CLIP's image tower and store them in Qdrant (`multimodal_medical`).

Only pixels are embedded. The manifest's hidden fact goes into the payload for scoring, never into the vector.

uv run week2/11_multimodal_rag/build_index.py
"""

import json
import uuid
from pathlib import Path

from fastembed import ImageEmbedding
from qdrant_client import QdrantClient, models

HERE = Path(__file__).parent
IMAGES = HERE / "images"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "multimodal_medical"
IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"  # CLIP's image tower, 512-d
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"     # CLIP's text tower: same 512-d space, used for questions


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in (IMAGES / "manifest.jsonl").read_text().splitlines()]


def main():
    manifest = load_manifest()
    vectors = [v.tolist() for v in ImageEmbedding(IMAGE_MODEL).embed([str(IMAGES / m["file"]) for m in manifest])]

    client = QdrantClient(url=QDRANT_URL)
    # Recreate on every run so re-indexing never leaves stale images behind
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
    )
    client.upload_points(
        COLLECTION,
        points=[
            models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, m["id"])), vector=vec,
                               payload={"modality": "image", **m})
            for m, vec in zip(manifest, vectors)
        ],
    )
    print(f"indexed {client.count(COLLECTION).count} images into '{COLLECTION}' ({len(vectors[0])}-d, {IMAGE_MODEL})")


if __name__ == "__main__":
    main()
