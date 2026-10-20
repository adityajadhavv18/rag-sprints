"""Phase 2: text question -> CLIP text tower -> nearest images. Plus a look at the modality gap. Free, all local.

uv run week2/11_multimodal_rag/retrieve.py
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
from fastembed import ImageEmbedding, TextEmbedding
from qdrant_client import QdrantClient

from build_index import COLLECTION, IMAGE_MODEL, IMAGES, QDRANT_URL, TEXT_MODEL, load_manifest

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions_multimodal.jsonl"


@cache
def _text_model() -> TextEmbedding:
    return TextEmbedding(TEXT_MODEL)


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def search_images(question: str, k: int = 3) -> list[tuple[str, float]]:
    """(image id, cosine score) for the k nearest images."""
    vector = next(iter(_text_model().embed([question]))).tolist()
    hits = _qdrant().query_points(COLLECTION, query=vector, limit=k).points
    return [(h.payload["id"], h.score) for h in hits]


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines()]


def unit(vectors) -> np.ndarray:
    m = np.array(list(vectors))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def modality_gap(manifest: list[dict]):
    """Same image's title as text vs as picture: does CLIP put matching pairs close together?"""
    images = unit(ImageEmbedding(IMAGE_MODEL).embed([str(IMAGES / m["file"]) for m in manifest]))
    titles = unit(_text_model().embed([m["title"] for m in manifest]))
    off = ~np.eye(len(manifest), dtype=bool)
    print("\nModality gap (cosine):")
    print(f"  title  vs ITS OWN image      {np.mean(np.diag(titles @ images.T)):.2f}   <- same meaning, different modality")
    print(f"  title  vs a DIFFERENT title  {np.mean((titles @ titles.T)[off]):.2f}   <- different meaning, same modality")
    print(f"  image  vs a DIFFERENT image  {np.mean((images @ images.T)[off]):.2f}")


def main():
    questions = load_questions()
    hit1 = hit3 = 0
    answerable_best = []
    print(f"CLIP retrieval over {_qdrant().count(COLLECTION).count} images ({TEXT_MODEL} -> {IMAGE_MODEL})\n")
    for q in questions:
        hits = search_images(q["question"])
        ids = [i for i, _ in hits]
        if q["expected_id"]:
            ok1, ok3 = ids[0] == q["expected_id"], q["expected_id"] in ids
            hit1, hit3 = hit1 + ok1, hit3 + ok3
            answerable_best.append(hits[0][1])
            mark = "✅" if ok1 else ("⚠️ " if ok3 else "❌")
        else:
            mark = "—"
        print(f"{mark} {q['id']} {q['question']}")
        print(f"     expected {q['expected_id'] or 'nothing'} | got " + "  ".join(f"{i} {s:.3f}" for i, s in hits))

    n = sum(1 for q in questions if q["expected_id"])
    print(f"\nRight image at #1: {hit1}/{n}   in top 3: {hit3}/{n}")
    no_answer_best = [search_images(q["question"])[0][1] for q in questions if not q["expected_id"]]
    print(f"Best score on answerable Qs: {min(answerable_best):.3f}–{max(answerable_best):.3f}   "
          f"on the no-answer Q: {', '.join(f'{s:.3f}' for s in no_answer_best)}")
    modality_gap(load_manifest())


if __name__ == "__main__":
    main()
