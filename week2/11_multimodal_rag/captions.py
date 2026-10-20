"""Phase 4: the other way to do multimodal RAG: describe each image in words ONCE, then it's plain text RAG.

Index:  each image -> gpt-4o-mini (high detail) writes a caption -> captions_medical.json (paid once, cached)
Query:  question -> bge (naive's embedder) over the 6 captions -> top 2 captions as TEXT -> naive's generate()
No image is sent at question time.

uv run week2/11_multimodal_rag/captions.py --build   # write the 6 captions (~$0.025, then free from cache)
uv run week2/11_multimodal_rag/captions.py           # answer the 7 questions from captions (<$0.001)
"""

import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np

from eval_harness import PipelineOutput, settings, traceable
from eval_harness.metrics import cost_usd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_index import load_manifest  # noqa: E402
from pipeline import _image_part, _llm  # noqa: E402
from retrieve import load_questions  # noqa: E402

CAPTIONS = HERE / "captions_medical.json"
TOP_K = 2  # same as the image pipeline

CAPTION_PROMPT = """Describe this image so it can be found and used by text search.
Transcribe every word and number exactly. State every fact the picture shows that the words alone don't:
which colour means what, what is where (top-left, bottom-right...), the order of things, which bar is highest or lowest.
Plain sentences, no preamble."""


def load_naive():
    """Naive's embedder and generate(), loaded by path because the folder name starts with a digit."""
    os.environ.setdefault("RAG_CORPUS", "medical")
    naive_dir = HERE.parents[1] / "week1" / "00_naive_baseline"
    sys.path.append(str(naive_dir))
    spec = importlib.util.spec_from_file_location("naive_pipeline", naive_dir / "pipeline.py")
    naive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(naive)
    return naive


def build():
    rows, tokens_in, tokens_out = [], 0, 0
    for m in load_manifest():
        resp = _llm().chat.completions.create(
            model=settings.llm_model, temperature=0,
            messages=[{"role": "user", "content": [{"type": "text", "text": CAPTION_PROMPT}, _image_part(m["id"])]}],
        )
        tokens_in, tokens_out = tokens_in + resp.usage.prompt_tokens, tokens_out + resp.usage.completion_tokens
        rows.append({"id": m["id"], "caption": resp.choices[0].message.content.strip(),
                     "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens})
        print(f"--- {m['id']} ({resp.usage.prompt_tokens:,} tokens in)\n{rows[-1]['caption']}\n")
    CAPTIONS.write_text(json.dumps(rows, indent=2))
    print(f"captioned {len(rows)} images: {tokens_in:,} in / {tokens_out:,} out, "
          f"${cost_usd(tokens_in, tokens_out, settings.llm_model):.4f} once")


@cache
def _index():
    """6 captions fit in memory: a numpy matrix is the whole 'vector store'."""
    naive = load_naive()
    rows = json.loads(CAPTIONS.read_text())
    vectors = np.array([v for v in naive._embedder().passage_embed([r["caption"] for r in rows])])
    return naive, rows, vectors


def search_captions(question: str, k: int = TOP_K) -> list[tuple[dict, float]]:
    naive, rows, vectors = _index()
    q = next(naive._embedder().query_embed(question))
    scores = vectors @ q  # bge vectors are normalised, so the dot product is the cosine
    return [(rows[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


@traceable(run_type="chain", name="caption_pipeline")
def caption_pipeline(question: str) -> PipelineOutput:
    naive = load_naive()
    hits = [naive.Hit(r["id"], r["id"], r["caption"], s) for r, s in search_captions(question)]
    resp = naive.generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )


def main():
    if "--build" in sys.argv:
        build()
        return
    naive = load_naive()
    right = tokens_in = tokens_out = 0
    questions = load_questions()
    for q in questions:
        out = caption_pipeline(q["question"])
        tokens_in, tokens_out = tokens_in + out.tokens_in, tokens_out + out.tokens_out
        ok = (q["key_fact"].lower() in out.answer.lower()) if q["key_fact"] else out.answer == naive.NO_ANSWER
        right += ok
        print(f"{'✅' if ok else '❌'} {q['id']} {q['question']}")
        print(f"     read {out.context_ids} | expected {q['expected_id'] or 'refusal'} / '{q['key_fact'] or ''}'")
        print(f"     {out.answer}\n")
    cost = cost_usd(tokens_in, tokens_out, settings.llm_model) or 0.0
    print(f"Correct: {right}/{len(questions)}   tokens in {tokens_in:,} / out {tokens_out:,}   "
          f"cost ${cost:.4f} (${cost / len(questions):.5f}/query)")


if __name__ == "__main__":
    main()
