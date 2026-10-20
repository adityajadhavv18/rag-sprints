"""Phase 3: question -> CLIP finds the top images -> gpt-4o-mini LOOKS at them and answers.

If even the best image scores below MIN_SCORE, refuse in code without calling the LLM
(sprint 09's lesson: an empty or off-topic context is not a reliable "I don't know").

uv run week2/11_multimodal_rag/pipeline.py      # the 7 sprint-local questions (~$0.005)
"""

import base64
import sys
from functools import cache
from pathlib import Path

from eval_harness import PipelineOutput, settings, traceable, traced_openai
from eval_harness.metrics import cost_usd

sys.path.insert(0, str(Path(__file__).parent))
from build_index import IMAGES  # noqa: E402
from retrieve import load_questions, search_images  # noqa: E402

TOP_K = 2            # the vision model sees 2 images, as a text pipeline reads several chunks
MIN_SCORE = 0.25     # rough: set from only 7 questions (answerable best 0.268+, no-answer 0.222)
NO_ANSWER = "I don't know based on the provided documents."  # same wording as naive's generate()
# "high": full-size tiles, ~10x the tokens. "low" (one 512 px view) misread colours: purple band -> "fall risk"
DETAIL = "high"

PROMPT = """Answer the question using ONLY the images provided.
If the answer is not shown in the images, reply exactly: "{no_answer}"
Be concise and name which image the answer comes from.

Question: {question}"""


@cache
def _llm():
    return traced_openai()


def _image_part(image_id: str) -> dict:
    path = IMAGES / f"{image_id}_*.png"
    file = next(IMAGES.glob(path.name))
    data = base64.b64encode(file.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": DETAIL}}


@traceable(run_type="chain", name="generate_from_images")
def generate(question: str, image_ids: list[str]):
    content = [{"type": "text", "text": PROMPT.format(no_answer=NO_ANSWER, question=question)}]
    for i, image_id in enumerate(image_ids, start=1):
        content += [{"type": "text", "text": f"[{i}] image {image_id}"}, _image_part(image_id)]
    return _llm().chat.completions.create(
        model=settings.llm_model, temperature=0, messages=[{"role": "user", "content": content}]
    )


@traceable(run_type="chain", name="multimodal_pipeline")
def multimodal_pipeline(question: str) -> PipelineOutput:
    hits = search_images(question, k=TOP_K)
    if hits[0][1] < MIN_SCORE:
        return PipelineOutput(answer=NO_ANSWER, contexts=[], context_ids=[])
    image_ids = [i for i, _ in hits]
    resp = generate(question, image_ids)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[f"[image: {i}]" for i in image_ids],  # the harness expects text; images are named, not inlined
        context_ids=image_ids,
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )


def main():
    right = total_in = total_out = 0
    questions = load_questions()
    for q in questions:
        out = multimodal_pipeline(q["question"])
        total_in, total_out = total_in + out.tokens_in, total_out + out.tokens_out
        if q["key_fact"]:
            ok = q["key_fact"].lower() in out.answer.lower()
        else:
            ok = out.answer == NO_ANSWER
        right += ok
        print(f"{'✅' if ok else '❌'} {q['id']} {q['question']}")
        print(f"     saw {out.context_ids or 'nothing (refused in code)'} | expected '{q['key_fact'] or 'refusal'}'")
        print(f"     {out.answer}\n")
    cost = cost_usd(total_in, total_out, settings.llm_model) or 0.0
    print(f"Correct: {right}/{len(questions)}   tokens in {total_in:,} / out {total_out:,}   "
          f"cost ${cost:.4f} (${cost / len(questions):.5f}/query)")


if __name__ == "__main__":
    main()
