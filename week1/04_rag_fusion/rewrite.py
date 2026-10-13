"""RAG Fusion, step 1: turn one question into 4 rephrasings, so the search can ask 5 times in different words.

  question     = what the user typed, often in lay words ("arm vein", "kidneys not working")
  rephrasings  = 4 versions from gpt-4o-mini; at least one in clinical words ("peripheral line", "renal")
  used for     = SEARCH only. The LLM that answers still sees the original question.

The model sees only the question, never the documents, so it can't copy the answer into a rephrasing.
Rephrasings are cached in rewrites_medical.json (keyed by question text), so retrieval experiments are free.
Medical corpus only.

Run from rag_sprints/:
    uv run week1/04_rag_fusion/rewrite.py --sample   # 5 hand-picked questions, printed in full
    uv run week1/04_rag_fusion/rewrite.py            # all 30 golden questions
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI

from eval_harness import settings  # importing it loads .env, so OpenAI() finds the key
from eval_harness.golden import load_golden

os.environ.setdefault("RAG_CORPUS", "medical")
if os.environ["RAG_CORPUS"] != "medical":
    raise SystemExit("RAG Fusion runs on the medical corpus only: unset RAG_CORPUS or set it to medical")

GOLDEN_SET = "golden_medical_v1.jsonl"  # a name; the harness finds it in eval_harness/golden_sets/
N_REPHRASINGS = 4  # + the original = 5 queries
CACHE_FILE = Path(__file__).with_name("rewrites_medical.json")
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens

# Three rules for this corpus: keep codes exact (BM25 needs them), add clinical words (the vocabulary gap),
# and don't add facts (a rephrasing that invents an answer would search for the wrong thing).
# The example terms are deliberately NOT from the golden set, so the prompt can't leak an answer to the test.
PROMPT = """You help a search engine over hospital documents: a drug formulary, a lab test catalog, \
clinical protocols and infusion pump manuals.
Rewrite the question below in {n} different ways that ask for the same thing.
- Keep every code, drug name and device name exactly as written (for example HV-RX-2074, E-71, CP-09, Kestrel IV-220).
- At least one version must use the clinical terms a hospital document would use \
(for example "hypertension" for "high blood pressure", "pulmonary embolism" for "blood clot in the lung").
- Make the versions differ in wording, not in meaning. Do not add facts, numbers or answers.
Reply as JSON: {{"rephrasings": ["...", "..."]}}

Question: {question}"""


def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def ask_model(client: OpenAI, question: str) -> tuple[list[str], dict]:
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,  # same question -> same rephrasings, so runs are comparable
        response_format={"type": "json_object"},  # forces valid JSON, so json.loads can't fail on stray text
        messages=[{"role": "user", "content": PROMPT.format(n=N_REPHRASINGS, question=question)}],
    )
    rephrasings = json.loads(resp.choices[0].message.content)["rephrasings"][:N_REPHRASINGS]
    return rephrasings, {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}


def rephrase(question: str, client: OpenAI | None = None) -> list[str]:
    """The 4 rephrasings for one question: from the cache if we have them, otherwise ask and save."""
    cache = load_cache()
    if question not in cache:
        cache[question], _ = ask_model(client or OpenAI(), question)
        CACHE_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return cache[question]


def main():
    golden = load_golden(GOLDEN_SET)
    # Sample: three vocabulary gaps (m11, m13, m20), a code that must survive (m01), and a no-answer
    # question that must not turn answerable (m27)
    if "--sample" in sys.argv:
        golden = [g for g in golden if g.id in ("m01", "m11", "m13", "m20", "m27")]

    cache, client = load_cache(), OpenAI()
    usage = {"in": 0, "out": 0, "calls": 0}
    for g in golden:
        if g.question in cache:
            continue
        cache[g.question], u = ask_model(client, g.question)
        CACHE_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))  # save each one: a crash loses nothing
        usage["in"] += u["in"]
        usage["out"] += u["out"]
        usage["calls"] += 1

    cost = (usage["in"] * PRICE_IN + usage["out"] * PRICE_OUT) / 1e6
    print(f"{usage['calls']} new calls ({usage['in']:,} in, {usage['out']:,} out tokens) cost ${cost:.4f}. "
          f"{len(cache)} questions in {CACHE_FILE.name}\n")
    for g in golden:
        print(f"[{g.id}] {g.question}")
        for i, r in enumerate(cache[g.question], start=1):
            print(f"   {i}. {r}")
        print()


if __name__ == "__main__":
    main()
