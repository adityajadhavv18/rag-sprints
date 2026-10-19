"""Speculative RAG, the children: search top 9 -> deal into 3 piles of 3 -> 3 gpt-4o-mini drafts at the same time.

  split      round-robin by rank: pile 1 = chunks 1,4,7, pile 2 = 2,5,8, pile 3 = 3,6,9 (each pile gets a strong chunk)
  draft      one child reads ONLY its 3 chunks and writes the full answer + a one-line rationale quoting its chunk
  confidence how sure the child was while writing: exp(mean token logprob), free from the same call (0..1)
  draft_all  the 3 children in parallel threads, so the wait is the slowest child, not the sum

Demo (<$0.01): uv run week2/12_speculative_rag/drafter.py   (from rag_sprints/)
"""

import importlib.util
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from langsmith.utils import ContextThreadPoolExecutor  # threads that keep LangSmith's parent trace

from eval_harness import traceable, traced_openai

DRAFTER_MODEL = "gpt-4o-mini"
SEARCH_K = 9  # chunks from Contextual search
N_DRAFTS = 3  # children, each reading SEARCH_K / N_DRAFTS chunks

DRAFT_PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly:
Answer: {no_answer}
Rationale: none

Otherwise reply in exactly this format:
Answer: <the full answer, concise, with the specific figures from the context>
Rationale: <one sentence: which chunk [n] supports the answer, quoting its key words>

Context:
{context}

Question: {question}"""


def load_contextual():
    """Sprint 03's search, loaded the same way earlier sprints do (its folder on the path, its file by path)."""
    os.environ.setdefault("RAG_CORPUS", "medical")
    if os.environ["RAG_CORPUS"] != "medical":
        raise SystemExit("Speculative RAG runs on the medical corpus only: unset RAG_CORPUS or set it to medical")
    ctx_dir = Path(__file__).resolve().parents[2] / "week1" / "03_contextual_rag"
    sys.path.append(str(ctx_dir))
    spec = importlib.util.spec_from_file_location("contextual_pipeline", ctx_dir / "pipeline.py")
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    return ctx


ctx = load_contextual()
NO_ANSWER = ctx.hybrid.naive_pipeline.NO_ANSWER


@cache
def _llm():
    return traced_openai()


@dataclass
class Draft:
    answer: str
    rationale: str
    confidence: float  # the child's own certainty, 0..1
    hits: list  # the 3 chunks this child read
    tokens_in: int
    tokens_out: int
    ms: float
    refused: bool = field(init=False)

    def __post_init__(self):
        self.refused = NO_ANSWER.lower() in self.answer.lower()


def search(question: str, k: int = SEARCH_K) -> list:
    return ctx.hybrid_search(question, k=k)


def split(hits: list, n: int = N_DRAFTS) -> list[list]:
    """Deal the chunks like cards: rank 1 to pile 1, rank 2 to pile 2, ... so every pile gets one strong chunk."""
    return [hits[i::n] for i in range(n)]


def _parse(text: str) -> tuple[str, str]:
    m = re.search(r"Answer:\s*(.*?)\s*(?:Rationale:\s*(.*))?$", text, re.S)
    return (m.group(1).strip(), (m.group(2) or "").strip()) if m else (text.strip(), "")


@traceable(run_type="llm", name="speculative_draft")
def draft(question: str, hits: list) -> Draft:
    t0 = time.perf_counter()
    context = "\n\n".join(f"[{i}] ({h.doc_id})\n{h.text}" for i, h in enumerate(hits, start=1))
    resp = _llm().chat.completions.create(
        model=DRAFTER_MODEL,
        temperature=0,
        logprobs=True,  # each token's log-probability: the child's confidence comes from these
        messages=[{"role": "user", "content": DRAFT_PROMPT.format(no_answer=NO_ANSWER, context=context,
                                                                  question=question)}],
    )
    choice = resp.choices[0]
    lps = [t.logprob for t in choice.logprobs.content]
    answer, rationale = _parse(choice.message.content)
    return Draft(answer, rationale, math.exp(sum(lps) / len(lps)), hits,
                 resp.usage.prompt_tokens, resp.usage.completion_tokens, (time.perf_counter() - t0) * 1000)


@traceable(run_type="chain", name="speculative_draft_all")
def draft_all(question: str, hits: list) -> list[Draft]:
    """All children at once. Wall time = the slowest child."""
    with ContextThreadPoolExecutor(max_workers=N_DRAFTS) as pool:
        return list(pool.map(lambda pile: draft(question, pile), split(hits)))


# ---------- Demo: 3 questions, what each child read and wrote ----------

def main():
    from eval_harness.golden import load_golden

    sys.path.append(str(Path(__file__).resolve().parents[1] / "06_corrective_rag"))
    from grader import holds_evidence

    golden = {g.id: g for g in load_golden("golden_medical_v1.jsonl")}
    for qid in ["m01", "m21", "m28"]:  # one exact-term, one multi-part, one no-answer
        g = golden[qid]
        hits = search(g.question)
        print(f"\n{'=' * 100}\n{qid} ({g.question_type}): {g.question}")
        ranks = {h.chunk_id: r for r, h in enumerate(hits, start=1)}
        ev = [r for r, h in enumerate(hits, start=1) if holds_evidence(h.text, g.evidence)]
        print(f"evidence at search ranks: {ev or 'none'}")

        t0 = time.perf_counter()
        drafts = draft_all(g.question, hits)
        wall = (time.perf_counter() - t0) * 1000

        for i, d in enumerate(drafts, start=1):
            pile = ", ".join(f"#{ranks[h.chunk_id]} {h.doc_id}{' ★' if holds_evidence(h.text, g.evidence) else ''}"
                             for h in d.hits)
            print(f"\n  child {i}  reads [{pile}]")
            print(f"    answer     {d.answer}")
            print(f"    rationale  {d.rationale or '-'}")
            print(f"    confidence {d.confidence:.3f}   {d.tokens_in} in / {d.tokens_out} out   {d.ms:.0f} ms"
                  f"{'   (refused)' if d.refused else ''}")
        print(f"\n  3 children in parallel: {wall:.0f} ms wall  vs  {sum(d.ms for d in drafts):.0f} ms if one by one")


if __name__ == "__main__":
    main()
