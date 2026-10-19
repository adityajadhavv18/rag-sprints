"""Speculative RAG, the parent: gpt-4o judges the children's answers and never writes one.

  verify  the parent reads question + answer + rationale (NOT the chunks) and replies with ONE token, Yes or No.
          Its score is P(Yes) from that token's logprobs, so the big model reads ~150 tokens and writes 1.
  pick    refusals are dropped; the rest are verified in parallel; score = child's confidence x parent's P(Yes);
          the highest score wins. All children refused -> refuse, and the parent is never called.

Demo (<$0.01): uv run week2/12_speculative_rag/verifier.py   (from rag_sprints/)
"""

import math
import time
from dataclasses import dataclass

from langsmith.utils import ContextThreadPoolExecutor

from drafter import N_DRAFTS, Draft, _llm, draft_all, search
from eval_harness import traceable

VERIFIER_MODEL = "gpt-4o"

VERIFY_PROMPT = """Question: {question}
Answer: {answer}
Rationale: {rationale}

Does the rationale support the answer, and does the answer address the question? Reply with one word: Yes or No."""


@dataclass
class Verdict:
    draft: Draft
    p_yes: float  # the parent's score, 0..1
    tokens_in: int
    tokens_out: int
    ms: float

    @property
    def score(self) -> float:
        return self.draft.confidence * self.p_yes


@traceable(run_type="llm", name="speculative_verify")
def verify(question: str, d: Draft) -> Verdict:
    t0 = time.perf_counter()
    resp = _llm().chat.completions.create(
        model=VERIFIER_MODEL,
        temperature=0,
        max_tokens=1,  # one word: the parent judges, it doesn't write
        logprobs=True,
        top_logprobs=5,  # the 5 likeliest first tokens with their probabilities: "Yes", "No", "yes", ...
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(question=question, answer=d.answer,
                                                                   rationale=d.rationale)}],
    )
    top = resp.choices[0].logprobs.content[0].top_logprobs
    p_yes = sum(math.exp(t.logprob) for t in top if t.token.strip().lower() == "yes")
    return Verdict(d, p_yes, resp.usage.prompt_tokens, resp.usage.completion_tokens,
                   (time.perf_counter() - t0) * 1000)


@traceable(run_type="chain", name="speculative_pick")
def pick(question: str, drafts: list[Draft]) -> list[Verdict]:
    """Verdicts for the non-refused drafts, best first. Empty list = every child refused."""
    candidates = [d for d in drafts if not d.refused]
    if not candidates:
        return []
    with ContextThreadPoolExecutor(max_workers=N_DRAFTS) as pool:
        verdicts = list(pool.map(lambda d: verify(question, d), candidates))
    return sorted(verdicts, key=lambda v: v.score, reverse=True)


# ---------- Demo: the same 3 questions, scores and a time breakdown ----------

def main():
    from eval_harness.golden import load_golden

    golden = {g.id: g for g in load_golden("golden_medical_v1.jsonl")}
    for qid in ["m01", "m21", "m28"]:
        g = golden[qid]
        print(f"\n{'=' * 100}\n{qid}: {g.question}")
        t0 = time.perf_counter()
        hits = search(g.question)
        t1 = time.perf_counter()
        drafts = draft_all(g.question, hits)
        t2 = time.perf_counter()
        verdicts = pick(g.question, drafts)
        t3 = time.perf_counter()

        for i, d in enumerate(drafts, start=1):
            v = next((v for v in verdicts if v.draft is d), None)
            line = "refused, dropped" if v is None else \
                f"child {d.confidence:.3f} x parent {v.p_yes:.3f} = {v.score:.3f}"
            win = "  ← WINNER" if verdicts and v is verdicts[0] else ""
            print(f"  child {i}: {line}{win}\n           {d.answer[:140]}")
        print(f"  final: {verdicts[0].draft.answer if verdicts else 'refuse (all children refused, parent not called)'}")
        par_in = sum(v.tokens_in for v in verdicts)
        print(f"  time: search {(t1 - t0) * 1000:.0f} ms | children {(t2 - t1) * 1000:.0f} ms | "
              f"parent {(t3 - t2) * 1000:.0f} ms | total {(t3 - t0) * 1000:.0f} ms")
        print(f"  parent read {par_in} tokens in total and wrote {sum(v.tokens_out for v in verdicts)}")


if __name__ == "__main__":
    main()
