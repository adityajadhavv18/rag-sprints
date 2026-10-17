"""Self-RAG, the loop: answer, read the answer back against the chunks, and fix it if the check fails.

  search      Contextual hybrid search (sprint 03), top 10: ranks 1-5 now, ranks 6-10 saved for a wider search
  [IsRel]     sprint 06's grader on ranks 1-5 -> keep relevant + partial (relevant first)
  answer      naive's generate() on the kept chunks
  [IsSup+Use] checker.py reads question + kept chunks + answer
  decide      fully + useful                  -> done
              not useful, or "I don't know"   -> wider search ONCE: grade ranks 6-10, add the keepers, answer again
              useful but not fully supported  -> rewrite, dropping the claims the checker listed (max 3 rewrites)
              still failing at a brake        -> return the last answer and flag it

The brakes: 1 wider search, 3 rewrites. Worst case is 5 answers + 5 checks; most questions stop after one check.

Run from rag_sprints/:  uv run week2/07_self_rag/loop.py
Runs the loop on 10 questions and prints every step. ~$0.02.
"""

from dataclasses import dataclass, field

from checker import check, crag, format_chunks
from eval_harness import settings, traceable
from eval_harness.golden import load_golden

MAX_REWRITES = 3
TOP_K = 5  # chunks to the answer step, as in every sprint
ctx = crag.load_contextual()
generate = ctx.hybrid.naive_pipeline.generate  # naive's answer step, unchanged since sprint 00
NO_ANSWER = ctx.hybrid.naive_pipeline.NO_ANSWER

# naive's answer prompt + the earlier answer and the checker's list. "Correct them" because a changed value
# (2047 vs 2074, 50 mL vs 25 mL) should be fixed, not just deleted.
REWRITE_PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "{no_answer}"
Be concise and include the specific figures from the context.

A checker found that an earlier answer states things the context does not say. Write a new answer: keep every fact \
the context states, and leave out these claims or correct them to what the context says:
{claims}

Earlier answer: {answer}

Context:
{context}

Question: {question}"""


@dataclass
class Step:
    kind: str  # "grade", "answer", "check", "widen", "rewrite"
    detail: dict


@dataclass
class SelfRagResult:
    answer: str
    hits: list  # the chunks the last answer was written from
    steps: list[Step] = field(default_factory=list)
    rewrites: int = 0
    widened: bool = False
    flag: str | None = None  # why the loop stopped without "fully + useful", if it did
    tokens_in: int = 0  # every grader, answer, check and rewrite call
    tokens_out: int = 0


def is_no_answer(answer: str) -> bool:
    return answer.strip().lower().startswith("i don't know")


@traceable(run_type="llm", name="rewrite_answer")
def rewrite(question: str, hits: list, answer: str, claims: list[str]):
    listed = "\n".join(f"- {c}" for c in claims) or "- (not listed: check every fact against the context)"
    return crag._openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(
            no_answer=NO_ANSWER, claims=listed, answer=answer, context=format_chunks(hits), question=question)}],
    )


@traceable(run_type="chain", name="self_rag")
def self_rag(question: str, max_rewrites: int = MAX_REWRITES) -> SelfRagResult:
    res = SelfRagResult(answer=NO_ANSWER, hits=[])

    def count(usage):
        res.tokens_in += usage.prompt_tokens
        res.tokens_out += usage.completion_tokens

    relevant, partial = [], []

    def grade_into_kept(hits: list, ranks: str):
        """[IsRel] with sprint 06's grader; relevant chunks go ahead of partial ones."""
        labels, _, usage = crag.grade(question, hits)
        count(usage)
        relevant.extend(h for h, lab in zip(hits, labels) if lab == "relevant")
        partial.extend(h for h, lab in zip(hits, labels) if lab == "partial")
        res.steps.append(Step("grade", {"ranks": ranks, "hits": hits, "labels": labels}))

    def answer_from_kept() -> bool:
        """Answer from the kept chunks. False when nothing is kept (no call; the answer stays "I don't know")."""
        res.hits = (relevant + partial)[:TOP_K]
        if not res.hits:
            res.answer = NO_ANSWER
            res.steps.append(Step("answer", {"text": NO_ANSWER, "skipped": "no chunks kept"}))
            return False
        resp = generate(question, res.hits)
        count(resp.usage)
        res.answer = resp.choices[0].message.content.strip()
        res.steps.append(Step("answer", {"text": res.answer}))
        return True

    top10 = ctx.hybrid_search(question, k=2 * TOP_K)
    grade_into_kept(top10[:TOP_K], "1-5")
    answer_from_kept()

    while True:
        verdict = None
        if not is_no_answer(res.answer):
            verdict, usage = check(question, res.hits, res.answer)
            count(usage)
            res.steps.append(Step("check", verdict))
            if verdict["supported"] == "fully" and verdict["useful"] == "yes":
                return res

        # Not useful, or "I don't know": look one level deeper, once
        if verdict is None or verdict["useful"] == "no":
            if not res.widened and top10[TOP_K:]:
                res.widened = True
                before = len(relevant) + len(partial)
                grade_into_kept(top10[TOP_K:], "6-10")
                res.steps.append(Step("widen", {"added": len(relevant) + len(partial) - before}))
                if len(relevant) + len(partial) > before:  # nothing new kept -> a new answer would be the same
                    answer_from_kept()
                    continue
            if verdict is None:
                return res  # still "I don't know" after the wider search: an honest refusal, no flag
            if verdict["supported"] == "fully":
                res.flag = "not useful"
                return res

        # Useful (or nothing wider to try) but not fully supported: rewrite without the unsupported claims
        if res.rewrites >= max_rewrites:
            res.flag = f"unsupported after {max_rewrites} rewrites"
            return res
        res.rewrites += 1
        resp = rewrite(question, res.hits, res.answer, verdict["unsupported_claims"])
        count(resp.usage)
        res.answer = resp.choices[0].message.content.strip()
        res.steps.append(Step("rewrite", {"n": res.rewrites, "text": res.answer}))


# ---------- Demo: every step, printed ----------

def show(res: SelfRagResult, evidence: str | None):
    for s in res.steps:
        d = s.detail
        if s.kind == "grade":
            marks = " ".join(f"{lab[:3]}{'*' if crag.holds_evidence(h.text, evidence) else ''}"
                             for h, lab in zip(d["hits"], d["labels"]))
            print(f"  [IsRel] ranks {d['ranks']:4}: {marks}")
        elif s.kind == "answer":
            print(f"  answer        : {d['text'][:150]}{'  (no call: ' + d['skipped'] + ')' if 'skipped' in d else ''}")
        elif s.kind == "check":
            claims = f"  unsupported: {d['unsupported_claims']}" if d["unsupported_claims"] else ""
            print(f"  [IsSup/IsUse] : {d['supported']} / {'useful' if d['useful'] == 'yes' else 'NOT useful'}{claims}")
        elif s.kind == "widen":
            print(f"  widen         : {d['added']} new chunk(s) kept{'' if d['added'] else ' -> no new answer'}")
        elif s.kind == "rewrite":
            print(f"  rewrite {d['n']}     : {d['text'][:150]}")


def main():
    # m11 (vague), m20 (evidence beyond rank 10), m04/m08/m09 (evidence graded "partial" in 06), m12/m26 (judge
    # noise in 06), m27/m29 (no answer), m01 (the 2047/2074 trap, should pass straight through)
    ids = ["m11", "m01", "m20", "m04", "m08", "m09", "m12", "m26", "m27", "m29"]
    golden = {g.id: g for g in load_golden("golden_medical_v1.jsonl")}
    tokens_in = tokens_out = 0
    for qid in ids:
        g = golden[qid]
        res = self_rag(g.question)
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        print(f"\n{qid} ({g.question_type}) {g.question}")
        print(f"  golden        : {g.ground_truth[:150]}")
        show(res, g.evidence)
        ev = any(crag.holds_evidence(h.text, g.evidence) for h in res.hits)
        print(f"  -> {len(res.steps)} steps | rewrites {res.rewrites} | widened {'yes' if res.widened else 'no'} | "
              f"evidence read: {'yes' if ev else 'no'}{'  | FLAG: ' + res.flag if res.flag else ''}")
    cost = (tokens_in * crag.PRICE_IN + tokens_out * crag.PRICE_OUT) / 1e6
    print(f"\nCost ${cost:.4f} for {len(ids)} questions (* = chunk holds the golden evidence)")


if __name__ == "__main__":
    main()
