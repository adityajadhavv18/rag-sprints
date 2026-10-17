"""Self-RAG, the checker: two gpt-4o-mini calls, run at the same time, judge the WRITTEN answer.

  [IsSup]  supported   fully / partially / no: is every claim in the answer stated in the chunks?
                       + the list of claims it couldn't find, so a rewrite knows what to drop
                       (reads question + chunks + answer)
  [IsUse]  useful      yes / no: could a nurse act on it, or is it vague? (reads question + answer only)

[IsUse] started inside the [IsSup] call and passed vague answers: seeing chunks that were just as vague, it called
m11's "give medicine through the drip" useful. On its own, without the chunks, it caught all 27 vague test answers.

Sprint 06's grader judged the chunks BEFORE the answer was written. This judges the answer AFTER, like an editor
reading the finished story next to the clippings. "I don't know" answers are not checked (nothing to support).

Run from rag_sprints/:  uv run week2/07_self_rag/checker.py
Checks the checker on the 26 answerable golden questions, with the golden answers as the key:
  1. true answer  + chunks holding the evidence     -> should be "fully" supported and useful
  2. wrong answer + chunks holding the evidence     -> should NOT be "fully" (one fact changed)
  3. true answer  + chunks WITHOUT the evidence     -> should NOT be "fully"
  4. vague answer + chunks holding the evidence     -> should be "not useful" (the m11 case)
Wrong and vague answers are written once by gpt-4o-mini and saved to test_answers.json. Cost: ~$0.04.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from eval_harness import settings, traceable
from eval_harness.golden import load_golden
from langsmith.utils import ContextThreadPoolExecutor

# Sprint 06's folder on the path: its grader is our [IsRel], and it has holds_evidence() and load_contextual()
sys.path.append(str(Path(__file__).resolve().parents[1] / "06_corrective_rag"))
import grader as crag  # noqa: E402

SUPPORT = ("fully", "partially", "no")
PRICE_IN, PRICE_OUT = crag.PRICE_IN, crag.PRICE_OUT
TEST_ANSWERS = Path(__file__).with_name("test_answers.json")

# The checker sees the chunks exactly as the answer step saw them ([i] (doc) text), with no notes: a claim that only
# a sprint-03 note backs up is not grounded in what the answer step read. Claims are listed BEFORE the verdict, so the
# model looks before it decides.
CHECK_PROMPT = """You check answers written by an assistant that answers from one hospital's internal documents.
Compare the answer with the chunks the assistant read, and reply as JSON:
{{"unsupported_claims": ["<claim>", ...], "supported": "<fully|partially|no>"}}
- unsupported_claims: every fact in the answer (number, dose, code, name, step, rule) that the chunks do not state. \
A fact is unsupported if the chunks say something different, or say it about a different drug, code or device. \
Empty list if all facts are in the chunks.
- supported: fully = every fact is stated in the chunks; partially = some are; no = the main facts are not.

Question: {question}

Chunks:
{chunks}

Answer: {answer}"""

# No chunks on purpose: usefulness is about the reader, not the documents (see the docstring).
USE_PROMPT = """Would this answer let a hospital nurse act on the question without asking a follow-up? \
Reply as JSON: {{"useful": "<yes|no>"}}
yes = it names the specific thing asked for (the drug and amount, the exact code, value, time, tube or step).
no = it is vague or generic ("give medicine", "adjust the dose", "follow the protocol"), or answers a different question.

Question: {question}
Answer: {answer}"""

MUTATE_PROMPT = """Here is a question and its correct answer, from a hospital's internal documents.
Write two altered answers for testing an answer checker, and reply as JSON: {{"wrong": "...", "vague": "..."}}
- wrong: the same answer with exactly ONE fact changed to a plausible but wrong value (a number, dose, code, \
colour, time or name). Keep everything else word for word.
- vague: an answer that stays on topic but gives none of the specific facts (no numbers, doses, codes or names \
from the correct answer), like a generic textbook reply.

Question: {question}
Correct answer: {answer}"""


def format_chunks(hits: list) -> str:
    """Same layout as naive's generate(), so the checker reads what the answer step read."""
    return "\n\n".join(f"[{i}] ({h.doc_id})\n{h.text}" for i, h in enumerate(hits, start=1))


def _json_call(prompt: str) -> tuple[dict, object]:
    resp = crag._openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content), resp.usage


@traceable(run_type="llm", name="is_sup")
def is_sup(question: str, hits: list, answer: str) -> tuple[dict, object]:
    return _json_call(CHECK_PROMPT.format(question=question, chunks=format_chunks(hits), answer=answer))


@traceable(run_type="llm", name="is_use")
def is_use(question: str, answer: str) -> tuple[dict, object]:
    return _json_call(USE_PROMPT.format(question=question, answer=answer))


@traceable(run_type="chain", name="check_answer")
def check(question: str, hits: list, answer: str) -> tuple[dict, object]:
    """[IsSup] and [IsUse] in parallel. Returns ({"supported", "useful", "unsupported_claims"}, summed token usage).

    A missing or garbled field becomes the cautious value ("partially", "no"): a broken reply triggers a rewrite or
    a wider search, never a silent pass.
    """
    with ContextThreadPoolExecutor(max_workers=2) as pool:  # keeps both calls under this trace in LangSmith
        sup = pool.submit(is_sup, question, hits, answer)
        use = pool.submit(is_use, question, answer)
        (raw, u1), (raw_use, u2) = sup.result(), use.result()
    claims = raw.get("unsupported_claims") or []
    result = {
        "supported": raw.get("supported") if raw.get("supported") in SUPPORT else "partially",
        "useful": raw_use.get("useful") if raw_use.get("useful") in ("yes", "no") else "no",
        "unsupported_claims": [str(c) for c in claims] if isinstance(claims, list) else [],
    }
    usage = SimpleNamespace(prompt_tokens=u1.prompt_tokens + u2.prompt_tokens,
                            completion_tokens=u1.completion_tokens + u2.completion_tokens)
    return result, usage


# ---------- The check: checker vs the golden answers and evidence quotes ----------

def test_answers(golden: list) -> dict:
    """{id: {"wrong", "vague"}}, written once and saved, so every run tests the checker on the same answers."""
    saved = json.loads(TEST_ANSWERS.read_text()) if TEST_ANSWERS.exists() else {}
    for g in golden:
        if g.id not in saved:
            resp = crag._openai().chat.completions.create(
                model=settings.llm_model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": MUTATE_PROMPT.format(question=g.question,
                                                                           answer=g.ground_truth)}])
            saved[g.id] = json.loads(resp.choices[0].message.content)
            TEST_ANSWERS.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def main():
    ctx = crag.load_contextual()
    golden = [g for g in load_golden("golden_medical_v1.jsonl") if g.question_type != "no_answer"]
    altered = test_answers(golden)

    rows, tokens_in, tokens_out, times = [], 0, 0, []
    for g in golden:
        top10 = ctx.hybrid_search(g.question, k=10)
        with_ev = [h for h in top10 if crag.holds_evidence(h.text, g.evidence)]
        without = [h for h in top10 if h not in with_ev][:5]
        if not with_ev:
            print(f"{g.id}: evidence not in the top 10, skipped")
            continue
        # Evidence chunks first, then other top hits up to 5: a realistic context, not just the answer chunk
        context = (with_ev + [h for h in top10 if h not in with_ev])[:max(5, len(with_ev))]
        cases = {"true": (g.ground_truth, context), "wrong": (altered[g.id]["wrong"], context),
                 "no_ev": (g.ground_truth, without), "vague": (altered[g.id]["vague"], context)}
        out = {}
        for name, (answer, hits) in cases.items():
            t0 = time.perf_counter()
            out[name], usage = check(g.question, hits, answer)
            times.append((time.perf_counter() - t0) * 1000)
            tokens_in += usage.prompt_tokens
            tokens_out += usage.completion_tokens
        rows.append((g, out))

    def cell(r: dict) -> str:
        return f"{r['supported'][:4]:4}/{'use' if r['useful'] == 'yes' else 'VAG'}"

    print(f"\n{'id':4} {'type':10}  {'1 true':9} {'2 wrong':9} {'3 no-ev':9} {'4 vague':9}   (supported/useful)")
    for g, out in rows:
        print(f"{g.id:4} {g.question_type:10}  " + " ".join(f"{cell(out[c]):9}" for c in
                                                            ("true", "wrong", "no_ev", "vague")))

    n = len(rows)
    print(f"\nOn {n} answerable questions with the evidence in the top 10:")
    print(f"  1. true answer, evidence present  -> 'fully':        {sum(o['true']['supported'] == 'fully' for _, o in rows):2} / {n}"
          f"   useful: {sum(o['true']['useful'] == 'yes' for _, o in rows)} / {n}")
    print(f"  2. one fact changed               -> caught:         {sum(o['wrong']['supported'] != 'fully' for _, o in rows):2} / {n}")
    print(f"  3. evidence removed               -> caught:         {sum(o['no_ev']['supported'] != 'fully' for _, o in rows):2} / {n}")
    print(f"  4. vague answer                   -> 'not useful':   {sum(o['vague']['useful'] == 'no' for _, o in rows):2} / {n}")
    print(f"\nVerdicts on true answers: {dict(Counter(o['true']['supported'] for _, o in rows))}")

    missed = [(g.id, altered[g.id]["wrong"]) for g, o in rows if o["wrong"]["supported"] == "fully"]
    false_alarm = [(g.id, o["true"]["unsupported_claims"]) for g, o in rows if o["true"]["supported"] != "fully"]
    if missed:
        print("\nWrong answers it passed as 'fully':")
        for qid, ans in missed:
            print(f"  {qid}: {ans}")
    if false_alarm:
        print("\nTrue answers it flagged (claims it said were unsupported):")
        for qid, claims in false_alarm:
            print(f"  {qid}: {claims}")

    times.sort()
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
    print(f"\nChecker latency p50 {times[len(times) // 2]:.0f} ms, max {times[-1]:.0f} ms | "
          f"{tokens_in // len(times)} tokens in per call | cost ${cost:.4f} (${cost / len(times):.6f} each)")


if __name__ == "__main__":
    main()
