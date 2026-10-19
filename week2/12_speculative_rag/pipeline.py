"""Speculative RAG, the pipelines the harness calls.

  speculative_pipeline   search top 9 -> 3 gpt-4o-mini children draft in parallel -> gpt-4o parent picks one
                         contexts = the winner's 3 chunks (what the final answer was written from).
                         All children refused -> refuse in code; contexts = all 9 (what was read).
  contextual_4o_pipeline the baseline to race: the same 9 chunks, gpt-4o reads them all and writes the answer
                         with naive's prompt. No children, no parent.

Cost: the harness prices every token at settings.llm_model (gpt-4o-mini), which is wrong here. So every LLM call
is also logged in USAGE as (question, model, tokens_in, tokens_out), and run_eval.py prices each model itself.
ALL_HITS keeps the 9 search hits per question, for "was the evidence anywhere in the 9?".

Quick check (<$0.02): uv run week2/12_speculative_rag/pipeline.py   (from rag_sprints/)
"""

import time

from drafter import DRAFTER_MODEL, NO_ANSWER, _llm, ctx, draft_all, search
from eval_harness import PipelineOutput, traceable
from verifier import VERIFIER_MODEL, pick

naive = ctx.hybrid.naive_pipeline
BASELINE_MODEL = VERIFIER_MODEL  # gpt-4o: the big model doing everything

USAGE: list[tuple[str, str, int, int]] = []  # (question, model, tokens_in, tokens_out), one row per LLM call
ALL_HITS: dict[str, list] = {}


@traceable(run_type="chain", name="speculative_pipeline")
def speculative_pipeline(question: str) -> PipelineOutput:
    hits = search(question)
    ALL_HITS[question] = hits
    drafts = draft_all(question, hits)
    verdicts = pick(question, drafts)
    USAGE.extend((question, DRAFTER_MODEL, d.tokens_in, d.tokens_out) for d in drafts)
    USAGE.extend((question, VERIFIER_MODEL, v.tokens_in, v.tokens_out) for v in verdicts)

    shown = verdicts[0].draft.hits if verdicts else hits
    return PipelineOutput(
        answer=verdicts[0].draft.answer if verdicts else NO_ANSWER,
        contexts=[h.text for h in shown],
        context_ids=[h.doc_id for h in shown],
        tokens_in=sum(d.tokens_in for d in drafts) + sum(v.tokens_in for v in verdicts),
        tokens_out=sum(d.tokens_out for d in drafts) + sum(v.tokens_out for v in verdicts),
    )


@traceable(run_type="chain", name="contextual_4o_pipeline")
def contextual_4o_pipeline(question: str) -> PipelineOutput:
    hits = search(question)
    ALL_HITS[question] = hits
    context = "\n\n".join(f"[{i}] ({h.doc_id})\n{h.text}" for i, h in enumerate(hits, start=1))  # as naive's
    resp = _llm().chat.completions.create(
        model=BASELINE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": naive.PROMPT.format(no_answer=NO_ANSWER, context=context,
                                                                  question=question)}],
    )
    USAGE.append((question, BASELINE_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens))
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )


def real_cost(usage=None) -> float:
    """USD for the logged calls, each priced at its own model's rate."""
    from eval_harness.metrics import cost_usd
    return sum(cost_usd(i, o, m) for _, m, i, o in (USAGE if usage is None else usage))


# ---------- Quick check: both pipelines on 3 questions ----------

def main():
    from eval_harness.golden import load_golden

    golden = {g.id: g for g in load_golden("golden_medical_v1.jsonl")}
    search("warm up the embedder")  # so the first question's time isn't model loading
    for qid in ["m01", "m21", "m28"]:
        g = golden[qid]
        print(f"\n{'=' * 100}\n{qid}: {g.question}\n  truth: {g.ground_truth}")
        for name, fn in [("gpt-4o alone", contextual_4o_pipeline), ("speculative ", speculative_pipeline)]:
            n = len(USAGE)
            t0 = time.perf_counter()
            out = fn(g.question)
            ms = (time.perf_counter() - t0) * 1000
            print(f"\n  {name}  {ms:5.0f} ms  ${real_cost(USAGE[n:]):.5f}  ({len(out.contexts)} chunks in contexts)\n"
                  f"    {out.answer}")


if __name__ == "__main__":
    main()
