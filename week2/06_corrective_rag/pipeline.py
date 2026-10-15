"""Corrective RAG, the harness entry point: grade, retry if needed, then answer from the chunks that passed.

  corrective_search   search -> grade -> refine and search again (max 3 rounds) -> kept chunks (retry.py)
  answer              naive's generate() on the kept chunks, relevant first, at most 5
  nothing kept        "I don't know" without an answer call; the web-search event is already recorded

The answer step also runs when the web-search event fired but some "partial" chunks were kept: the grader
sometimes calls the evidence chunk partial (m04, m09), and generate() already says "I don't know" when the
fact isn't there.

Requires Qdrant running (docker start qdrant) and Contextual's collection from sprint 03. Medical corpus only.
"""

from eval_harness import PipelineOutput, traceable
from retry import corrective_search, ctx

generate = ctx.hybrid.naive_pipeline.generate  # naive's answer step, unchanged since sprint 00
NO_ANSWER = ctx.hybrid.naive_pipeline.NO_ANSWER
TOP_K = ctx.TOP_K  # 5, same as every sprint

# question -> (rounds, web_search_needed), filled as the eval runs, so run_eval.py can count paths
LOOP_LOG: dict[str, tuple[int, bool]] = {}


@traceable(run_type="chain", name="corrective_pipeline")
def corrective_pipeline(question: str) -> PipelineOutput:
    """contexts = exactly the chunks the LLM read. Tokens include every grader and refine call."""
    found = corrective_search(question)
    LOOP_LOG[question] = (len(found.rounds), found.web_search_needed)
    hits = found.kept[:TOP_K]
    if not hits:
        return PipelineOutput(answer=NO_ANSWER, contexts=[], context_ids=[],
                              tokens_in=found.tokens_in, tokens_out=found.tokens_out)
    resp = generate(question, hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens + found.tokens_in,
        tokens_out=resp.usage.completion_tokens + found.tokens_out,
    )
