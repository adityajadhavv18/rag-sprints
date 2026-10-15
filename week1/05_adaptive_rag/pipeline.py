"""Adaptive RAG, the whole thing: the router picks a path, the path answers.

  question -> route() ─┬─ no_rag -> direct_answer()            the LLM alone, no search
                       ├─ single -> contextual_pipeline()      one hybrid search on the notes index (sprint 03)
                       └─ multi  -> multi_step_pipeline()      search -> read -> search again, max 3 rounds

Every path returns the harness's PipelineOutput. The router's tokens are added on top, so the cost of the decision
is counted too. ROUTE_LOG remembers which path each question took, for the summary in run_eval.py.
Requires Qdrant running (docker start qdrant). Medical corpus only.
"""

from eval_harness import PipelineOutput, settings, traceable
from multistep import _openai, ctx, multi_step_pipeline
from router import route

ROUTE_LOG: dict[str, str] = {}  # question -> route it took

DIRECT_PROMPT = "Answer briefly and plainly.\n\nQuestion: {question}"


@traceable(run_type="llm", name="direct_answer")
def direct_answer(question: str) -> PipelineOutput:
    """The no_rag path: no search, so no contexts. The LLM answers from what it already knows."""
    resp = _openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{"role": "user", "content": DIRECT_PROMPT.format(question=question)}],
    )
    return PipelineOutput(answer=resp.choices[0].message.content.strip(), contexts=[], context_ids=[],
                          tokens_in=resp.usage.prompt_tokens, tokens_out=resp.usage.completion_tokens)


PATHS = {"no_rag": direct_answer, "single": ctx.contextual_pipeline, "multi": multi_step_pipeline}


@traceable(run_type="chain", name="adaptive_pipeline")
def adaptive_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls: route first, then run that path, then add the router's tokens."""
    chosen, _reason, usage = route(question)
    ROUTE_LOG[question] = chosen
    out = PATHS[chosen](question)
    return out.model_copy(update={"tokens_in": out.tokens_in + usage.prompt_tokens,
                                  "tokens_out": out.tokens_out + usage.completion_tokens})
