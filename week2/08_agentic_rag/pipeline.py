"""Agentic RAG, the harness entry point: the agent loop, then its own final message as the answer.

  run_agent   the LLM picks search / lookup_entry and their inputs, round by round (max 5), then answers (agent.py)
  contexts    every item any tool handed back, in order, no repeats: that is exactly what the LLM read
              (search items carry their sprint 03 note on top, because the agent read the note too)
  tokens      every agent call, including the forced answer when the cap is hit

Requires Qdrant running (docker start qdrant) and Contextual's collection from sprint 03. Medical corpus only.
"""

from collections import Counter

from agent import run_agent
from eval_harness import PipelineOutput, traceable

# question -> (tool rounds, tools called, hit the cap), filled as the eval runs, so run_eval.py can count them
LOOP_LOG: dict[str, tuple[int, Counter, bool]] = {}


@traceable(run_type="chain", name="agentic_pipeline")
def agentic_pipeline(question: str) -> PipelineOutput:
    r = run_agent(question)
    LOOP_LOG[question] = (r.rounds, Counter(s.tool for s in r.steps), r.hit_cap)
    return PipelineOutput(
        answer=r.answer,
        contexts=[it.text for it in r.items],
        context_ids=[it.doc_id for it in r.items],
        tokens_in=r.tokens_in,
        tokens_out=r.tokens_out,
    )
