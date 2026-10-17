"""Self-RAG, the harness entry point: the loop from loop.py, wrapped as a PipelineOutput.

  self_rag   search -> [IsRel] -> answer -> [IsSup + IsUse] -> wider search (once) / rewrite (max 3) -> answer
  contexts   the chunks the LAST answer was written from, so the judge sees what that answer saw
  tokens     every grader, answer, checker and rewrite call

Requires Qdrant running (docker start qdrant) and Contextual's collection from sprint 03. Medical corpus only.
"""

from eval_harness import PipelineOutput, traceable
from loop import self_rag

# question -> what the loop did (rewrites, widened, flag, answers written), filled as the eval runs, for run_eval.py
PATH_LOG: dict[str, dict] = {}


@traceable(run_type="chain", name="self_rag_pipeline")
def self_rag_pipeline(question: str) -> PipelineOutput:
    res = self_rag(question)
    PATH_LOG[question] = {"rewrites": res.rewrites, "widened": res.widened, "flag": res.flag,
                          "answers": [s.detail["text"] for s in res.steps if s.kind in ("answer", "rewrite")],
                          "checks": [s.detail for s in res.steps if s.kind == "check"]}
    return PipelineOutput(
        answer=res.answer,
        contexts=[h.text for h in res.hits],
        context_ids=[h.doc_id for h in res.hits],
        tokens_in=res.tokens_in,
        tokens_out=res.tokens_out,
    )
