"""RAPTOR lite, the pipeline: collapsed-tree search (leaves + summaries, top 5) -> naive's generate().

  question -> hybrid search over raptor_medical (retrieve.py) -> top 5 nodes, any level -> generate()

Same 5 places and the same answer step as Contextual, so any difference comes from the summaries.
The LLM sees a summary labelled "(summary of hvmc_formulary)" so it knows it isn't reading a source passage.
The harness gets context_ids "raptor:L1-05": a summary has no single document, and evidence recall only counts
verbatim quotes, so a summary is never an evidence hit (correct: it's a retelling, not the evidence).
"""

import json
from dataclasses import dataclass

from build_tree import TREE
from eval_harness import PipelineOutput, traceable
from retrieve import _ctx, collapsed_search, is_summary

generate = _ctx.hybrid.naive_pipeline.generate
_DOCS = {n["id"]: list(n["docs"]) for n in json.loads(TREE.read_text())["summaries"]}


@dataclass
class Shown:
    """What generate() reads: a source label and the text."""
    doc_id: str
    text: str


def _label(hit) -> str:
    return f"summary of {', '.join(_DOCS[hit.chunk_id])}" if is_summary(hit) else hit.doc_id


@traceable(run_type="chain", name="raptor_pipeline")
def raptor_pipeline(question: str) -> PipelineOutput:
    """The function the harness calls."""
    hits = collapsed_search(question)
    resp = generate(question, [Shown(_label(h), h.text) for h in hits])
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in hits],
        context_ids=[h.doc_id for h in hits],
        tokens_in=resp.usage.prompt_tokens,
        tokens_out=resp.usage.completion_tokens,
    )
