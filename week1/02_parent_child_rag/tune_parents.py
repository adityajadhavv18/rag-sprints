"""Phase 3: how many parents should the LLM get? Free: retrieval metrics only, nothing saved.

Search 30 children -> swap for parents -> drop repeats -> keep N. Tries N = 3, 5, 8, 10 for dense and hybrid
child search, next to naive (5 chunks) and hybrid (5 chunks). "Context tokens" = size of what the LLM would read
(gpt-4o-mini tokens), because more text costs more and can bury the answer.

Run from rag_sprints/:  uv run week1/02_parent_child_rag/tune_parents.py
"""

import contextlib
import io
import statistics

import tiktoken

from eval_harness import PipelineOutput, evaluate
from ingest import chunking
from pipeline import CHILD_DEPTH, dense_child_search, hybrid_child_search, hybrid_pipeline, parent_search

N_OPTIONS = (3, 5, 8, 10)
_enc = tiktoken.encoding_for_model("gpt-4o-mini")


def retrieval_only(search):
    """Wrap a retriever as a harness pipeline with no LLM call: only retrieval gets scored."""
    def pipeline(question: str) -> PipelineOutput:
        hits = search(question)
        return PipelineOutput(answer="", contexts=[h.text for h in hits], context_ids=[h.doc_id for h in hits])
    return pipeline


def score(name: str, search) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):  # evaluate() prints a full report; we only want the numbers
        run = evaluate(retrieval_only(search), arch=name, golden_set=chunking.GOLDEN_SET,
                       use_llm_judge=False, save=False, compare_to=None)
    outputs = [q.output for q in run.questions if q.output]
    return {
        "recall": run.summary["evidence_recall"],
        "mrr": run.summary["evidence_mrr"],
        "items": statistics.mean(len(o.contexts) for o in outputs),
        "tokens": statistics.mean(len(_enc.encode("\n\n".join(o.contexts))) for o in outputs),
    }


def main():
    configs = {
        "naive      5 chunks": lambda q: hybrid_pipeline.dense_search(q, k=5),  # same index + model as naive
        "hybrid     5 chunks": lambda q: hybrid_pipeline.hybrid_search(q),
    }
    for n in N_OPTIONS:
        configs[f"PC dense  {n:>2} parents"] = lambda q, n=n: parent_search(q, n, search=dense_child_search)
    for n in N_OPTIONS:
        configs[f"PC hybrid {n:>2} parents"] = lambda q, n=n: parent_search(q, n, search=hybrid_child_search)

    print(f"Retrieval only, {CHILD_DEPTH} children searched before swapping to parents\n")
    print(f"{'':22} {'evidence recall':>15} {'MRR':>6} {'items':>6} {'context tokens':>15}")
    for name, search in configs.items():
        s = score(name, search)
        print(f"{name:22} {s['recall']:>15.2f} {s['mrr']:>6.2f} {s['items']:>6.1f} {s['tokens']:>15.0f}")
        if name.startswith("hybrid"):
            print()


if __name__ == "__main__":
    main()
