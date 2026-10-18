"""RAPTOR lite, collapsed-tree search: sprint 03's hybrid search, pointed at raptor_medical (leaves + summaries).

Demo (free, no LLM): what the top 5 holds, Contextual vs collapsed tree.
  thematic   per question: the 5 hits (summaries marked), how many of the 5 points' quotes are in a retrieved leaf,
             and how many more sit UNDER a retrieved summary (the summary covers that chunk, but not word for word)
  golden     evidence in the top 5 on all 30 questions, and how many top-5 places summaries took

Run from rag_sprints/:  uv run week2/10_raptor/retrieve.py
"""

import json
import re

from build_tree import TREE
from eval_harness.golden import load_golden
from eval_harness.metrics import _norm
from index_tree import COLLECTION, CTX_DIR
from thematic import _load, load_questions

_ctx = _load("contextual_pipeline", CTX_DIR / "pipeline.py")
TOP_K = _ctx.TOP_K  # 5, as every earlier sprint


def collapsed_search(question: str, k: int = TOP_K) -> list:
    """Hybrid search over every level of the tree at once. A hit's doc_id is "raptor:L1-05" for a summary."""
    return _ctx.hybrid_search(question, k=k, collection=COLLECTION)


def is_summary(hit) -> bool:
    return hit.doc_id.startswith("raptor:")


def _tree():
    tree = json.loads(TREE.read_text())
    return {n["id"]: n["text"] for n in tree["leaves"]}, {n["id"]: n for n in tree["summaries"]}


def holds(evidence: str, text: str) -> bool:
    """The harness's rule: any verbatim fragment of the evidence (split on ' ... ' or ' / ') inside the text."""
    return any(_norm(f) in _norm(text) for f in re.split(r" \.\.\. | / ", evidence) if f.strip())


def coverage(points: list[dict], hits: list, leaves: dict, summaries: dict) -> tuple[int, int]:
    """(points whose quote is in a retrieved leaf, points whose quote is only under a retrieved summary)."""
    under = {leaf for h in hits if is_summary(h) for leaf in summaries[h.chunk_id]["leaves"]}
    in_leaf = in_summary = 0
    for p in points:
        if any(holds(p["evidence"], h.text) for h in hits if not is_summary(h)):
            in_leaf += 1
        elif any(holds(p["evidence"], leaves[leaf]) for leaf in under):
            in_summary += 1
    return in_leaf, in_summary


def short(hit) -> str:
    return f"[{hit.chunk_id}]" if is_summary(hit) else hit.chunk_id.replace("hvmc_", "").replace("_manual", "")


def thematic_demo(leaves, summaries):
    totals = {"contextual": [0, 0], "raptor": [0, 0]}
    for q in load_questions():
        print(f"\n{q['id']}: {q['question']}")
        for name, hits in (("contextual", _ctx.hybrid_search(q["question"])), ("raptor", collapsed_search(q["question"]))):
            in_leaf, in_summary = coverage(q["points"], hits, leaves, summaries)
            totals[name][0] += in_leaf
            totals[name][1] += in_summary
            print(f"  {name:<11} leaf quotes {in_leaf}/5, +{in_summary} under a summary | "
                  + "  ".join(short(h) for h in hits))
    print("\nthematic, 25 points:  " + "   ".join(f"{n}: {a} in a leaf + {b} under a summary"
                                                for n, (a, b) in totals.items()))


def golden_demo():
    golden = [g for g in load_golden("golden_medical_v1.jsonl") if g.evidence]
    found = {"contextual": 0, "raptor": 0}
    summary_slots, summary_first, lost = 0, [], []
    for g in golden:
        ctx_hits, rap_hits = _ctx.hybrid_search(g.question), collapsed_search(g.question)
        ctx_ok = any(holds(g.evidence, h.text) for h in ctx_hits)
        rap_ok = any(holds(g.evidence, h.text) for h in rap_hits)
        found["contextual"] += ctx_ok
        found["raptor"] += rap_ok
        summary_slots += sum(is_summary(h) for h in rap_hits)
        if is_summary(rap_hits[0]):
            summary_first.append(f"{g.id}->{rap_hits[0].chunk_id}")
        if ctx_ok and not rap_ok:
            lost.append(f"{g.id} ({', '.join(h.chunk_id for h in rap_hits if is_summary(h))} took its place)")
    print(f"\ngolden, {len(golden)} answerable: evidence in top 5  contextual {found['contextual']}  "
          f"raptor {found['raptor']}")
    print(f"  summaries took {summary_slots} of {len(golden) * TOP_K} top-5 places; a summary ranked #1 on "
          f"{len(summary_first)}: {', '.join(summary_first) or '-'}")
    print(f"  evidence lost to a summary: {', '.join(lost) or 'none'}")


def main():
    leaves, summaries = _tree()
    thematic_demo(leaves, summaries)
    golden_demo()


if __name__ == "__main__":
    main()
