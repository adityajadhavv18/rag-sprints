"""Single-step vs multi-step on the 5 chain questions, plus 2 golden questions as a control.

For each question it prints every round of the loop (what was searched, which documents came back, what the reader
decided), then checks each hop of the chain: did a chunk holding that hop's evidence quote reach the LLM?
The 2 golden controls (m01, m22) need one search, so a good reader should stop after round 1.

Cost: ~2 to 4 small gpt-4o-mini calls per question, well under $0.01 in total.
Run from rag_sprints/:  uv run week1/05_adaptive_rag/compare_chain.py
"""

import json
import re
import time
from pathlib import Path

from eval_harness.golden import load_golden
from multistep import ctx, generate, is_done, multi_step_search

EXTRA = Path(__file__).with_name("questions_extra.jsonl")
CONTROLS = ("m01", "m22")


def norm(text: str) -> str:  # same normaliser as the harness: collapse whitespace, lower case
    return re.sub(r"\s+", " ", text).strip().lower()


def hop_found(evidence: str, hits: list) -> int | None:
    """Position (1-based) of the first chunk holding this evidence quote, or None."""
    return next((i for i, h in enumerate(hits, start=1) if norm(evidence) in norm(h.text)), None)


def main():
    chains = [json.loads(line) for line in EXTRA.read_text().splitlines() if '"route": "multi"' in line]
    controls = [{"id": g.id, "question": g.question, "ground_truth": g.ground_truth, "hops": []}
                for g in load_golden("golden_medical_v1.jsonl") if g.id in CONTROLS]

    summary = []
    for q in chains + controls:
        print(f"\n{'=' * 100}\n{q['id']}  {q['question']}\n{'=' * 100}")

        t0 = time.perf_counter()
        single_hits = ctx.hybrid_search(q["question"])
        single_answer = generate(q["question"], single_hits).choices[0].message.content.strip()
        single_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        found = multi_step_search(q["question"])
        multi_answer = generate(q["question"], found.hits).choices[0].message.content.strip()
        multi_s = time.perf_counter() - t0

        for n, r in enumerate(found.rounds, start=1):
            docs = ", ".join(h.doc_id.removeprefix("hvmc_") for h in r.new_hits) or "(nothing new)"
            print(f"  round {n}  search: {r.query!r}\n           new chunks: {docs}")
            print(f"           found:   {r.decision.get('found', '')!r}")
            print(f"           missing: {r.decision.get('missing', '')!r}" + ("  -> DONE" if is_done(r.decision) else ""))

        hops_single, hops_multi = [], []
        for i, hop in enumerate(q["hops"], start=1):
            s, m = hop_found(hop["evidence"], single_hits), hop_found(hop["evidence"], found.hits)
            hops_single.append(s is not None)
            hops_multi.append(m is not None)
            print(f"  hop {i} ({hop['doc'].removeprefix('hvmc_')}): single {'#' + str(s) if s else 'missed'}"
                  f" | multi {'#' + str(m) if m else 'missed'}")

        print(f"\n  TRUTH  {q['ground_truth']}")
        print(f"  SINGLE ({single_s:.1f}s, {len(single_hits)} chunks)  {single_answer}")
        print(f"  MULTI  ({multi_s:.1f}s, {len(found.hits)} chunks, {len(found.rounds)} rounds)  {multi_answer}")
        summary.append((q["id"], hops_single, hops_multi, len(found.rounds), single_s, multi_s))

    print(f"\n{'=' * 100}\nSUMMARY  (hops found = every hop's evidence reached the LLM)\n{'=' * 100}")
    print(f"{'id':5} {'single: hops found':>20} {'multi: hops found':>20} {'rounds':>7} {'single s':>9} {'multi s':>8}")
    for qid, hs, hm, rounds, ss, ms in summary:
        fmt = lambda h: f"{sum(h)}/{len(h)}" if h else "-"
        print(f"{qid:5} {fmt(hs):>20} {fmt(hm):>20} {rounds:>7} {ss:>9.1f} {ms:>8.1f}")


if __name__ == "__main__":
    main()
