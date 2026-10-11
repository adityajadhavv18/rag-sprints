"""Stretch stage: how do RRF's three knobs change retrieval? Free: no LLM, nothing saved.

  RRF_K       damping in weight / (k + rank). Smaller k = rank #1 counts for much more than rank #20
  CANDIDATES  how deep each path's list goes into fusion. Fewer = fewer lukewarm chunks found by both
  weights     how much each path's points count

Each question is searched ONCE per path at full depth; every config then just re-fuses those lists in memory.
Scoring uses the harness's rules: evidence recall@TOP_K (fraction of evidence fragments in the top 5) and
evidence MRR (1 / rank of the first chunk holding any fragment).

Tune on one corpus, then re-run on the other as a held-out check:
    RAG_CORPUS=medical uv run week1/01_hybrid_rag/tune_fusion.py
    uv run week1/01_hybrid_rag/tune_fusion.py                      # 10-K corpus (needs hybrid_10k ingested)
"""

import re
import statistics

from eval_harness import load_golden
from ingest import CORPUS, GOLDEN_SET
from pipeline import CANDIDATES, RRF_K, TOP_K, bm25_search, dense_search, rrf_fuse

KS = [60, 30, 10, 5, 2]
DEPTHS = [20, 10, 5]
WEIGHTS = {"equal": {"dense": 1, "bm25": 1}, "bm25 x2": {"dense": 1, "bm25": 2}, "dense x2": {"dense": 2, "bm25": 1}}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def score(hits, fragments) -> tuple[float, float]:
    texts = [_norm(h.text) for h in hits[:TOP_K]]
    recall = sum(any(f in t for t in texts) for f in fragments) / len(fragments)
    rank = next((i for i, t in enumerate(texts, start=1) if any(f in t for f in fragments)), None)
    return recall, (1 / rank if rank else 0.0)


def summarize(rows) -> dict[str, float]:
    """rows: (question_type, recall, mrr) per question -> mean recall, MRR, and recall per type."""
    out = {"recall": statistics.mean(r for _, r, _ in rows), "mrr": statistics.mean(m for _, _, m in rows)}
    for qtype in ("exact_term", "simple", "multi_part"):
        typed = [r for t, r, _ in rows if t == qtype]
        out[qtype] = statistics.mean(typed) if typed else float("nan")
    return out


def main():
    items = [g for g in load_golden(GOLDEN_SET) if g.question_type != "no_answer"]
    cache = []  # (item, fragments, dense list, bm25 list) at full depth: one search per path per question
    for g in items:
        fragments = [_norm(f) for f in re.split(r" \.\.\. | / ", g.evidence) if f.strip()]
        cache.append((g, fragments, dense_search(g.question, k=max(DEPTHS)), bm25_search(g.question, k=max(DEPTHS))))

    def run(k, depth, weights):
        rows = []
        for g, fragments, dense, bm25 in cache:
            fused = rrf_fuse({"dense": dense[:depth], "bm25": bm25[:depth]}, k=k, weights=weights)
            rows.append((g.question_type, *score(fused, fragments)))
        return summarize(rows)

    print(f"Corpus '{CORPUS}' · {GOLDEN_SET} · {len(items)} answerable questions · top {TOP_K} to the LLM")
    print("Cells: evidence recall / MRR\n")
    singles = {name: summarize([(g.question_type, *score(lst, fr)) for g, fr, *lists in cache
                                for lst in [lists[i]]])
               for i, name in enumerate(["dense", "bm25"])}
    for name, s in singles.items():
        print(f"   {name} alone        {s['recall']:.2f} / {s['mrr']:.2f}   "
              f"(exact {s['exact_term']:.2f}, simple {s['simple']:.2f})")

    results = {}
    for wname, weights in WEIGHTS.items():
        print(f"\n weights: {wname}")
        print("   " + "RRF_K".ljust(7) + "".join(f"{f'depth {d}':>14}" for d in DEPTHS))
        for k in KS:
            cells = []
            for d in DEPTHS:
                results[(k, d, wname)] = s = run(k, d, weights)
                mark = "*" if (k, d, wname) == (RRF_K, CANDIDATES, "equal") else " "
                cells.append(f"{s['recall']:.2f} / {s['mrr']:.2f}{mark}")
            print(f"   {k:<7}" + "".join(f"{c:>14}" for c in cells))
    print("\n   * = current default")

    print("\n Best 8 configs by recall, then MRR (exact_term and simple recall shown, they pull in opposite ways)")
    best = sorted(results.items(), key=lambda kv: (-kv[1]["recall"], -kv[1]["mrr"]))[:8]
    for (k, d, w), s in best:
        print(f"   k={k:<3} depth={d:<3} {w:<9} recall {s['recall']:.2f}  MRR {s['mrr']:.2f}   "
              f"exact {s['exact_term']:.2f}  simple {s['simple']:.2f}  multi {s['multi_part']:.2f}")


if __name__ == "__main__":
    main()
