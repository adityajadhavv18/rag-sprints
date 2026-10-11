"""Phase 3 check: golden set loads, is balanced, and every evidence quote exists in its source doc.

Run from rag_sprints/:
    uv run check_golden.py                                       # 10-K set: golden_v1.jsonl vs data/
    uv run check_golden.py golden_medical_v1.jsonl data/medical  # any golden set vs its corpus folder
"""

import re
import sys
from collections import Counter
from pathlib import Path

from eval_harness import load_golden

GOLDEN_SET = sys.argv[1] if len(sys.argv) > 1 else "golden_v1.jsonl"
DATA_DIR = Path(__file__).resolve().parent / (sys.argv[2] if len(sys.argv) > 2 else "data")
corpus = {p.stem: p.read_text() for p in DATA_DIR.glob("*.txt")}

golden = load_golden(GOLDEN_SET, known_docs=set(corpus))
print(f"✅ loaded {len(golden)} items")
print("   by type:", dict(Counter(g.question_type for g in golden)))
print("   by doc: ", dict(Counter(g.source_doc or "(none)" for g in golden)))

# Evidence may join several quotes with " ... " or " / "; each fragment must appear verbatim
bad = 0
for g in golden:
    if not g.evidence:
        continue
    for fragment in re.split(r" \.\.\. | / ", g.evidence):
        if fragment.strip() not in corpus[g.source_doc]:
            print(f"❌ {g.id}: evidence not found in {g.source_doc}: {fragment[:80]!r}")
            bad += 1
        # evidence_recall accepts the fragment in ANY retrieved chunk, so a quote that also appears elsewhere
        # lets a wrong chunk count as a hit
        elif (n := sum(text.count(fragment.strip()) for text in corpus.values())) > 1:
            print(f"⚠️  {g.id}: evidence fragment appears {n} times in the corpus: {fragment[:60]!r}")
print("✅ all evidence quotes found verbatim in source docs" if bad == 0 else f"❌ {bad} evidence fragment(s) missing")
