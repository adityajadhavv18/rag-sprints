"""Phase 3 check: golden set loads, is balanced, and every evidence quote exists in its source doc.

Run from rag_sprints/:  uv run check_golden.py
"""

import re
from collections import Counter
from pathlib import Path

from eval_harness import load_golden

DATA_DIR = Path(__file__).resolve().parent / "data"
corpus = {p.stem: p.read_text() for p in DATA_DIR.glob("*.txt")}

golden = load_golden(known_docs=set(corpus))
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
print("✅ all evidence quotes found verbatim in source docs" if bad == 0 else f"❌ {bad} evidence fragment(s) missing")
