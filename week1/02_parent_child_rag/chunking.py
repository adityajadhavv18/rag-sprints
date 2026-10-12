"""Parent-Child RAG, step 0: cut each document into parents (whole units) and children (single lines).

  parent = one blank-line block: a drug entry, a lab test, a fault code, a protocol...  -> what the LLM reads
  child  = one line of a parent (long lines split into sentences, max 64 tokens)       -> what the search matches

Children are embedded and searched. Parents are never embedded, so they may be any length.
Medical corpus only: the rules follow that corpus's layout.

Run from rag_sprints/:
    uv run week1/02_parent_child_rag/chunking.py   # prints examples, sizes, duplicates and the evidence check
"""

import importlib.util
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

os.environ.setdefault("RAG_CORPUS", "medical")
# Both sprints have a file called ingest.py, so load naive's by path under another name
_spec = importlib.util.spec_from_file_location(
    "naive_ingest", Path(__file__).resolve().parents[1] / "00_naive_baseline" / "ingest.py")
naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(naive)
if naive.CORPUS != "medical":
    raise SystemExit("Parent-Child is built for the medical corpus only: unset RAG_CORPUS or set it to medical")

DATA_DIR, GOLDEN_SET, EMBED_MODEL = naive.DATA_DIR, naive.GOLDEN_SET, naive.EMBED_MODEL
CHILD_MAX_TOKENS = 64  # bge tokens; only long protocol/leaflet lines ever reach this
DISCLAIMER = "SYNTHETIC DOCUMENT"  # identical line at the top of every doc: not a fact, never a child


@dataclass
class Parent:
    parent_id: str  # "hvmc_formulary::p012"
    doc_id: str  # becomes context_ids in PipelineOutput
    text: str
    n_tokens: int


@dataclass
class Child:
    child_id: str  # "hvmc_formulary::p012::c03"
    parent_id: str
    doc_id: str
    text: str
    n_tokens: int


def split_parents(text: str) -> list[str]:
    """One parent per blank-line block. A block that is only a heading is glued onto the block below it,
    so 'Section: Zephacillin Y-site compatibility' stays attached to its table, and the doc title to its first unit."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    parents, heading = [], ""
    for i, block in enumerate(blocks):
        is_heading = i == 0 or (block.startswith("Section:") and "\n" not in block)
        if is_heading:
            heading += block + "\n"
        else:
            parents.append(heading + block)
            heading = ""
    return parents


def split_children(parent_text: str, n_tokens) -> list[str]:
    """One child per line. A line over CHILD_MAX_TOKENS is cut into sentences, packed back up to the limit."""
    children = []
    for line in parent_text.split("\n"):
        line = line.strip()
        if not line or line.startswith(DISCLAIMER):
            continue
        if n_tokens(line) <= CHILD_MAX_TOKENS:
            children.append(line)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z])", line):
            if current and n_tokens(current + " " + sentence) > CHILD_MAX_TOKENS:
                children.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            children.append(current)
    return children


def load_parents_and_children() -> tuple[list[Parent], list[Child]]:
    tokenizer = Tokenizer.from_pretrained(EMBED_MODEL)
    tokenizer.no_truncation()  # parents can exceed 512 tokens; count them fully

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    parents, children = [], []
    for path in sorted(DATA_DIR.glob("*.txt")):
        for p, parent_text in enumerate(split_parents(path.read_text())):
            parent = Parent(f"{path.stem}::p{p:03d}", path.stem, parent_text, n_tokens(parent_text))
            parents.append(parent)
            for c, child_text in enumerate(split_children(parent_text, n_tokens)):
                children.append(Child(f"{parent.parent_id}::c{c:02d}", parent.parent_id, path.stem,
                                      child_text, n_tokens(child_text)))
    return parents, children


# ---------- Inspection (python chunking.py) ----------

def show_example(parents, children, first_line_start: str):
    parent = next(p for p in parents if any(line.startswith(first_line_start) for line in p.text.split("\n")))
    kids = [c for c in children if c.parent_id == parent.parent_id]
    print(f"\n  PARENT {parent.parent_id}  ({parent.n_tokens} tokens, {len(kids)} children)")
    for c in kids:
        print(f"    child c{c.child_id[-2:]} ({c.n_tokens:>2} tok)  {c.text[:95]}{'…' if len(c.text) > 95 else ''}")


def evidence_check(parents, children):
    """Does every golden evidence quote land inside ONE parent and inside ONE child? (no quote cut in half)"""
    from eval_harness.golden import load_golden
    from eval_harness.metrics import _evidence_fragments, _norm

    parent_texts = [_norm(p.text) for p in parents]
    child_texts = [_norm(c.text) for c in children]
    answerable = [g for g in load_golden(GOLDEN_SET) if g.question_type != "no_answer"]
    whole_in_parent = sum(any(all(f in p for f in _evidence_fragments(g)) for p in parent_texts) for g in answerable)
    each_in_child = sum(all(any(f in c for c in child_texts) for f in _evidence_fragments(g)) for g in answerable)
    print(f"4. evidence check ({len(answerable)} answerable questions)")
    print(f"     all evidence inside one parent:      {whole_in_parent}/{len(answerable)}"
          "   (a miss means the answer needs 2+ parents)")
    print(f"     each quote inside one child:         {each_in_child}/{len(answerable)}"
          "   (a miss means a quote was cut in half)")


def main():
    parents, children = load_parents_and_children()
    print(f"1. split: {len(parents)} parents, {len(children)} children from {len({p.doc_id for p in parents})} documents")
    for doc_id in sorted({p.doc_id for p in parents}):
        n_par = sum(p.doc_id == doc_id for p in parents)
        n_kid = sum(c.doc_id == doc_id for c in children)
        print(f"     {doc_id:26s} {n_par:>4} parents  {n_kid:>5} children")

    print("2. examples")
    for start in ("HV-RX-2074", "E-71", "CP-09"):
        show_example(parents, children, start)

    p_sizes, c_sizes = [p.n_tokens for p in parents], [c.n_tokens for c in children]
    print(f"\n3. sizes (bge tokens)   naive chunks: 450")
    print(f"     parents:  median {statistics.median(p_sizes):.0f}, max {max(p_sizes)}")
    print(f"     children: median {statistics.median(c_sizes):.0f}, max {max(c_sizes)}")
    counts = Counter(c.text for c in children)
    dup_children = sum(n for n in counts.values() if n > 1)
    print(f"     children whose exact text also appears in another parent: {dup_children}/{len(children)} "
          f"({dup_children / len(children):.0%}). Most common:")
    for text, n in counts.most_common(4):
        print(f"       x{n:<4} {text}")

    evidence_check(parents, children)


if __name__ == "__main__":
    main()
