"""Agentic RAG, the two tools. Each one is a plain Python function plus a short description the LLM reads.

  search(query)               sprint 03's Contextual hybrid search: the 5 chunks that best match a few words,
                              each with its sprint 03 note on top (the shelf browser)
  lookup_entry(code_or_name)  one whole entry from the formulary, lab catalog or protocols, by its exact code
                              (HV-RX-2074, LC-0625, CP-07) or exact name (Corvetrol, aPTT) (the index card drawer)

The LLM never runs these. It asks for one by name with an input; the loop in agent.py runs it and hands back the text.

Run from rag_sprints/:  uv run week2/08_agentic_rag/tools.py
Runs both tools by hand on c01's two hops and on the look-alike traps, so you see exactly what the agent will see.
Cost: free (no LLM calls; search embeds the query locally). Needs Qdrant running and sprint 03's collection.
"""

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from eval_harness import traceable

ROOT = Path(__file__).resolve().parents[2]  # rag_sprints/
DATA = ROOT / "data" / "medical"
NOTES_FILE = ROOT / "week1" / "03_contextual_rag" / "notes_medical.json"
ENTRY_FILES = ("hvmc_formulary", "hvmc_lab_catalog", "hvmc_clinical_protocols")
CODE = re.compile(r"\b(HV-RX-\d{4}|LC-\d{4}|CP-\d{2})\b", re.IGNORECASE)
MAX_MATCHES = 3  # a name like "Potassium" can match several entries; never flood the agent


@dataclass
class Item:
    """One thing a tool handed back. `text` is exactly what the LLM reads."""
    id: str      # chunk id (hvmc_formulary::0002) or entry code (LC-0625); the loop uses it to skip repeats
    doc_id: str  # which document, for the harness's context_ids
    text: str


# ---------- Tool 1: search (sprint 03's Contextual hybrid search) ----------

@cache
def load_contextual():
    """Sprint 03's search, loaded the same way sprints 04 to 06 do (its folder on the path, its file by path)."""
    os.environ.setdefault("RAG_CORPUS", "medical")
    if os.environ["RAG_CORPUS"] != "medical":
        raise SystemExit("Agentic RAG runs on the medical corpus only: unset RAG_CORPUS or set it to medical")
    ctx_dir = ROOT / "week1" / "03_contextual_rag"
    sys.path.append(str(ctx_dir))
    spec = importlib.util.spec_from_file_location("contextual_pipeline", ctx_dir / "pipeline.py")
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    return ctx


@cache
def _notes() -> dict:
    return {chunk_id: entry["note"] for chunk_id, entry in json.loads(NOTES_FILE.read_text()).items()}


@traceable(run_type="tool", name="search")
def search(query: str) -> list[Item]:
    """Top 5 chunks for the query. The note goes on top: chunks often start mid-entry, and without it the LLM
    can't tell which drug a chunk is about (sprint 06's lesson)."""
    hits = load_contextual().hybrid_search(query)
    return [Item(h.chunk_id, h.doc_id, f"Note: {_notes().get(h.chunk_id, h.doc_id)}\n{h.text}") for h in hits]


# ---------- Tool 2: lookup_entry (exact code or name -> the whole entry) ----------

def _names(title: str) -> set[str]:
    """The names an entry answers to, lower case.

    "Corvetrol (oral direct thrombin inhibitor)" -> {corvetrol (oral ...), corvetrol}
    "Activated partial thromboplastin time (aPTT)" -> {..., activated partial thromboplastin time, aptt}
    "Sodium, serum" -> {sodium, serum, sodium}
    A bracket counts as a name only when it is one word: in the formulary the bracket is the drug class,
    and "low molecular weight heparin" must not pull up every heparin.
    """
    names = {title.lower()}
    main = title.split(" (")[0]
    names |= {main.lower(), main.split(",")[0].lower()}
    alias = re.search(r"\(([^()\s]+)\)", title)
    if alias:
        names.add(alias.group(1).lower())
    return names


@cache
def _entries() -> dict[str, tuple[str, str, set[str]]]:
    """code -> (doc_id, whole entry text, names). An entry is a blank-line block that starts with its code."""
    entries = {}
    for doc_id in ENTRY_FILES:
        for block in (DATA / f"{doc_id}.txt").read_text().split("\n\n"):
            block = block.strip()
            head = re.match(r"(HV-RX-\d{4}|LC-\d{4}|CP-\d{2}) (.+)", block)
            if head:
                entries[head.group(1)] = (doc_id, block, _names(head.group(2)))
    return entries


@traceable(run_type="tool", name="lookup_entry")
def lookup_entry(code_or_name: str) -> list[Item]:
    """Exact match only, never "closest": a code anywhere in the input wins; otherwise a whole-name match
    (case ignored). No match returns an empty list, which the loop turns into "no entry found"."""
    entries = _entries()
    code = CODE.search(code_or_name)
    if code:
        key = code.group(1).upper()
        matches = [key] if key in entries else []
    else:
        name = code_or_name.strip().lower()
        matches = [c for c, (_, _, names) in entries.items() if name in names][:MAX_MATCHES]
    return [Item(c, entries[c][0], entries[c][1]) for c in matches]


# ---------- What the LLM reads about each tool (OpenAI function-calling format) ----------

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search all of the hospital's documents (formulary, lab catalog, clinical protocols, "
                       "antibiogram, IV compatibility, policies, directory, patient leaflets, device manuals). "
                       "Returns the 5 best-matching passages, each with a note saying what it covers. Use it when "
                       "you don't yet know the exact code or name, or for anything outside the formulary, lab "
                       "catalog and protocols. Use short keywords, not sentences.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A few keywords, e.g. 'Norvessa extravasation antidote'"}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "lookup_entry",
        "description": "Fetch one whole entry by its exact code or exact name. Covers the formulary (HV-RX-####, "
                       "drug names like Corvetrol), the lab catalog (LC-####, test names like aPTT or TSH) and the "
                       "clinical protocols (CP-##). Exact match only: similar codes are different entries. Returns "
                       "'no entry found' when nothing matches exactly.",
        "parameters": {"type": "object", "properties": {
            "code_or_name": {"type": "string", "description": "An exact code (LC-0625) or name (Corvetrol)"}},
            "required": ["code_or_name"], "additionalProperties": False}}},
]

FUNCTIONS = {"search": search, "lookup_entry": lookup_entry}  # tool name -> the Python function the loop runs


# ---------- Demo: run each tool by hand ----------

def show(call: str, items: list[Item], chars: int = 300):
    print(f"\n>>> {call}   ->  {len(items)} item(s)")
    for it in items or []:
        print(f"  [{it.id}]  ({it.doc_id})\n    " + it.text[:chars].replace("\n", "\n    ") + (" ..." if len(it.text) > chars else ""))
    if not items:
        print("  no entry found")


def main():
    e = _entries()
    print(f"lookup_entry index: {len(e)} entries "
          + ", ".join(f"{sum(d == doc for d, _, _ in e.values())} {doc.removeprefix('hvmc_')}" for doc in ENTRY_FILES))

    print("\n" + "=" * 100 + "\nc01 by hand: 'Which tube is used for the lab test that monitors Corvetrol...?'\n" + "=" * 100)
    show("lookup_entry('Corvetrol')        # hop 1: drug -> its monitoring test", lookup_entry("Corvetrol"), 600)
    show("lookup_entry('LC-0625')          # hop 2: test code -> tube + critical value", lookup_entry("LC-0625"), 600)

    print("\n" + "=" * 100 + "\nThe look-alike traps\n" + "=" * 100)
    show("lookup_entry('Corvatrel')        # one letter off: a different drug", lookup_entry("Corvatrel"), 120)
    show("lookup_entry('HV-RX-2047')       # digits swapped: a different code", lookup_entry("HV-RX-2047"), 120)
    show("lookup_entry('HV-RX-9999')       # made-up code", lookup_entry("HV-RX-9999"))
    show("lookup_entry('low molecular weight heparin')   # a drug class is not a name", lookup_entry("low molecular weight heparin"))
    show("lookup_entry('potassium')        # one name, two entries", lookup_entry("potassium"), 120)

    print("\n" + "=" * 100 + "\nThe same hop with search: ranked guesses, not one card\n" + "=" * 100)
    show("search('Corvetrol monitoring')", search("Corvetrol monitoring"), 200)


if __name__ == "__main__":
    main()
