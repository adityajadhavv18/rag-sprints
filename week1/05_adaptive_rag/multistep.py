"""Adaptive RAG, the multi-step route: search, read, search again with what was learned, then answer.

  round 1      search with the user's question (Contextual's hybrid search, top 5)
  read         gpt-4o-mini reads every chunk found so far and writes what it FOUND, what is MISSING, and the
               next search. Nothing missing -> stop.
  round 2, 3   search with next_query, add the NEW chunks to the pile (a repeat search stops the loop)
  answer       naive's generate() on the whole pile, with the ORIGINAL question

The point: round 2's search words come from the documents, not the user. "Corvetrol" finds the formulary entry,
which says "aPTT (LC-0625)"; only then can the loop search for LC-0625 in the lab catalog.
MAX_ROUNDS is the brake: a confused reader can't keep searching forever.

Requires Qdrant running (docker start qdrant) and Contextual's collection from sprint 03. Medical corpus only.
"""

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from openai import OpenAI

from eval_harness import PipelineOutput, settings, traceable

os.environ.setdefault("RAG_CORPUS", "medical")
if os.environ["RAG_CORPUS"] != "medical":
    raise SystemExit("Adaptive RAG runs on the medical corpus only: unset RAG_CORPUS or set it to medical")

# Reuse Contextual's search (same trick as sprint 04): its folder goes on the path so its `import ingest` finds
# its own ingest.py, and its pipeline.py is loaded by path under another name.
_CTX_DIR = Path(__file__).resolve().parents[1] / "03_contextual_rag"
sys.path.append(str(_CTX_DIR))
_spec = importlib.util.spec_from_file_location("contextual_pipeline", _CTX_DIR / "pipeline.py")
ctx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctx)
generate = ctx.hybrid.naive_pipeline.generate  # naive's answer step, unchanged since sprint 00

MAX_ROUNDS = 3  # our longest chain (c05) needs 3 searches
TOP_K = ctx.TOP_K  # 5 chunks per search

# The reader never answers. It only judges the pile and, if something is missing, writes the next search.
# "Use the exact codes and names from the notes" is the whole trick: BM25 matches LC-0625 exactly.
READ_PROMPT = """You are gathering facts from hospital documents to answer a question.
Check the notes below against every part of the question, then reply as JSON:
{{"found": "<the facts in the notes that answer the question, briefly>",
  "missing": "<the part of the question the notes do not answer yet, or empty if nothing is missing>",
  "next_query": "<a short search for the missing part, or empty if nothing is missing>"}}
Only count a fact as found if a note states it about the thing the question asks about. When the question names \
something indirectly ("the drug used for ...", "the test that monitors ..."), a note must first say which one it is; \
until then, that is what is missing.
The next query must name the exact codes, drug names or test names that appear in the notes, because the search \
engine matches exact words. Search for the missing part only, and do not repeat an earlier search.

Earlier searches: {searches}

Question: {question}

Notes:
{notes}"""


@dataclass
class Round:
    query: str  # what was searched this round
    new_hits: list  # chunks this search found that the pile didn't have yet
    decision: dict  # the reader's JSON after this round


@dataclass
class MultiStepResult:
    hits: list  # the whole pile, in the order found: round 1's chunks first
    rounds: list[Round] = field(default_factory=list)
    tokens_in: int = 0  # reader calls only; the answer call is counted by the pipeline
    tokens_out: int = 0


@cache
def _openai() -> OpenAI:
    return OpenAI()


def is_done(decision: dict) -> bool:
    """The reader is done when it names nothing missing, or has no search to run for it."""
    return not decision.get("missing", "").strip() or not decision.get("next_query", "").strip()


@traceable(run_type="llm", name="read_notes")
def read_notes(question: str, hits: list, searches: list[str]) -> tuple[dict, object]:
    """One reader call: is the pile enough? Returns its JSON decision and the token usage."""
    notes = "\n\n".join(f"[{i}] ({h.doc_id})\n{h.text}" for i, h in enumerate(hits, start=1))
    resp = _openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,  # same pile -> same decision, so runs are comparable
        response_format={"type": "json_object"},  # forces valid JSON
        messages=[{"role": "user", "content": READ_PROMPT.format(
            searches="; ".join(searches), question=question, notes=notes)}],
    )
    return json.loads(resp.choices[0].message.content), resp.usage


@traceable(run_type="retriever", name="multi_step_search")
def multi_step_search(question: str, max_rounds: int = MAX_ROUNDS) -> MultiStepResult:
    result = MultiStepResult(hits=[])
    seen = set()  # chunk_ids already in the pile, so a repeat search adds nothing twice
    query = question  # round 1 uses the user's own words
    for _ in range(max_rounds):
        new = [h for h in ctx.hybrid_search(query) if h.chunk_id not in seen]
        seen.update(h.chunk_id for h in new)
        result.hits += new
        decision, usage = read_notes(question, result.hits, [r.query for r in result.rounds] + [query])
        result.tokens_in += usage.prompt_tokens
        result.tokens_out += usage.completion_tokens
        result.rounds.append(Round(query, new, decision))
        if is_done(decision):
            break  # nothing missing, or nothing left to search for
        if decision["next_query"].strip().lower() in {r.query.strip().lower() for r in result.rounds}:
            break  # a repeat search can only bring back chunks we already have
        query = decision["next_query"]
    return result


@traceable(run_type="chain", name="multi_step_pipeline")
def multi_step_pipeline(question: str) -> PipelineOutput:
    """The multi route as a harness pipeline. contexts = the whole pile, which is exactly what the LLM read."""
    found = multi_step_search(question)
    resp = generate(question, found.hits)
    return PipelineOutput(
        answer=resp.choices[0].message.content.strip(),
        contexts=[h.text for h in found.hits],
        context_ids=[h.doc_id for h in found.hits],
        tokens_in=resp.usage.prompt_tokens + found.tokens_in,
        tokens_out=resp.usage.completion_tokens + found.tokens_out,
    )
