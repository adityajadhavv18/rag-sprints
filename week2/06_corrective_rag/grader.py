"""Corrective RAG, the grader: one gpt-4o-mini call reads the question and the 5 chunks, and labels each one.

  relevant    states a fact that answers the question, or part of it, about the thing asked
  partial     about the right thing (same drug, test, device...) but not the fact asked for
  irrelevant  about something else

  verdict  correct    at least one chunk is relevant           -> answer
           ambiguous  none relevant, at least one partial      -> refine the query and search again
           incorrect  every chunk irrelevant                   -> refine the query and search again

The grader never answers. It only judges what retrieval brought back, like a fact-checker sorting clippings.

Run from rag_sprints/:  uv run week2/06_corrective_rag/grader.py
Searches and grades all 30 golden questions, then checks the grader against the evidence quotes: a chunk that holds
the evidence should never be called irrelevant, and no-answer questions should never get "correct".
Cost: 30 calls, about $0.01.
"""

import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from functools import cache
from pathlib import Path

from openai import OpenAI

from eval_harness import settings, traceable  # importing it loads .env, so OpenAI() finds the key
from eval_harness.golden import load_golden

LABELS = ("relevant", "partial", "irrelevant")
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens

# Sprint 03's notes, one per chunk: "this chunk covers HV-RX-2074 Corvetrol ...". Chunks often start mid-entry
# ("be opened or crushed. Renal adjustment: ..."), so without the note the grader can't tell which drug it is about.
NOTES_FILE = Path(__file__).resolve().parents[2] / "week1" / "03_contextual_rag" / "notes_medical.json"

# Labels only, no reasons: output tokens are the slow part of a call (sprint 04), and this runs on every question.
GRADE_PROMPT = """You check search results for an assistant that answers from one hospital's internal documents.
Each chunk starts with a note saying which document and entries it comes from; use it to tell what the chunk is about.
Label every numbered chunk below for the question, and reply as JSON: {{"grades": ["<label>", ...]}} \
with exactly one label per chunk, in order.
- relevant: the chunk states a fact that answers the question, or one part of it, about the exact thing asked \
(the same drug, code, test, device or protocol).
- partial: the chunk is about the thing asked, but does not state the fact the question wants.
- irrelevant: the chunk is about something else, even if it uses similar words.
A chunk about a different drug, code or device is irrelevant, even when it answers the same kind of question.

Question: {question}

Chunks:
{chunks}"""


@cache
def _openai() -> OpenAI:
    return OpenAI()


@cache
def _notes() -> dict:
    return {chunk_id: entry["note"] for chunk_id, entry in json.loads(NOTES_FILE.read_text()).items()}


def verdict(labels: list[str]) -> str:
    """The 3-way verdict from the per-chunk labels. One good chunk is enough to answer."""
    if "relevant" in labels:
        return "correct"
    return "ambiguous" if "partial" in labels else "incorrect"


@traceable(run_type="llm", name="grade_chunks")
def grade(question: str, hits: list) -> tuple[list[str], str, object]:
    """One grader call for all chunks. Returns (one label per chunk, verdict, token usage).

    A missing or unknown label becomes "partial": a broken reply should not throw away a chunk that might be good.
    """
    chunks = "\n\n".join(f"[{i}] Note: {_notes().get(h.chunk_id, h.doc_id)}\n{h.text}"
                         for i, h in enumerate(hits, start=1))
    resp = _openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,  # same chunks -> same labels, so runs are comparable
        response_format={"type": "json_object"},  # forces valid JSON
        messages=[{"role": "user", "content": GRADE_PROMPT.format(question=question, chunks=chunks)}],
    )
    raw = json.loads(resp.choices[0].message.content).get("grades", [])
    labels = [raw[i] if i < len(raw) and raw[i] in LABELS else "partial" for i in range(len(hits))]
    return labels, verdict(labels), resp.usage


# ---------- The check: grader vs the golden set's evidence quotes ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def holds_evidence(text: str, evidence: str | None) -> bool:
    """Same rule as the harness's evidence recall: any verbatim evidence fragment inside the chunk."""
    fragments = [_norm(f) for f in re.split(r" \.\.\. | / ", evidence or "") if f.strip()]
    return any(f in _norm(text) for f in fragments)


def load_contextual():
    """Sprint 03's search, loaded the same way sprints 04 and 05 do (its folder on the path, its file by path)."""
    os.environ.setdefault("RAG_CORPUS", "medical")
    if os.environ["RAG_CORPUS"] != "medical":
        raise SystemExit("Corrective RAG runs on the medical corpus only: unset RAG_CORPUS or set it to medical")
    ctx_dir = Path(__file__).resolve().parents[2] / "week1" / "03_contextual_rag"
    sys.path.append(str(ctx_dir))
    spec = importlib.util.spec_from_file_location("contextual_pipeline", ctx_dir / "pipeline.py")
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    return ctx


def main():
    ctx = load_contextual()
    rows, ev_labels, tokens_in, tokens_out, times = [], Counter(), 0, 0, []
    for g in load_golden("golden_medical_v1.jsonl"):
        hits = ctx.hybrid_search(g.question)
        t0 = time.perf_counter()
        labels, v, usage = grade(g.question, hits)
        times.append((time.perf_counter() - t0) * 1000)
        tokens_in += usage.prompt_tokens
        tokens_out += usage.completion_tokens
        ev = [holds_evidence(h.text, g.evidence) for h in hits]
        ev_labels.update(lab for lab, e in zip(labels, ev) if e)
        rows.append((g, labels, v, ev))

    print(f"\n{'id':4} {'type':10} {'verdict':9} {'ev':4}  labels (* = holds the evidence)")
    for g, labels, v, ev in rows:
        marks = " ".join(f"{lab[:3]}{'*' if e else ''}" for lab, e in zip(labels, ev))
        print(f"{g.id:4} {g.question_type:10} {v:9} {'yes' if any(ev) else 'no':4}  {marks}")

    print("\nVerdicts per question type:")
    for qtype in ("simple", "exact_term", "multi_part", "no_answer"):
        group = [r for r in rows if r[0].question_type == qtype]
        counts = Counter(r[2] for r in group)
        print(f"  {qtype:10} ({len(group):2})  " + "  ".join(f"{k}: {counts[k]}" for k in
                                                         ("correct", "ambiguous", "incorrect")))

    print(f"\nLabels on chunks that hold the evidence: {dict(ev_labels)}")
    lost = [r[0].id for r in rows if any(r[3]) and r[2] != "correct"]
    fooled = [r[0].id for r in rows if r[0].question_type == "no_answer" and r[2] == "correct"]
    blind = [r[0].id for r in rows if r[0].question_type != "no_answer" and not any(r[3]) and r[2] == "correct"]
    print(f"Evidence in top 5 but verdict not 'correct' (a needless retry): {lost or 'none'}")
    print(f"No-answer question graded 'correct' (grader fooled):            {fooled or 'none'}")
    print(f"Evidence missing but verdict 'correct' (no retry where needed):  {blind or 'none'}")

    times.sort()
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
    print(f"\nGrader latency p50 {times[len(times) // 2]:.0f} ms, max {times[-1]:.0f} ms | "
          f"{tokens_in // len(rows)} tokens in per call | cost ${cost:.4f} (${cost / len(rows):.6f} each)")


if __name__ == "__main__":
    main()
