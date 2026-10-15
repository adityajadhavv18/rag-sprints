"""Corrective RAG, the retry loop: grade what search found; if it isn't good enough, refine the query and search again.

  round 1     search with the user's question (Contextual's hybrid search, top 5), grade the chunks
  correct?    stop: at least one chunk answers the question
  otherwise   gpt-4o-mini writes a refined search, using the notes of the "partial" chunks as hints
  round 2, 3  search with the refined query, grade only the NEW chunks
  still not correct after round 3 -> record a "web search needed" event. No web call is made.

Relevant and partial chunks are kept across rounds; irrelevant ones are dropped. MAX_ROUNDS is the brake.

Run from rag_sprints/:  uv run week2/06_corrective_rag/retry.py
Runs the loop on 10 questions (the retries the grader asked for, plus 2 controls) and prints every round. ~$0.01.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from eval_harness import settings, traceable
from eval_harness.golden import load_golden
from grader import _notes, _openai, grade, load_contextual

MAX_ROUNDS = 3  # the first search + up to 2 refined searches
EVENTS_FILE = Path(__file__).with_name("web_search_events.jsonl")
ctx = load_contextual()

# Keyword style, not sentences: sprint 05's wordy next-searches missed. The hints are the notes of chunks that were
# about the right thing but lacked the fact, so the refined query can borrow the documents' own words.
REFINE_PROMPT = """A search over one hospital's internal documents did not find the answer to a question.
Write ONE new search query and reply as JSON: {{"query": "<the query>"}}
- Short keywords, not a sentence (at most 10 words).
- Keep every code and name from the question exactly as written.
- Use the clinical words a hospital document would use, not the everyday words of the question.
- Do not repeat an earlier search, and do not add facts or numbers the question doesn't give.

Question: {question}
Earlier searches: {searches}
Close but not enough (what the last search found about the right topic):
{hints}"""


@dataclass
class Round:
    query: str  # what was searched this round
    new_hits: list  # chunks not seen in an earlier round
    labels: list[str]  # the grader's label for each new chunk
    verdict: str  # verdict over everything kept so far


@dataclass
class RetryResult:
    kept: list  # relevant chunks first, then partial ones, each in the order found
    rounds: list[Round] = field(default_factory=list)
    web_search_needed: bool = False  # True when 3 rounds didn't find a relevant chunk
    tokens_in: int = 0  # grader + refine calls; the answer call is counted by the pipeline
    tokens_out: int = 0


@traceable(run_type="llm", name="refine_query")
def refine(question: str, searches: list[str], partial_hits: list) -> tuple[str, object]:
    hints = "\n".join(f"- {_notes().get(h.chunk_id, h.doc_id)}" for h in partial_hits) or "(nothing)"
    resp = _openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": REFINE_PROMPT.format(
            question=question, searches="; ".join(searches), hints=hints)}],
    )
    return json.loads(resp.choices[0].message.content).get("query", "").strip(), resp.usage


@traceable(run_type="tool", name="web_search_needed")
def record_web_search_event(question: str, result: RetryResult) -> dict:
    """The fallback we don't run: log that the documents ran out, and what was tried. Shows up in LangSmith too."""
    event = {"time": datetime.now().isoformat(timespec="seconds"), "question": question,
             "queries_tried": [r.query for r in result.rounds],
             "last_verdict": result.rounds[-1].verdict if result.rounds else None}
    with EVENTS_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")
    return event


@traceable(run_type="retriever", name="corrective_search")
def corrective_search(question: str, max_rounds: int = MAX_ROUNDS) -> RetryResult:
    result = RetryResult(kept=[])
    relevant, partial, seen = [], [], set()
    query = question  # round 1 uses the user's own words
    for n in range(max_rounds):
        new = [h for h in ctx.hybrid_search(query) if h.chunk_id not in seen]
        seen.update(h.chunk_id for h in new)
        labels = []
        if new:  # a search that brings back only chunks we already graded costs no grader call
            labels, _, usage = grade(question, new)
            result.tokens_in += usage.prompt_tokens
            result.tokens_out += usage.completion_tokens
        relevant += [h for h, lab in zip(new, labels) if lab == "relevant"]
        partial += [h for h, lab in zip(new, labels) if lab == "partial"]
        verdict = "correct" if relevant else "ambiguous" if partial else "incorrect"
        result.rounds.append(Round(query, new, labels, verdict))
        if verdict == "correct" or n == max_rounds - 1:
            break
        query, usage = refine(question, [r.query for r in result.rounds], partial)
        result.tokens_in += usage.prompt_tokens
        result.tokens_out += usage.completion_tokens
        if not query or query.lower() in {r.query.lower() for r in result.rounds}:
            break  # nothing new to search for
    result.kept = relevant + partial
    if not relevant:
        result.web_search_needed = True
        record_web_search_event(question, result)
    return result


def main():
    # The 4 retries the grader asked for (m04, m09, m20, m27) + the 3 "incorrect" no-answers + m11 (missed by the
    # grader) + 2 controls that should stop after round 1.
    ids = ["m04", "m09", "m20", "m27", "m28", "m29", "m30", "m11", "m01", "m21"]
    golden = {g.id: g for g in load_golden("golden_medical_v1.jsonl")}
    from grader import holds_evidence
    tokens_in = tokens_out = 0
    for qid in ids:
        g = golden[qid]
        res = corrective_search(g.question)
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        ev = any(holds_evidence(h.text, g.evidence) for h in res.kept)
        print(f"\n{qid} ({g.question_type}) {g.question}")
        for i, r in enumerate(res.rounds, start=1):
            marks = " ".join(f"{lab[:3]}{'*' if holds_evidence(h.text, g.evidence) else ''}"
                             for h, lab in zip(r.new_hits, r.labels)) or "(no new chunks)"
            print(f"  round {i}: {r.verdict:9} {marks:32} | {r.query[:70]}")
        print(f"  -> kept {len(res.kept)} chunks, evidence kept: {'yes' if ev else 'no'}"
              f"{'  | WEB SEARCH NEEDED (recorded)' if res.web_search_needed else ''}")
    cost = (tokens_in * 0.15 + tokens_out * 0.60) / 1e6
    print(f"\nCost ${cost:.4f} for {len(ids)} questions. Events appended to {EVENTS_FILE.name}")


if __name__ == "__main__":
    main()
