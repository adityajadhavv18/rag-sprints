"""Adaptive RAG, the router: one gpt-4o-mini call reads the question and picks a path, before any search.

  no_rag   general knowledge or small talk -> the LLM answers directly, no search
  single   one fact from the hospital documents -> one search (sprint 03), then answer
  multi    one fact is needed to find another -> the search-read-search loop (multistep.py)

The dangerous mistake is a hospital question sent to no_rag: our drugs and codes are made up, so the LLM's own
memory would invent an answer. The prompt's examples are deliberately NOT from the test questions.

Run from rag_sprints/:  uv run week1/05_adaptive_rag/router.py
Routes the 30 golden + 10 extra questions and prints where each went. Cost: 40 tiny calls, about $0.002.
"""

import json
import time
from functools import cache
from pathlib import Path

from openai import OpenAI

from eval_harness import settings, traceable  # importing it loads .env, so OpenAI() finds the key
from eval_harness.golden import load_golden

ROUTES = ("no_rag", "single", "multi")
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens

ROUTER_PROMPT = """You route questions for an assistant that answers from one hospital's internal documents: \
drug formulary, lab test catalog, clinical protocols, policies, phone directory and device manuals. \
The hospital's drugs, codes and devices are unique to it, so general knowledge cannot answer questions about them.
Pick one route and reply as JSON: {{"route": "no_rag" | "single" | "multi", "reason": "<a few words>"}}
- no_rag: small talk, or general knowledge that does not depend on this hospital \
(e.g. "what does BP stand for?", "thanks!").
- single: needs facts from the documents, and the question names what to look up \
(e.g. "what is the dose of drug X?", "what does alarm code Y mean?").
- multi: needs one fact from the documents just to know what to look up next, because the question names \
something only indirectly (e.g. "what are the side effects of the antibiotic recommended in protocol Z?").
Rules: if the question could depend on this hospital's documents in any way, never pick no_rag. \
If unsure between single and multi, pick single.

Question: {question}"""


@cache
def _openai() -> OpenAI:
    return OpenAI()


@traceable(run_type="llm", name="route_question")
def route(question: str) -> tuple[str, str, object]:
    """One router call. Returns (route, reason, usage). An unknown route falls back to single, the safe middle."""
    resp = _openai().chat.completions.create(
        model=settings.llm_model,
        temperature=0,  # same question -> same route
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
    )
    reply = json.loads(resp.choices[0].message.content)
    chosen = reply.get("route") if reply.get("route") in ROUTES else "single"
    return chosen, reply.get("reason", ""), resp.usage


def load_questions() -> list[dict]:
    """40 questions with the route we expect. Golden questions all need the documents, so 'needs_docs'."""
    golden = [{"id": g.id, "question": g.question, "expected": "needs_docs"}
              for g in load_golden("golden_medical_v1.jsonl")]
    extra = [json.loads(line) for line in Path(__file__).with_name("questions_extra.jsonl").read_text().splitlines()]
    return golden + [{"id": q["id"], "question": q["question"], "expected": q["route"]} for q in extra]


def verdict(expected: str, chosen: str) -> str:
    if expected == "needs_docs":  # any search route is fine; skipping search is the dangerous mistake
        return "DANGER" if chosen == "no_rag" else "ok"
    if expected == "no_rag":  # searching for "what does mg mean" wastes money but can't invent a hospital fact
        return "ok" if chosen == "no_rag" else "wasted"
    return "ok" if chosen == "multi" else "missed chain"  # expected multi


def main():
    rows, tokens_in, tokens_out = [], 0, 0
    for q in load_questions():
        t0 = time.perf_counter()
        chosen, reason, usage = route(q["question"])
        ms = (time.perf_counter() - t0) * 1000
        tokens_in += usage.prompt_tokens
        tokens_out += usage.completion_tokens
        rows.append((q["id"], q["expected"], chosen, verdict(q["expected"], chosen), ms, reason, q["question"]))

    print(f"\n{'id':4} {'expected':11} {'route':7} {'verdict':13} {'ms':>5}  reason / question")
    for qid, exp, chosen, v, ms, reason, question in rows:
        print(f"{qid:4} {exp:11} {chosen:7} {v:13} {ms:5.0f}  {reason}  |  {question[:70]}")

    print("\nRoutes per group:")
    for exp in ("needs_docs", "no_rag", "multi"):
        group = [r for r in rows if r[1] == exp]
        counts = {route_: sum(r[2] == route_ for r in group) for route_ in ROUTES}
        print(f"  {exp:11} ({len(group):2} questions)  " + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    times = sorted(r[4] for r in rows)
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
    print(f"\nDANGER (hospital question sent to no_rag): {sum(r[3] == 'DANGER' for r in rows)}")
    print(f"Router latency p50 {times[len(times) // 2]:.0f} ms, max {times[-1]:.0f} ms | "
          f"cost ${cost:.4f} for {len(rows)} questions (${cost / len(rows):.6f} each)")


if __name__ == "__main__":
    main()
