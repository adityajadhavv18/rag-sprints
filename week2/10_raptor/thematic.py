"""The thematic check: 5 broad questions (questions_thematic.jsonl), each needing ~5 points spread over 3-4 documents.

The golden set is all fact lookups, which is not what RAPTOR is for. These questions are. Golden v1 stays untouched;
the harness's evaluate() takes one evidence quote per question, so these are scored here instead, two ways:

  quote coverage   free: how many of the 5 points' verbatim quotes are in the chunks the LLM read
  point coverage   one gpt-4o-mini call: how many of the 5 key points the ANSWER contains

Run from rag_sprints/:
  uv run week2/10_raptor/thematic.py --check        # every quote sits inside one sprint-03 chunk (free)
  uv run week2/10_raptor/thematic.py [raptor]       # Contextual, GraphRAG, RAPTOR on the 5 questions (~$0.005)
Results are saved to thematic_results.json (one entry per pipeline, re-runs overwrite that entry).
"""

import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path

from eval_harness import settings
from eval_harness.metrics import _norm
from openai import OpenAI

os.environ.setdefault("RAG_CORPUS", "medical")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUESTIONS = HERE / "questions_thematic.jsonl"
RESULTS = HERE / "thematic_results.json"
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, $ per million tokens

JUDGE_PROMPT = """Here is a question, an answer, and a numbered list of key points a complete answer would make.
For each key point, decide whether the answer states it (the same fact, in any wording). Reply as JSON:
{{"covered": [<true|false>, ...]}}  (one entry per key point, in order)

Question: {question}

Answer: {answer}

Key points:
{points}"""


def _load(name: str, path: Path):
    """Load another sprint's file by path (folders start with digits, so a normal import can't reach them)."""
    sys.path.append(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _openai() -> OpenAI:
    return OpenAI()


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def quote_positions(q: dict, contexts: list[str]) -> list[int | None]:
    """For each point: the position (1-based) of the first chunk holding its quote, or None."""
    return [next((i for i, text in enumerate(contexts, 1) if _norm(p["evidence"]) in _norm(text)), None)
            for p in q["points"]]


def judge_points(q: dict, answer: str) -> tuple[list[bool], int, int]:
    """Which key points the answer states. Returns (covered per point, tokens in, tokens out)."""
    points = "\n".join(f"{i}. {p['point']}" for i, p in enumerate(q["points"], 1))
    resp = _openai().chat.completions.create(
        model=settings.llm_model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=q["question"], answer=answer,
                                                                    points=points)}])
    covered = [bool(c) for c in json.loads(resp.choices[0].message.content).get("covered", [])]
    covered = (covered + [False] * len(q["points"]))[:len(q["points"])]
    return covered, resp.usage.prompt_tokens, resp.usage.completion_tokens


def run(name: str, pipeline, questions: list[dict]) -> dict:
    """Run one pipeline on the thematic questions, print each answer, and return its scores."""
    rows, cost = [], 0.0
    for q in questions:
        out = pipeline(q["question"])
        covered, j_in, j_out = judge_points(q, out.answer)
        cost += (out.tokens_in * PRICE_IN + out.tokens_out * PRICE_OUT) / 1e6
        judge_cost = (j_in * PRICE_IN + j_out * PRICE_OUT) / 1e6
        positions = quote_positions(q, out.contexts)
        rows.append({"id": q["id"], "answer": out.answer, "context_ids": out.context_ids,
                     "quote_positions": positions, "points_covered": covered, "judge_cost": judge_cost})
        docs = sorted({p["doc"] for p, pos in zip(q["points"], positions) if pos})
        print(f"\n  [{name}] {q['id']}  quotes {sum(p is not None for p in positions)}/5 "
              f"{['#' + str(p) if p else '-' for p in positions]}  points {sum(covered)}/5  "
              f"docs read: {sorted(set(out.context_ids))}")
        print(f"    {out.answer}")
    n = len(questions) * len(questions[0]["points"])
    summary = {"quote_coverage": round(sum(p is not None for r in rows for p in r["quote_positions"]) / n, 2),
               "point_coverage": round(sum(c for r in rows for c in r["points_covered"]) / n, 2),
               "cost_per_query": round(cost / len(questions), 5)}
    return {"summary": summary, "questions": rows}


def check_quotes():
    """Every quote must sit inside ONE sprint-03 chunk, or no retrieved leaf could ever match it."""
    ctx = _load("contextualize", ROOT / "week1" / "03_contextual_rag" / "contextualize.py")
    chunks = ctx.load_chunks()
    ok = True
    for q in load_questions():
        for p in q["points"]:
            hits = [c.chunk_id for c in chunks if _norm(p["evidence"]) in _norm(c.text)]
            ok &= bool(hits)
            print(f"{q['id']} {'OK ' if hits else 'BAD'} {p['doc']:<26} chunks {hits}")
    print("\nall quotes inside a single chunk" if ok else "\nSOME QUOTES CROSS A CHUNK BOUNDARY")


def main():
    if "--check" in sys.argv:
        return check_quotes()
    questions = load_questions()
    # Sprint 09 has its own retrieve.py and pipeline.py: load it with ITS folder first on the path, then forget
    # those module names so the import below finds this folder's files
    graph_dir = str(ROOT / "week2" / "09_graph_rag")
    sys.path.insert(0, graph_dir)
    try:
        graph = _load("graph_pipeline", ROOT / "week2" / "09_graph_rag" / "pipeline.py")
    finally:
        sys.path = [p for p in sys.path if p != graph_dir]
        for name in ("retrieve", "pipeline", "build_graph"):
            sys.modules.pop(name, None)
    from pipeline import raptor_pipeline  # this folder's pipeline.py (imported here: it imports this module)
    pipelines = {"contextual": graph._ctx.contextual_pipeline, "graph_rag": graph.graph_rag_pipeline,
                 "raptor": raptor_pipeline}
    only = [a for a in sys.argv[1:] if a in pipelines]  # e.g. `thematic.py raptor` re-runs just that one

    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for name, fn in pipelines.items():
        if only and name not in only:
            continue
        print(f"\n{'=' * 100}\n{name}")
        saved[name] = run(name, fn, questions)
    RESULTS.write_text(json.dumps(saved, indent=1))

    print(f"\n{'=' * 100}\n{'pipeline':<14}{'quote coverage':>16}{'point coverage':>16}{'$/query':>10}")
    for name, r in saved.items():
        s = r["summary"]
        print(f"{name:<14}{s['quote_coverage']:>16}{s['point_coverage']:>16}{s['cost_per_query']:>10}")


if __name__ == "__main__":
    main()
