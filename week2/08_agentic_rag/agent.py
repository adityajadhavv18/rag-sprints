"""Agentic RAG, the loop: the LLM picks a tool and its input, Python runs it, the result goes back, repeat.

  messages     the agent's memory: a plain list we append to and resend whole on every call
  each round   gpt-4o-mini sees the question, the rules, the 2 tool descriptions and everything so far, then either
               asks for tool calls (Python runs them and appends the results) or writes the answer (loop ends)
  round 1      tool_choice="required": it must look something up before answering (a rule in code, not a request)
  the brake    after MAX_ROUNDS tool rounds, one last call with tools switched off: answer from what you have,
               or say "I don't know"

Repeats are handled in code, not in the prompt (sprint 05's lesson): a tool hands back only items the agent has not
seen yet, and an identical call gets "you already ran this".

Run from rag_sprints/:  uv run week2/08_agentic_rag/agent.py
The 5 chain questions + 2 golden controls (m01, m22): every move printed, then each hop's evidence checked.
Cost: ~2 to 4 small gpt-4o-mini calls per question, well under $0.01. Needs Qdrant and sprint 03's collection.
"""

import json
import re
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from eval_harness import settings, traceable, traced_openai
from eval_harness.golden import load_golden
from tools import FUNCTIONS, TOOLS, Item, load_contextual

NO_ANSWER = load_contextual().hybrid.naive_pipeline.NO_ANSWER  # naive's exact refusal, so abstention is scored the same
MAX_ROUNDS = 5  # tool rounds; our longest chain (c05) needs 3 hops

# Naive's answer rules, plus how to use the tools. No hints from the test questions.
SYSTEM = f"""You answer questions using ONLY one hospital's internal documents, which you reach through two tools.
- Look things up before answering. Never answer from your own knowledge: the drugs, codes and devices are specific \
to this hospital.
- When the question names something indirectly ("the drug used for ...", "the test that monitors ..."), first find \
out which one it is, then look up the fact about it.
- Use search when you don't know the exact code or name. Use lookup_entry when the documents give you an exact code \
or name. Similar codes and names are different things.
- Stop as soon as the results answer every part of the question. You have at most {MAX_ROUNDS} rounds of tool calls.
- Then answer concisely with the specific figures from the results. If the results don't contain the answer, reply \
exactly: "{NO_ANSWER}\""""

CAP_MESSAGE = (f"Tool limit reached. Answer now using only the results above. If they don't contain the answer, "
               f"reply exactly: \"{NO_ANSWER}\"")


@dataclass
class Step:
    round: int
    tool: str
    args: dict
    new_ids: list[str]  # items this call added
    note: str = ""      # "already ran", "no entry found", "3 already seen" ...


@dataclass
class AgentResult:
    answer: str
    items: list[Item] = field(default_factory=list)  # everything any tool handed back, in order, no repeats
    steps: list[Step] = field(default_factory=list)
    rounds: int = 0          # tool rounds used
    hit_cap: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    messages: list = field(default_factory=list)


@cache
def _llm():
    return traced_openai()


def _format(items: list[Item]) -> str:
    return "\n\n".join(f"[{it.id}] ({it.doc_id})\n{it.text}" for it in items)


def run_tool(name: str, raw_args: str, round_no: int, result: AgentResult, seen_calls: set) -> str:
    """Run one tool call, add its new items to the result, and return the text the LLM gets back."""
    try:
        args = json.loads(raw_args or "{}")
        func = FUNCTIONS[name]
    except (json.JSONDecodeError, KeyError):
        result.steps.append(Step(round_no, name, {"raw": raw_args}, [], "bad call"))
        return f"Error: unknown tool or bad arguments. Tools: {', '.join(FUNCTIONS)}."

    key = (name, json.dumps(args, sort_keys=True).lower())
    if key in seen_calls:
        result.steps.append(Step(round_no, name, args, [], "already ran"))
        return "You already ran this exact call; its results are above. Try something different or answer."
    seen_calls.add(key)

    try:
        items = func(**args)
    except TypeError:
        result.steps.append(Step(round_no, name, args, [], "bad arguments"))
        return "Error: bad arguments for this tool."

    seen_ids = {it.id for it in result.items}
    new = [it for it in items if it.id not in seen_ids]
    result.items += new
    skipped = len(items) - len(new)

    if not items:
        note, text = "no entry found", f"No entry found for {next(iter(args.values()), '')!r}."
    else:
        note = f"{skipped} already seen" if skipped else ""
        text = _format(new) if new else "Nothing new: you already have every result this call returned."
        if new and skipped:
            text += f"\n\n({skipped} result(s) you already have were skipped.)"
    result.steps.append(Step(round_no, name, args, [it.id for it in new], note))
    return text


@traceable(run_type="chain", name="agent_loop")
def run_agent(question: str, max_rounds: int = MAX_ROUNDS) -> AgentResult:
    result = AgentResult(answer="")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    seen_calls: set = set()

    def call(**kwargs):
        resp = _llm().chat.completions.create(model=settings.llm_model, temperature=0, messages=messages, **kwargs)
        result.tokens_in += resp.usage.prompt_tokens
        result.tokens_out += resp.usage.completion_tokens
        return resp.choices[0].message

    for round_no in range(1, max_rounds + 1):
        msg = call(tools=TOOLS, tool_choice="required" if round_no == 1 else "auto")
        if not msg.tool_calls:  # it wrote the answer: done
            result.answer = (msg.content or "").strip()
            break
        result.rounds = round_no
        # Its own move goes into the diary, then one result per call, linked by tool_call_id
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            content = run_tool(tc.function.name, tc.function.arguments, round_no, result, seen_calls)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
    else:  # the brake: 5 tool rounds and still no answer
        result.hit_cap = True
        messages.append({"role": "user", "content": CAP_MESSAGE})
        result.answer = (call(tools=TOOLS, tool_choice="none").content or "").strip()

    result.answer = result.answer or NO_ANSWER
    result.messages = messages
    return result


# ---------- Demo: the 5 chains + 2 golden controls, every move printed, every hop checked ----------

EXTRA = Path(__file__).resolve().parents[2] / "week1" / "05_adaptive_rag" / "questions_extra.jsonl"
CONTROLS = ("m01", "m22")


def norm(text: str) -> str:  # same normaliser as the harness: collapse whitespace, lower case
    return re.sub(r"\s+", " ", text).strip().lower()


def hop_found(evidence: str, texts: list[str]) -> int | None:
    """Position (1-based) of the first text the LLM read that holds this hop's evidence quote, or None."""
    return next((i for i, t in enumerate(texts, start=1) if norm(evidence) in norm(t)), None)


def main():
    chains = [json.loads(line) for line in EXTRA.read_text().splitlines() if '"route": "multi"' in line]
    controls = [{"id": g.id, "question": g.question, "ground_truth": g.ground_truth, "hops": []}
                for g in load_golden("golden_medical_v1.jsonl") if g.id in CONTROLS]

    summary = []
    for q in chains + controls:
        print(f"\n{'=' * 100}\n{q['id']}  {q['question']}\n{'=' * 100}")
        t0 = time.perf_counter()
        r = run_agent(q["question"])
        secs = time.perf_counter() - t0

        for s in r.steps:
            arg = next(iter(s.args.values()), "")
            new = ", ".join(s.new_ids) or "-"
            print(f"  round {s.round}  {s.tool}({arg!r})\n           new: {new}" + (f"   [{s.note}]" if s.note else ""))
        if r.hit_cap:
            print("  ** hit the 5-round cap: forced final answer **")

        hops = []
        for i, hop in enumerate(q["hops"], start=1):
            pos = hop_found(hop["evidence"], [it.text for it in r.items])
            hops.append(pos is not None)
            print(f"  hop {i} ({hop['doc'].removeprefix('hvmc_')}): {'item #' + str(pos) if pos else 'MISSED'}")

        print(f"\n  TRUTH  {q['ground_truth']}")
        print(f"  AGENT  ({secs:.1f}s, {r.rounds} rounds, {len(r.steps)} tool calls, {len(r.items)} items, "
              f"{r.tokens_in}+{r.tokens_out} tokens)\n         {r.answer}")
        summary.append((q["id"], hops, r.rounds, len(r.steps), len(r.items), r.hit_cap, secs, r.tokens_in, r.tokens_out))

    print(f"\n{'=' * 100}\nSUMMARY  (hops found = that hop's evidence reached the LLM)\n{'=' * 100}")
    print(f"{'id':5} {'hops found':>11} {'rounds':>7} {'calls':>6} {'items':>6} {'cap':>4} {'secs':>6} {'tokens':>12}")
    total_in = total_out = 0
    for qid, hops, rounds, calls, items, cap, secs, tin, tout in summary:
        total_in, total_out = total_in + tin, total_out + tout
        print(f"{qid:5} {(f'{sum(hops)}/{len(hops)}' if hops else '-'):>11} {rounds:>7} {calls:>6} {items:>6} "
              f"{'YES' if cap else '':>4} {secs:>6.1f} {f'{tin}+{tout}':>12}")
    print(f"\ncost: ${(total_in * 0.15 + total_out * 0.60) / 1e6:.4f} for {len(summary)} questions (gpt-4o-mini)")


if __name__ == "__main__":
    main()
