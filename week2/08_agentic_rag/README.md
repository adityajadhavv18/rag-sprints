# 08 Agentic RAG

Give the LLM two tools and a budget, and let it decide each move: which tool, with what input, and when it has enough.

```
question → agent (gpt-4o-mini + 2 tool descriptions)
             ├─ asks for tool calls → Python runs them → results added to the conversation → agent again
             └─ no tool call        → that message is the answer
           round 1: must call a tool (it can't answer from memory)
           brake: 5 tool rounds → one last call with tools switched off: "answer from what you have, or say I don't know"

tools:  search(query)              sprint 03's Contextual hybrid search → 5 chunks, each with its note
        lookup_entry(code_or_name) one whole formulary / lab / protocol entry, exact code or name only
```

Think of a researcher in a library. `search` is browsing the shelves for books that look related. `lookup_entry` is walking straight to call number LC-0625. In sprint 05 the code told the researcher what to do next. Here the researcher decides, and we only hold the stopwatch.

This follows the ReAct pattern (Yao et al., 2022): the model alternates between thinking about what it needs and acting with a tool, and every result feeds the next decision. Here the "acting" is OpenAI's native tool calling in a plain Python loop. There's no LangGraph, so the wiring is all visible in `agent.py`. There's also no web search: the corpus is a private hospital. Medical corpus only.

## How it works

**The two tools (`tools.py`).** Each tool is a normal Python function plus a short description the LLM reads.

- **`search(query)`** is sprint 03's hybrid search, unchanged. Each chunk comes with its sprint 03 note on top, because chunks often start mid-entry, and without the note the LLM can't tell which drug a chunk is about (sprint 06's lesson).
- **`lookup_entry(code_or_name)`** is a dictionary of 357 entries (256 formulary, 76 lab, 25 protocols), built by splitting the files at blank lines. It matches exactly and never returns the "closest" entry. A code anywhere in the input wins; otherwise the input must equal a whole name ("Corvetrol", "aPTT", "potassium"). A bracket counts as a name only when it's one word: in the formulary the bracket is the drug class, and "low molecular weight heparin" must not return every heparin.

| `lookup_entry(...)` | Returns |
|---|---|
| `Corvetrol` | HV-RX-2074 only |
| `Corvatrel` (one letter off) | HV-RX-2047, a different drug |
| `HV-RX-9999` (made up) | nothing ("no entry found") |
| `low molecular weight heparin` (a class) | nothing |
| `potassium` | LC-0412 and LC-0414 (two lab entries) |

Hybrid search would always return 5 "closest" chunks, even for a made-up code. Lookup either finds the exact entry or says so.

**How the LLM "uses" a function.** It doesn't run anything. It replies with an order slip, `{"name": "lookup_entry", "arguments": "{\"code_or_name\": \"LC-0870\"}"}`, and our code turns that into a normal function call:

```python
args  = json.loads(tc.function.arguments)   # text → dict
func  = FUNCTIONS[tc.function.name]         # name → the real function
items = func(**args)                        # same as lookup_entry(code_or_name="LC-0870")
```

**The loop (`agent.py`).** Memory is one Python list, `messages`. The LLM keeps nothing between calls, so every round we resend the whole list, and it rereads its own earlier moves.

```python
messages = [system rules, question]
for round in 1..5:                                   # the brake
    msg = LLM(messages, tools, tool_choice="required" if round == 1 else "auto")
    if no tool calls: answer = msg; stop
    messages += its tool calls
    for each call: messages += run_tool(call)        # linked by tool_call_id
else:                                                # 5 rounds, still no answer
    answer = LLM(messages + "answer now", tool_choice="none")
```

- **Round 1 must call a tool.** That's a rule in code, so the agent can't answer "what's Corvetrol's dose" from its own imagination.
- **Repeats are handled in code, not in the prompt** (sprint 05's lesson). A tool hands back only items the agent hasn't seen, plus "3 results you already have were skipped". An identical call gets "you already ran this".
- **The agent writes the answer itself.** It answers from its own trail ("Corvetrol pointed to LC-0625, and the LC-0625 card says above 100 s"), not from a cold pile. Sprint 05's separate `generate()` re-read 8 chunks and took the wrong critical value on c01. The system prompt uses naive's rules and its exact "I don't know" sentence.
- **The LLM can ask for several tools in one round.** It often does: one search per part of the question.

**What the harness gets (`pipeline.py`).** `contexts` holds every item any tool returned, in arrival order, without repeats: exactly what the LLM read, notes included. `tokens_in/out` add up every agent call, including the forced final one.

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week2/08_agentic_rag/tools.py      # both tools by hand: c01's two hops + the look-alike traps (free)
uv run week2/08_agentic_rag/agent.py      # the loop on the 5 chains + m01, m22, every move printed (~$0.005)
uv run week2/08_agentic_rag/run_eval.py   # judged eval vs Corrective, then the 5 chains end to end (~$0.10 incl. judge)
```

## Results (medical corpus)

| Metric | Contextual hybrid | Adaptive | Corrective | **Agentic** |
|---|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | 0.92 / 0.75 | 0.96 / **0.86** | **1.00** / 0.71 |
| Context recall / precision | 0.92 / 0.76 | 0.91 / 0.76 | 0.95 / **0.90** | **0.97** / 0.84 |
| Answer relevancy | 0.85 | 0.85 | **0.90** | **0.90** |
| Faithfulness on real answers | 0.92 | 0.92 | 0.91 | **0.94** |
| Refused although answerable | 1 / 26 | 1 / 26 | **0 / 26** | **0 / 26** |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 | 1.00 |
| Items the LLM reads (mean) | 5 | 5+ | **1.7** | 5.3 |
| Prompt tokens per question (mean) | 1,921 | – | 3,437 | 4,949 |
| Latency p50 / p95 | **1.2 s** / – | 3.0 s / 7.6 s | 2.6 s / 5.5 s | 2.5 s / 5.3 s |
| Cost per query | **$0.00031** | $0.00049 | $0.00056 | $0.00080 |

**In one line:** the first sprint to get the evidence in front of the LLM on every answerable question, as fast as Corrective, at about 1.4× its cost and 2.6× Contextual's.

**Moves used (30 golden questions):** 23 questions needed 1 round, 3 needed 2, 1 needed 3, 1 needed 4, and 2 hit the 5-round cap. In total it made 39 searches and 16 lookups. It searches first unless the question already contains a code (m01 "HV-RX-2074" was a single lookup with 1,079 prompt tokens).

**Why MRR dropped while recall rose.** `contexts` is in arrival order, not ranked. When two searches run in one round, the second search's #1 sits at #6, and an entry fetched in round 2 lands after everything from round 1. The agent reads its whole diary anyway, so the low rank doesn't hurt the answer (see "Ideas not built").

### The 5 chain questions

| Chain | Adaptive (sprint 05) | **Agentic** | Moves |
|---|---|---|---|
| c01 Corvetrol's test tube | right tube, **wrong** critical value | ✅ light blue, above 100 s | search → `lookup_entry('LC-0625')` |
| c02 Thyrolex's test | "don't know" | ✅ TSH, mint green, 24 h | search → `lookup_entry('LC-0870')` (1st run); 2 searches (2nd run) |
| c03 DKA ketone tube | ✅ | ✅ | 2 searches in round 1 |
| c04 Vasopressor + Vancorin | ✅ | ✅ incompatible | 2 searches in round 1 |
| c05 Vasopressor leak → extension | ✅ 4200 | ❌ **5555**, then **3610** | 2 searches, then stopped |

4 of 5 right vs Adaptive's 3. Every hop's evidence reached the LLM except c05's last one (the directory).

### Per question

| Question | Before | Agentic | Why |
|---|---|---|---|
| m11 "sugar crashed… what should the nurse give?" | vague in 00–06; fixed by Self-RAG (07) | **"Dextriva 25 g (50 mL) IV push"** | It kept searching after the first low-sugar chunks and reached the evidence. The judge still scored faithfulness 0.33 (judge noise, as with m12) |
| c02 Thyrolex | Adaptive: "don't know" | ✅ | The search showed "TSH (LC-0870)", and `lookup_entry('LC-0870')` brought the whole card. Sprint 05's wordy second search never found it |
| m22 Corvatrel + Nephrazole | ✅ | ✅ | Two lookups in parallel. It tried `lookup_entry('Corvatrel')` again in round 2 and the code answered "you already ran this" |
| m29, m30 (no answer) | "I don't know" | "I don't know" after the cap | The two cap hits, judging by their prompt tokens (22k and 19k vs a 4.9k mean): it kept looking for something that isn't there |

## What it gets wrong

- **Its "do I have enough?" call can be wrong (c05).** It read "call pharmacy for the antidote" and stopped without looking up pharmacy's extension. It then took a number from a chunk it already had: 5555 (rapid response team) in one run, 3610 (Medical ICU) in the next. It didn't run out of moves; it stopped too early. Sprint 05's separate reader got this one right.
- **No-answer questions are the expensive ones.** It can't tell "not found yet" from "not in the documents", so it searches until the cap. m29 and m30 cost about 4× an average question. The brake did its job: both still ended in "I don't know".
- **The diary grows every round.** Each call resends every chunk so far, so prompt tokens average 4,949 vs Corrective's 3,437. That's why it's the most expensive sprint yet.
- **Same question, different path.** Even at temperature 0, c02 went search → lookup in one run and two searches in the next, and c05 gave two different wrong extensions. The final answer was the same on c02, but you can't predict the moves.
- **It reads more junk.** 5.3 items on average vs Corrective's 1.7; context precision is 0.84 vs 0.90. Junk in the diary is what c05 picked its wrong number from.
- **Lookup is underused.** Only 16 of 55 calls. On c03 it ran a second search where `lookup_entry('LC-0452')` would have been exact. It worked because the search happened to find the entry. On c01 it also fetched LC-0620 (the INR test), which it didn't need.

## What it gets right

- **Evidence recall 1.00.** Every answerable question's evidence reached the LLM. m11 is answered correctly, which only Self-RAG (07) had managed before, and m20 too, which Self-RAG lost.
- **Zero false refusals, abstention still 1.00.** It matches Corrective on refusals, with the best faithfulness so far (0.94).
- **Chains without a fixed script.** c02 is the clean case: find the code, then fetch the exact entry. Nobody told it to do that for this question.
- **Easy questions stay cheap.** A code in the question means one lookup, one round and about 1k prompt tokens.
- **Exact lookups can't confuse look-alikes.** HV-RX-2047 vs HV-RX-2074 and Corvatrel vs Corvetrol are different dictionary keys.
- **The brake holds.** 2 cap hits out of 35 questions, both ending safely in "I don't know".

## Ideas not built

- **Fuse and trim before answering.** RRF over each tool call's ranked list (a lookup is a list of 1), then answer from the top 5. This would raise MRR and precision and cut tokens, and might have dropped c05's stray extension. But it could also drop a hop, and it gives up "answer from its own trail". Just re-sorting `contexts` would raise MRR without changing anything the LLM saw, so that alone would be gaming the metric.
- **Stop when the tools go dry.** If a round brings back nothing new, force the answer. This would cut m29/m30's extra rounds, with the rule held in code.
- **A check before answering.** One cheap call asking "does each part of the question have a result about the exact thing asked?" before the agent may stop. That targets c05, but it adds a call to every question. It's sprint 06's grader idea moved to the end of the loop.
- **Push lookup harder.** Have `search` results list the codes they mention, so the next move is more likely an exact lookup than another fuzzy search.
- **A classifier as the loop gate.** The plan's ⚡ item: a small model decides "enough / keep going". Dropped from this sprint.

## When I'd reach for it

When questions need different numbers of steps and you can't script the path in advance: one question is a single code lookup, the next is a three-document chain. Internal assistants over linked records (drug → test → lab, order → shipment → carrier) are the natural fit, especially when some sources have exact keys a tool can fetch. Skip it when one search nearly always finds the answer, or when cost and predictability matter more than chains. Then Corrective is cheaper and steadier, and Contextual hybrid is cheaper still.

## Files

| File | Purpose |
|---|---|
| `tools.py` | `search`, `lookup_entry` (the 357-entry index), the `TOOLS` descriptions the LLM reads, `FUNCTIONS` (name → function), and a by-hand demo |
| `agent.py` | The system prompt, `run_tool`, `run_agent` (the loop with the 5-round brake), and a demo on the 5 chains + 2 golden controls with a per-hop evidence check |
| `pipeline.py` | `agentic_pipeline`, the harness entry point: agent → answer, every item read → `contexts` |
| `run_eval.py` | Judged eval on golden v1 (`arch="agentic"`) vs Corrective, then the 5 chains end to end |
