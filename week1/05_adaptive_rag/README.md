# 05 Adaptive RAG

Read the question first, then pick how much retrieval it needs: none, one search, or a search loop that follows clues.

```
question → router (gpt-4o-mini, question only) ─┬─ no_rag → the LLM answers directly
                                                ├─ single → Contextual hybrid search (sprint 03) → answer
                                                └─ multi  → search → reader: "what's missing?" → next search … (max 3) → answer
```

Think of a triage nurse. A cold gets two minutes; chest pain gets the full workup. The router is the nurse, and the multi-step loop is the full workup.

This follows the Adaptive-RAG paper (Jeong et al., 2024): no retrieval, single-step, or multi-step. Here an LLM picks the route; the paper used a small trained classifier. Medical corpus only.

## Why we needed extra questions

All 30 golden questions need the documents. The drugs and codes are made up, so `no_rag` is never right on them, and one search already gets 0.92 evidence recall. A router has nothing to win there. So `questions_extra.jsonl` adds 10 sprint-local questions (golden v1 is untouched):

- **5 that need no search:** "what does mg stand for?", "how many mL in a litre?", a greeting…
- **5 chain questions:** the question names something only indirectly, so one fact is needed to find the next.
  Example c01: "Which tube is used for the lab test that monitors **Corvetrol**?" The formulary says Corvetrol is monitored by **aPTT (LC-0625)**, and only the lab catalog says LC-0625 needs a light blue tube. Each hop has a verbatim evidence quote, so the test can check which hops reached the LLM.

## How it works

**Router (`router.py`).** One gpt-4o-mini call, temperature 0, JSON: `{"route": ..., "reason": ...}`. The prompt describes the three routes with examples that are not from the test questions ("drug X", "protocol Z"). It has two rules: anything that could depend on this hospital is never `no_rag`, and when unsure between single and multi, pick single. An unknown reply falls back to `single`.

**Multi-step loop (`multistep.py`).**

```python
query = question                       # round 1: the user's own words
for _ in range(3):                     # the brake
    pile += new chunks from hybrid_search(query)
    decision = reader(question, pile)  # {"found": ..., "missing": ..., "next_query": ...}
    if nothing missing: break
    if next_query was already searched: break
    query = decision["next_query"]     # the next search comes from the DOCUMENTS, not the user
answer = generate(question, pile)      # naive's generate(), on the whole pile
```

The reader never answers. It only judges the pile and writes the next search. The prompt tells it to use exact codes and names from the notes, because BM25 matches `LC-0870` exactly.

**Getting the reader to stop at the right time was the hard part.** Three versions, tested on the 5 chains plus 2 golden controls:

| Reader version | What happened |
|---|---|
| 1. `{"done": true/false}` | Never once said done. Every question used all 3 rounds, and it repeated its own search in 5 of 7, despite the prompt saying not to |
| 2. Write `found` and `missing` first; no-repeat rule moved into code | Stopped after round 1 on easy questions. But on c01 it **made up** "critical below 10 or above 40 mmol/L" from an unrelated lab entry and stopped |
| 3. + "when the question names something indirectly, a note must say which one it is first" | Stops after 1 round on easy questions, hops on chains |

A rule in Python always holds; a rule in a prompt is only a request.

**The paths** (`pipeline.py`). `no_rag` is one plain LLM call with no contexts. `single` is sprint 03's `contextual_pipeline`, unchanged. `multi` is the loop. The router's tokens are added to every answer's cost.

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week1/05_adaptive_rag/compare_chain.py   # single vs multi on the chains, every round printed (<$0.01)
uv run week1/05_adaptive_rag/router.py          # route all 40 questions (~$0.002)
uv run week1/05_adaptive_rag/run_eval.py        # judged eval on golden v1 + the 10 extra end to end (~$0.09)
```

## Results (medical corpus)

### Golden set: did routing lose anything?

| Metric | Contextual hybrid | RAG Fusion | **Adaptive** |
|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | **0.96 / 0.83** | 0.92 / 0.75 |
| Context recall / precision | 0.92 / 0.76 | **0.96 / 0.82** | 0.91 / 0.76 |
| Answer relevancy | 0.85 | 0.85 | 0.85 |
| Faithfulness on real answers | 0.92 | 0.92 | 0.92 |
| Refused although answerable | 1 / 26 | 1 / 26 | 1 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 |
| Latency p50 | **1.2 s** | 3.1 s | 3.0 s |
| Cost per query | **$0.00031** | $0.00040 | $0.00049 |

Routes taken: 21 `single`, 9 `multi`, 0 `no_rag`.

**In one line:** Contextual's quality, at 2.5× the latency and 1.6× the cost. Every golden question needs exactly one search, so the router (+0.9 s) and 9 unneeded trips to `multi` bought nothing. This is the worst case for Adaptive RAG.

### The router (40 questions)

| Group | no_rag | single | multi |
|---|---|---|---|
| 30 golden (all need the documents) | **0** | 21 | 9 |
| 5 general questions | **5** | 0 | 0 |
| 5 chain questions | 0 | 1 | **4** |

- **No dangerous mistakes.** No hospital question went to `no_rag`, including "visiting hours for the HVMC ICU", where the LLM would happily invent an answer.
- **"Two questions in one" looks like a chain to the router.** All 6 multi_part golden questions went to `multi`, though both facts sit in one chunk. That's harmless, because the reader stops after round 1, but it costs a reader call.
- **c04 went to `single`.** Also harmless: one search finds both hops for it.
- The router costs $0.00005 and ~0.9 s per question.

### The 10 extra questions, end to end

| | Single-step for everything | **Adaptive** |
|---|---|---|
| 5 general questions | 5 searches for nothing | 5/5 answered without searching |
| 5 chain questions | 2 right, 2 "don't know", **1 confidently wrong** | **3 right**, 1 wrong detail, 1 "don't know" |

| Chain | Single-step | Adaptive |
|---|---|---|
| c01 Corvetrol's test tube | "don't know" | Right tube, wrong critical value (see below) |
| c02 Thyrolex's test | "don't know" | "don't know" |
| c03 DKA ketone tube | ✅ | ✅ (1 round) |
| c04 Vasopressor + Vancorin | ✅ | ✅ (routed to single) |
| c05 Vasopressor leak → who to call | ❌ ext **3610** (the ICU) | ✅ ext **4200** (central pharmacy), 3 rounds |

c03 and c04 turned out not to be real chains: one search already found both hops. A question that sounds like a chain isn't always one.

## What it gets wrong

- **It's a toll booth when traffic isn't mixed.** On the golden set the router waves everyone through to search, and adds ~0.9 s doing it.
- **The router judges by wording.** "And" in a question reads as "chain". A wrong `multi` costs one reader call; a wrong `single` is final, because `single` has no reader and no second chance.
- **The answer step can undo the loop's work (c01).** The reader's notes were right (aPTT, light blue tube, above 100 s). The final `generate()` then re-read 8 chunks and took "INR above 5.0" from a neighbouring entry. More chunks means more room to mix up entries.
- **A wordy next search misses (c02).** Round 2 searched "Thyrolex blood test turnaround time and required tube", never found the Thyrolex entry, and the loop stopped.
- **Slow.** Each round is another LLM call: 2–6 s for multi, and up to 20 s when OpenAI is slow.
- **`no_rag` doesn't know where it is.** "What can you help with?" got a generic ChatGPT answer, not "questions about this hospital's documents".

## What it gets right

- **Skipping search is safe here.** 0 hospital questions sent to `no_rag`, 5/5 general questions answered without search.
- **Follows clues that one search can't.** On c05, single-step confidently gave the wrong extension. The loop went vasopressor → Norvessa → "call pharmacy" → extension 4200.
- **Stops early on easy questions.** m01, m22 and c03 ended after round 1.
- **Nothing got worse on the golden set.** Same quality metrics as Contextual, with the same 1 refusal.

## Ideas not built

- **Skip the single/multi guess.** Always run the loop and let the evidence decide. It costs one reader call per question, and removes the router's "wrong `single` is final" risk.
- **Answer from the reader's notes.** Pass the reader's `found` text to `generate()`, so it doesn't re-pick facts from 8+ chunks (the c01 failure).
- **Keyword-style next searches.** Ask for "LC-0870 TSH" style queries, not sentences (the c02 failure).
- **Give `no_rag` one line of identity:** "You are this hospital's document assistant."

## When I'd reach for it

When traffic is mixed: many questions that need no search (greetings, definitions, "how do I…") alongside some that need a chain of lookups across documents, such as help desks or internal assistants over linked records. Skip it when nearly every question needs exactly one search, like our golden set. Then the router is pure overhead, and a good single-step pipeline wins.

## Files

| File | Purpose |
|---|---|
| `questions_extra.jsonl` | 5 no-search + 5 chain questions, each hop with its document and a verbatim evidence quote |
| `multistep.py` | The reader prompt, `is_done`, `multi_step_search` (the loop), `multi_step_pipeline` |
| `compare_chain.py` | Single vs multi on the chains + 2 golden controls, every round and hop printed |
| `router.py` | The router prompt, `route()`, and a script that routes all 40 questions and flags dangerous choices |
| `pipeline.py` | `direct_answer` (no_rag), `PATHS`, `adaptive_pipeline` (the harness entry point) |
| `run_eval.py` | Judged eval on golden v1 (`arch="adaptive"`), then the 10 extra questions end to end |
