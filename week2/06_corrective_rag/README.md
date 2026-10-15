# 06 Corrective RAG

Check what search found before answering. If it isn't good enough, search again with better words.

```
question → Contextual hybrid search (sprint 03) → 5 chunks
         → grader (gpt-4o-mini, one call): each chunk is relevant / partial / irrelevant
         → correct (≥1 relevant)  → answer from the relevant + partial chunks
         → otherwise              → refine the query → search again → grade the new chunks   (max 3 rounds)
         → still no relevant chunk after round 3 → record a "web search needed" event
                                                 → answer from any partial chunks, or "I don't know"
```

Think of a fact-checker sorting a reporter's clippings. Good ones go to the writer, junk goes in the bin, and when nothing useful is left the fact-checker sends the reporter back out with better search words.

This follows the Corrective RAG paper (Yan et al., 2024): grade the retrieved documents, keep the useful ones, and fall back when they're bad. The paper falls back to a web search. Here the fallback is a retry on our own documents. When that fails too, we only **record** that a web search would be needed, and never make one. On a private hospital corpus, the web would answer "ICU visiting hours" with some other hospital's hours. Medical corpus only.

## How it works

**Grader (`grader.py`).** One gpt-4o-mini call per search, temperature 0, JSON: `{"grades": ["relevant", "partial", ...]}`. There's one label per chunk and no reasons, because output tokens are the slow part of a call.

- **relevant:** states a fact that answers the question, or part of it, about the exact thing asked
- **partial:** about the right drug, test or device, but not the fact asked for
- **irrelevant:** about something else. A chunk about a *different* drug is irrelevant, even if it answers the same kind of question (HV-RX-2047 vs HV-RX-2074)

A missing or garbled label becomes `partial`, so a broken reply never throws a chunk away.

**The grader needs the notes.** On the first try the grader saw only the chunk text, and it called **6 evidence chunks "irrelevant"**. The cause was sprint 03's old problem: chunks start mid-entry (`x (vasopressin analog)…`, `be opened or crushed. Renal adjustment…`), so the grader couldn't tell which drug a chunk was about. Showing it sprint 03's note above each chunk fixed that. The answer step still reads only the original chunk.

| Grader check (30 golden questions, top 5) | Chunk only | **Note + chunk** |
|---|---|---|
| Evidence chunks called "irrelevant" (answer thrown away) | 6 | **0** |
| Evidence in the top 5 and verdict "correct" | 21 / 26 | **24 / 26** |
| No-answer questions graded "correct" (grader fooled) | 0 | 0 |

The golden set's `evidence` quotes work as an answer key for the grader: they show which chunk really holds the answer.

**Retry loop (`retry.py`).**

```python
query = question                          # round 1: the user's own words
for round in range(3):                    # the brake
    new = hybrid_search(query) minus chunks already graded
    grade(new) → add to relevant / partial lists; drop irrelevant
    if any relevant: stop
    query = refine(question, earlier searches, notes of the partial chunks)
if no relevant chunk: record_web_search_event()
```

- **The refine prompt** asks for short keywords, not sentences (sprint 05's wordy searches missed). It keeps codes exactly, uses clinical words, and gets the notes of the "partial" chunks as hints. Those hints are how "an ordinary arm vein" became "peripheral IV".
- **Only new chunks are graded.** A search that brings back only old chunks costs no grader call.
- **The web-search event** is one line appended to `web_search_events.jsonl` (time, question, every query tried, last verdict). It's also a `tool` step in LangSmith. No network call is made. The file is a run log and is git-ignored.

**Answer (`pipeline.py`).** naive's `generate()` gets the kept chunks, relevant first, at most 5. If nothing was kept, it returns "I don't know" without an answer call.
The answer step also runs after a web-search event if any "partial" chunks were kept. The grader sometimes calls the evidence chunk "partial", and `generate()` already says "I don't know" when the fact isn't there.

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week2/06_corrective_rag/grader.py     # grade all 30 golden questions, check against the evidence (~$0.01)
uv run week2/06_corrective_rag/retry.py      # the loop on 10 questions, every round printed (~$0.01)
uv run week2/06_corrective_rag/run_eval.py   # judged eval vs Contextual hybrid (~$0.05 incl. judge)
```

## Results (medical corpus)

| Metric | Contextual hybrid | RAG Fusion | Adaptive | **Corrective** |
|---|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | **0.96** / 0.83 | 0.92 / 0.75 | **0.96 / 0.86** |
| Context recall / precision | 0.92 / 0.76 | **0.96** / 0.82 | 0.91 / 0.76 | 0.95 / **0.90** |
| Answer relevancy | 0.85 | 0.85 | 0.85 | **0.90** |
| Faithfulness on real answers | 0.92 | 0.92 | 0.92 | 0.91 |
| Refused although answerable | 1 / 26 | 1 / 26 | 1 / 26 | **0 / 26** |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 | 1.00 |
| Chunks the LLM reads (mean) | 5 | 5 | 5+ | **1.7** |
| Latency p50 / p95 | **1.2 s** / – | 3.1 s / 13 s | 3.0 s / – | 2.6 s / 5.5 s |
| Cost per query | **$0.00031** | $0.00040 | $0.00049 | $0.00056 |

**In one line:** the first sprint to answer all 26 answerable questions while still refusing all 4 unanswerable ones, at about 2× Contextual's latency and 1.8× its cost.

**Rounds used:** 22 questions stopped after round 1, 1 needed 2 rounds (m20), and 7 hit the 3-round limit.

### Per question

| Question | Contextual hybrid | Corrective | Why |
|---|---|---|---|
| m20 "potassium through an ordinary arm vein" | refused | **"10 mEq per hour through a peripheral line"** | Round 1 found a "partial" chunk. Its note gave the refine step the words "potassium infusion rate peripheral IV", and round 2 found the evidence |
| m27–m30 (no answer) | "I don't know" | "I don't know" + web event | 3 rounds found nothing relevant. m29 and m30 are the ones the web would have answered wrongly |
| m04, m09, m08 | ✅ | ✅ + a **false** web event | The evidence was found in round 1 but graded "partial". The retries found nothing better, but the partial chunk was kept and the answer is right |
| m11 "sugar crashed… what should the nurse give?" | vague | vague | Two low-sugar chunks were graded "relevant", so no retry. The evidence (rank 8) never came in |

## What it gets wrong

- **3 of 7 web events are false alarms (m04, m08, m09).** The grader is strict: a chunk that holds the answer can still be graded "partial". If these events triggered real web searches, 3 of the 7 would be wasted calls. m08 was "correct" in the grader check and "partial" in the eval, so even at temperature 0 the grader isn't perfectly stable.
- **It can't retry what the grader doesn't flag (m11).** The loop only runs when the grader says the chunks are bad. Chunks on the right topic with the wrong detail pass as "relevant".
- **Retries rarely help on our golden set.** Only m20 was fixed by a refined search. For the other 6 questions that reached round 3, the extra rounds cost about 2–3 s and found nothing new. In the `retry.py` demo, rounds for m04, m09, m27, m28 and m29 brought back "(no new chunks)": the refined keywords pulled up the same 5 chunks again.
- **Slower and pricier.** The grader adds ~1.2 s and ~2,000 prompt tokens to every question, retry or not. Prompt tokens rose from 1,921 to 3,437 per question.
- **Judge noise.** m12 and m26 scored low on faithfulness, but both answers are correct.

## What it gets right

- **Zero false refusals, and abstention still 1.00.** Every earlier sprint refused m20.
- **The LLM reads less and does better.** 1.7 chunks instead of 5, and context precision rose from 0.76 to 0.90. Answer relevancy rose from 0.85 to 0.90.
- **Dead ends are visible.** Every "the documents ran out" case is logged with the queries tried. That's a ready-made to-do list for whoever maintains the documents.
- **No dangerous fallback.** No-answer questions end in "I don't know", not in another hospital's visiting hours from the web.

## Ideas not built

- **Trust "partial" more on exact codes.** If the chunk's note names the exact code in the question (LC-0710, CP-09), "partial" is probably the answer. That would cut the false web events.
- **Retry on wide, not deep.** Instead of the same top 5 again, search with a refined query and take ranks 6–10. This would have reached m11's evidence at rank 8.
- **Real web fallback for public questions only.** Send only questions that don't depend on this hospital (sprint 05's `no_rag` kind) to the web, and never m29/m30-style ones.

## When I'd reach for it

When a wrong answer costs more than a slow one, and the documents have gaps: clinical references, policy assistants, internal help desks. The grader keeps junk out of the answer, and the event log shows where the documents fall short. Skip it when one search nearly always finds the answer and latency matters. Then the grader is a ~1 s toll on every question.

## Files

| File | Purpose |
|---|---|
| `grader.py` | The grader prompt, `grade()`, `verdict()`, and a check of the grader against the golden evidence quotes |
| `retry.py` | The refine prompt, `corrective_search` (the loop), `record_web_search_event`, and a 10-question demo |
| `pipeline.py` | `corrective_pipeline`, the harness entry point: loop → kept chunks → naive's generate() |
| `run_eval.py` | Judged eval on golden v1 (`arch="corrective"`), compared with Contextual hybrid |
