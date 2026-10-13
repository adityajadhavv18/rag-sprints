# 04 RAG Fusion

Ask the same question five ways, search with all five, and trust the chunks they agree on.

```
question → gpt-4o-mini writes 4 rephrasings → 5 queries
         → each query: dense top 10 + BM25 top 10 on the contextual index → 10 ranked lists
         → RRF over all 10 lists (k=10) → top 5 original chunks → gpt-4o-mini answers the ORIGINAL question
```

Think of five librarians hearing the same request in different words. The book most of them pull off the shelf is probably the right one.

The textbook version is rephrasings + RRF on one dense search. Ours stacks it on the best setup so far: Hybrid's dense + BM25 (sprint 01) on Contextual's notes index (sprint 03). No new index is built. Medical corpus only.

## How it works

**Rephrasings (`rewrite.py`).** One gpt-4o-mini call, temperature 0, JSON output. The model sees only the question, never the documents. The prompt has three rules:

- keep every code and name exactly as written (`HV-RX-2074`, `E-71`), because BM25 needs them
- at least one version in clinical words, to bridge the gap between lay and document vocabulary
- no added facts, numbers or answers

The prompt's examples ("hypertension", "pulmonary embolism") are deliberately not from the golden set. A first draft used "renal impairment" and "peripheral line", which are the answers to m13 and m20. That would have rigged the test, so it was changed before any results were taken.

Rephrasings are cached in `rewrites_medical.json` (30 questions, $0.002), so retrieval experiments are free.

**Search and fusion (`pipeline.py`).** Each of the 5 queries runs Contextual's `dense_search` and `bm25_search`, giving 10 lists named `q0_dense` … `q4_bm25`. Hybrid's `rrf_fuse` merges them unchanged: a chunk earns `1 / (10 + rank)` in every list it is in, the points are added up, and the top 5 go to the LLM.

The 10 lists have 100 slots but few distinct chunks, because the five queries mostly agree. The median is **22 distinct chunks** (min 11, max 50) out of 180.

**Answer.** Naive's `generate()`, unchanged. The LLM sees the original question and the 5 original chunks. It never sees the rephrasings or the notes.

**Live vs cached.** In the eval, `fusion_pipeline` makes the rewrite call live, as it would in production, so latency includes it. The rewrite's tokens are added to `tokens_in/out`; both calls are gpt-4o-mini, so the harness prices the sum correctly.

| Setting | Value | Meaning |
|---|---|---|
| `N_REPHRASINGS` | 4 | + the original = 5 queries, 10 lists |
| `CANDIDATES` | 10 | Chunks per list into fusion (Hybrid's tuned value) |
| `RRF_K` | 10 | Same damping as Hybrid |
| `original_weight` | 1 | The original question counts the same as each rephrasing |
| `TOP_K` | 5 | Chunks the LLM sees |

## Run

Start Qdrant first: `docker start qdrant`. It needs the `contextual_medical` and `hybrid_medical` collections from sprints 01 and 03. Then, from `rag_sprints/`:

```bash
uv run week1/04_rag_fusion/rewrite.py --sample      # rephrasings for 5 questions, printed (<$0.001)
uv run week1/04_rag_fusion/rewrite.py               # all 30, cached (~$0.002 the first time)
uv run week1/04_rag_fusion/compare_retrieval.py     # 6 setups, retrieval only, free
uv run week1/04_rag_fusion/run_eval.py              # full eval with the judge (~$0.08 incl. judge)
```

## Results (medical corpus)

| Metric | Naive | Hybrid | Contextual hybrid | **RAG Fusion** |
|---|---|---|---|---|
| Evidence recall / MRR | 0.75 / 0.54 | 0.88 / 0.69 | 0.92 / 0.75 | **0.96 / 0.83** |
| Evidence recall: simple / exact_term / multi_part | 0.60 / 0.80 / 0.92 | 0.80 / 0.90 / 1.00 | 0.80 / 1.00 / 1.00 | **0.90 / 1.00 / 1.00** |
| Context recall / precision | 0.75 / 0.59 | 0.90 / 0.68 | 0.92 / 0.76 | **0.96 / 0.82** |
| Answer relevancy | 0.69 | 0.78 | 0.85 | 0.85 |
| Faithfulness, reported / on real answers | 0.77 / 0.94 | 0.83 / 0.94 | 0.88 / 0.92 | 0.92 / 0.92 |
| Refused although answerable | 6 / 26 | 3 / 26 | 1 / 26 | 1 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 | 1.00 |
| Prompt tokens per question (incl. rewrite) | 2,248 | 2,277 | 1,921 | 2,135 |
| Latency p50 / cost per query | 1.0 s / $0.00035 | 1.0 s / $0.00036 | 1.2 s / $0.00031 | **3.1 s** / $0.00040 |

**In one line:** the best retrieval so far, at about 2.5× the latency.

This run used the harness's newer refusal rule: an "I don't know" answer gets no faithfulness score and is counted under refusals instead. That is why Fusion's reported and real-answer faithfulness are the same number. Compare the "on real answers" column across sprints.

The "simple" questions are the deliberately vague ones, written in lay words. They are where fusion was meant to help, and they rose from 0.80 to 0.90. With 10 of them, that is one question.

### Fusion needs the notes

Retrieval only (`compare_retrieval.py`, cached rephrasings), evidence recall / MRR:

| Setup | Score | Vague (simple) |
|---|---|---|
| Hybrid, 1 query | 0.88 / 0.69 | 0.80 |
| Contextual hybrid, 1 query | 0.92 / 0.75 | 0.80 |
| Fusion on Hybrid's index | 0.88 / 0.66 | 0.80 |
| **Fusion on Contextual's index** | **0.96 / 0.80** | **0.90** |
| …with 2 rephrasings instead of 4 | 0.96 / 0.80 | 0.90 |
| …with the original question ×2 | 0.92 / 0.79 | 0.80 |

- **On the plain index, fusion broke even.** It fixed m09 and m11 but lost m03 and m13. On the notes index it lost nothing.
- **Counting the original question double hurt.** On m11 the rephrasings ("hypoglycemia") found the chunk better than the user's words ("sugar crashed"), so extra weight on the original pulled it from #3 to #6.
- **2 rephrasings tied 4.** On 10 vague questions that is a tie, not proof that 2 is enough. Kept at 4.

The judged run scored MRR 0.83, against 0.80 here. The eval writes rephrasings live, and even at temperature 0 they can differ slightly from the cached ones.

### Per question

Rank of the evidence chunk. ✅ means it reached the LLM's top 5.

| Question | Contextual hybrid | RAG Fusion | Why |
|---|---|---|---|
| m13 "strong painkiller… kidneys not working well" | 4 ✅ | **1 ✅** | Rephrasings say "renal impairment" and "analgesic" |
| m11 "sugar crashed… what should the nurse give?" | 8 | **3 ✅** | Two rephrasings say "hypoglycemia". The answer is now right: "Dextriva 25 g IV push" |
| m20 "potassium through an ordinary arm vein" | not found | 16 | Outvoted. See below |

## What it gets wrong

- **A correct minority gets outvoted (m20).** Two rephrasings said "peripheral", and BM25 ranked the evidence **#4 and #7** for them. The other 8 lists didn't have it in their top 10. So it scored 1/14 + 1/17 = 0.13 and lost to chunks that appeared in many lists. RRF rewards agreement, and here only the minority was right. m20 is still refused.
- **The rephrasings didn't use the documents' exact term.** They said "peripheral vein" and "peripheral venous access", never "peripheral line". The model can't know the corpus's wording, because it never sees the documents.
- **Slower.** Timed on 6 questions:

  | Step | Time |
  |---|---|
  | Rewrite (the extra LLM call) | 1.6–5.4 s |
  | 10 searches | 0.15 s |
  | Answer | 0.7–1.6 s |

  The rewrite is the whole cost: it writes ~90 words, and output tokens are the slow part of an LLM call. The p95 of 13 s was OpenAI being slow at that moment. m11 took 13 s in the eval and 2.7 s when re-timed.
- **Judge noise.** m08 (0.33), m26 (0.33) and m12 (0.50) scored low on faithfulness, but all three answers are correct.

## What it gets right

- **Closes vocabulary gaps on the question side.** The notes (sprint 03) added context on the chunk side, and the rephrasings add clinical words on the question side. m13 went from #4 to #1, and m11 was fixed.
- **Safe on no-answer questions.** All 4 were still refused. The rephrasings stayed on the question and invented nothing.
- **Codes survive.** Every exact-term question stayed at evidence recall 1.00. `HV-RX-2074` came through all 4 rephrasings unchanged.
- **Cheap per query.** +$0.00009 for the rewrite call, and no new index.

## Ideas not built

- **Shorter rephrasings.** Asking for keyword-style queries instead of full sentences would cut the 2–5 s rewrite.
- **Search while rewriting.** Run the original question's two searches during the rewrite call, then add the rephrasings' lists when they arrive.
- **Let a confident list win.** A chunk that is #1–#5 in any list could be guaranteed a slot, which would protect a correct minority like m20.

## When I'd reach for it

When users type short, vague or lay questions and the documents use different words: patient portals, product search, help desks. It works best on top of an index that is already good, such as hybrid + notes. Skip it when latency matters more than the last few points of recall. The rewrite call costs seconds, not cents.

## Files

| File | Purpose |
|---|---|
| `rewrite.py` | The rephrasing prompt, the live call, the cache |
| `rewrites_medical.json` | 4 rephrasings for each of the 30 golden questions (cached, committed) |
| `pipeline.py` | `fusion_search` (10 lists → RRF), `fusion_pipeline` (live rewrite → search → naive's generate) |
| `compare_retrieval.py` | 6 setups, retrieval only, evidence rank per question, distinct chunks in the 10 lists |
| `run_eval.py` | Full evaluation with the LLM judge, `arch="rag_fusion"` |
