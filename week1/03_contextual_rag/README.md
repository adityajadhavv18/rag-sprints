# 03 Contextual RAG

Before indexing, give every chunk a short note saying where it comes from.

```
indexing:  document + chunk → gpt-4o-mini writes a 1–2 sentence note → embed (note + chunk) → Qdrant
query:     question → dense + BM25 over (note + chunk) → RRF → top 5 ORIGINAL chunks → gpt-4o-mini
```

Think of index cards with the book title and chapter written at the top. A chunk that starts mid-entry
(`: Platelet count every 2 to 3 days…`) now says which drugs it covers.

This is Anthropic's "contextual retrieval", applied to naive's chunks. Medical corpus only.

## How it works

**Chunks (`contextualize.py`).** Naive's chunker, unchanged, with a 370-token window instead of 450: 180 chunks.
bge-small reads at most 512 tokens, and 136 of naive's 147 chunks were already 430–450 tokens, so a note would have pushed the end of the chunk out.
370 leaves 140 tokens for the note. The real notes were 69–105 tokens, and no chunk + note went over 512.

**Notes (`contextualize.py`).** gpt-4o-mini, temperature 0. It reads the whole document plus the chunk, and is asked to:

- name the document, and every entry the chunk covers by its code and name, including an entry it starts or ends in the middle of
- say what kind of information the chunk holds, adding everyday words where they help

The document goes first in the prompt, so every chunk of the same document starts with the same text. OpenAI caches that repeated opening at half price, and 92% of the input tokens were cached.
The notes are saved in `notes_medical.json` (committed), so re-indexing is free.

| | |
|---|---|
| One-time note cost | **$0.26** (180 calls, 3.06M input tokens, 2.77M of them cached) |
| Note length | median 41 words, max 59 |

**Index (`ingest.py`).** Same two-vector layout as Hybrid: `dense` and `bm25`, both built from **note + chunk**. The payload `text` is the **original chunk**; the note is stored separately for inspection.
A second collection, `chunks370_medical`, holds the same chunks without notes. It is the control: it shows how much of the change comes from the smaller chunks alone.

**Query (`pipeline.py`).** Hybrid's search and RRF (k=10, 10 candidates each), naive's `generate()`, 5 chunks.
The LLM never sees the notes, and `contexts` = the original chunks, so evidence recall checks exactly what the LLM read.

## Run

Start Qdrant first: `docker start qdrant`. Then, from `rag_sprints/`:

```bash
uv run week1/03_contextual_rag/contextualize.py --sample   # notes for 4 chunks + size check (~$0.01)
uv run week1/03_contextual_rag/contextualize.py            # notes for every chunk (cached, ~$0.25 the first time)
uv run week1/03_contextual_rag/ingest.py                   # build contextual_medical and chunks370_medical
uv run week1/03_contextual_rag/compare_retrieval.py        # 6 setups, retrieval only, free
uv run week1/03_contextual_rag/run_eval.py                 # full eval with judge: contextual + contextual_dense
```

## Results (medical corpus)

| Metric | Naive | Hybrid | Parent-Child | Contextual dense | **Contextual hybrid** |
|---|---|---|---|---|---|
| Evidence recall / MRR | 0.75 / 0.54 | 0.88 / 0.69 | 0.77 / 0.59 | 0.88 / 0.67 | **0.92 / 0.75** |
| Evidence recall: simple / exact_term / multi_part | 0.60 / 0.80 / 0.92 | 0.80 / 0.90 / 1.00 | 0.70 / 0.70 / 1.00 | 0.80 / 0.90 / 1.00 | 0.80 / **1.00** / 1.00 |
| Context recall / precision | 0.75 / 0.59 | 0.90 / 0.68 | 0.74 / 0.58 | 0.88 / 0.65 | **0.92 / 0.76** |
| Answer relevancy | 0.69 | 0.78 | 0.69 | 0.83 | **0.85** |
| Faithfulness, reported / on real answers | 0.77 / 0.94 | 0.83 / 0.94 | 0.76 / 0.92 | 0.84 / 0.89 | 0.88 / 0.92 |
| Refused although answerable | 6 / 26 | 3 / 26 | 6 / 26 | 2 / 26 | **1 / 26** |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Prompt tokens per question | 2,248 | 2,277 | **1,349** | 1,922 | 1,921 |
| Latency p50 / cost per query | 1.0 s / $0.00035 | 1.0 s / $0.00036 | 1.1 s / **$0.00022** | 1.0 s / $0.00031 | 1.2 s / $0.00031 |

**In one line:** the best setup so far on every retrieval and answer-quality metric, for a one-time $0.26.
Only m20 is still refused.

### The notes do the work, not the smaller chunks

Retrieval only (`compare_retrieval.py`), evidence recall / MRR:

| Chunks | Dense | Hybrid |
|---|---|---|
| 450 tokens, no notes (naive / hybrid) | 0.75 / 0.54 | 0.88 / 0.69 |
| 370 tokens, no notes (control) | 0.71 / 0.57 | 0.83 / 0.66 |
| **370 tokens + notes** | **0.88 / 0.67** | **0.92 / 0.75** |

Smaller chunks alone made retrieval slightly worse. Notes on top of the same chunks added +0.17 (dense) and +0.09 (hybrid).
With notes, plain dense search matches the old Hybrid.

### Per question

Rank of the evidence chunk. ✅ means it reached the LLM's top 5.

| Question | Hybrid | Contextual hybrid | Why |
|---|---|---|---|
| m09 "first actions of protocol CP-09?" | 9 | **1 ✅** | The chunk starts mid-protocol; the note names "CP-09 Suspected transfusion reaction" |
| m13 "strong painkiller… kidneys not working" | not found | **4 ✅** | The note says "pain management medications". It matched "painkiller", not "kidney" |
| m01 "renal dose for HV-RX-2074?" | 3 ✅ | 2 ✅ | Only through BM25. Dense ranks it **#91 of 180** even with the code in the note |
| m11 "sugar crashed… what should the nurse give?" | 5 ✅ | **8** | New miss. Dense finds it at #1, BM25 doesn't, and fusion pulls it down |
| m20 "potassium through an arm vein" | not found | not found | No note says "potassium" or "peripheral" |

## What it gets wrong

- **Notes don't teach dense search to read codes.** m01's chunk names `HV-RX-2074` in its note and still ranks #91 by meaning. Contextual dense refused m01, Contextual hybrid answered it. Codes still need BM25.
- **Notes fix "where am I", not vocabulary gaps.** The prompt asked for everyday words, but the notes mostly listed entries and section names. m20 ("arm vein" vs "peripheral line") still fails everywhere.
- **Two new wrong answers.** Real-answer faithfulness fell slightly (0.94 → 0.92 hybrid, 0.89 dense):
  - m11 (hybrid): evidence at rank 8, so the model gave a vague answer ("give medicine through the drip") instead of "Dextriva 25 g IV push".
  - m12 (dense): it said the pump runs "at least 2 hours" on battery; the right answer is 8 hours. The evidence was at rank 10, and the model took another pump's figure.

  m26 scored 0.33 but the answer is correct, so that one is judge noise.
- **Notes are wordy.** 81 of 180 start with "This chunk…" or "The chunk…", and the long document title takes about 20 tokens of every note.

## What it gets right

- **Chunks that start mid-entry become findable.** m09 went from rank 9 to 1.
- **Every exact-term question answered** (evidence recall 1.00), and refusals fell from 3 (Hybrid) to 1.
- **Cheap.** $0.26 once, then free to re-index. Per-query cost is unchanged, because the LLM reads only the original chunks; it is slightly lower than naive's because 370-token chunks are smaller.

## When I'd reach for it

When chunks lose their anchor: they start mid-entry, or say "the company" or "this protocol" without naming it. It pairs best with BM25, which handles the codes that dense search can't read. It won't fix vocabulary gaps; that's a job for query rewriting.

## Files

| File | Purpose |
|---|---|
| `contextualize.py` | 370-token chunks, the note prompt, cached note writing, size check |
| `notes_medical.json` | The 180 notes (cached, committed) |
| `ingest.py` | Builds `contextual_medical` (note + chunk) and the control `chunks370_medical` (chunk only) |
| `pipeline.py` | `dense_search`, `bm25_search`, `hybrid_search`, `contextual_pipeline` |
| `compare_retrieval.py` | 6 setups, retrieval only, with the evidence rank per question |
| `run_eval.py` | Full evaluation with the LLM judge, hybrid and dense |
