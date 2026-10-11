# 00 Naive baseline

The simplest possible RAG pipeline. It exists so every later architecture has a number to beat.

```
question → bge-small vector → top-5 cosine search in Qdrant → gpt-4o-mini answers from those 5 chunks
```

## How it works

| Step | Choice | Why |
|---|---|---|
| Chunking | Fixed 450-token windows, 50-token overlap, cut wherever the window ends | Deliberately naive: ignores sections, paragraphs and tables |
| Token counting | bge-small's own tokenizer | bge-small reads at most 512 tokens and silently drops the rest. 450 + 2 special tokens always fits; 500 *OpenAI* tokens overflowed on 30% of chunks |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5`, 384 dimensions, local and free | |
| Store | Qdrant collection `naive_<corpus>`, cosine distance | One collection per architecture and corpus, so nothing overwrites anything else |
| Retrieval | Top 5 by cosine similarity | No keywords, no rewriting, no reranking |
| Generation | gpt-4o-mini, temperature 0. Prompt: answer only from the context, otherwise reply exactly *"I don't know based on the provided documents."* Each chunk is labelled with its source document | |

Later sprints reuse `load_chunks()` from `ingest.py` and `generate()` from `pipeline.py`, so generation stays identical and only retrieval changes.

## Run

Start Qdrant first: `docker start qdrant`. Then, from `rag_sprints/`:

```bash
RAG_CORPUS=medical uv run week1/00_naive_baseline/ingest.py     # build naive_medical
RAG_CORPUS=medical uv run week1/00_naive_baseline/run_eval.py   # score on golden_medical_v1.jsonl
```

Leave out `RAG_CORPUS` to use the 10-K corpus and `golden_v1.jsonl`. Add `--set-baseline` to `run_eval.py` to promote the run to that golden set's baseline.

## Results

**Medical corpus** (`golden_medical_v1.jsonl`, 147 chunks). This run is the baseline, `baseline_medical_v1.json`.

| Metric | Value |
|---|---|
| Evidence recall / MRR | 0.75 / 0.54 |
| Evidence recall by type | simple 0.60, exact_term 0.80, multi_part 0.92 |
| Doc hit / doc MRR | 0.96 / 0.88 |
| Faithfulness / answer relevancy | 0.77 / 0.69 |
| Context precision / recall | 0.59 / 0.75 |
| Abstention (no-answer questions) | 1.00 |
| Refused although answerable | 6 / 26 |
| Latency p50 / cost per query | 1.0 s / $0.00035 |

**10-K corpus** (`golden_v1.jsonl`, 476 chunks): evidence recall 0.70, MRR 0.56, context precision 0.70, context recall 0.83, faithfulness 0.84, abstention 1.00, refused although answerable 5/27.

## What it gets wrong

- **Exact codes are invisible to embeddings.** For "renal dose for HV-RX-2074?" the right chunk ranked #74 of 147.
- **Embedding dilution.** A one-sentence fact inside a 450-token chunk about other topics gets drowned out. On the 10-Ks, Apple's headcount sentence ranked #31.
- **Plausible neighbours win.** For "painkiller for poor kidney function", a look-alike opioid entry outranked the right one.
- **Near misses cause wrong answers, not just refusals.** For the septic shock question, the correct starting dose ranked #8. The model mixed it up with another threshold and gave a dose 5× too high.
- **Faithfulness includes refusal noise.** RAGAS scores "I don't know" inconsistently (0.0 to 1.0), so the reported faithfulness sits below the true value on real answers (0.94 on the medical set).

Hybrid RAG (`01_hybrid_rag`) targets the first three.
