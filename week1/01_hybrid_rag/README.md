# 01 Hybrid RAG

Two retrievers that fail in different ways, merged into one ranking.

```
question ─┬→ dense search (meaning)  → top 10 ─┐
          └→ BM25 search  (keywords) → top 10 ─┴→ Reciprocal Rank Fusion → top 5 → gpt-4o-mini
```

- **Dense** (bge-small vectors) finds paraphrases but blurs exact strings such as `HV-RX-2074` or `CP-09`.
- **BM25** (sparse keyword vectors) finds rare exact terms but has no idea that "kidney" means "renal".

It uses the same chunks and the same `generate()` step as the naive baseline, so retrieval is the only thing that changes.

## How it works

**Indexing (`ingest.py`).** Every chunk is stored in one Qdrant collection, `hybrid_<corpus>`, with two named vectors:

| Vector | What it holds |
|---|---|
| `dense` | 384-number bge-small embedding, cosine distance |
| `bm25` | Sparse "word id → weight" vector from FastEmbed `Qdrant/bm25`. Words are lowercased and stemmed, stopwords dropped, and punctuation split: `HV-RX-2074` → `hv rx 2074` |

Two details matter:

- **`Modifier.IDF`.** FastEmbed's BM25 vectors contain only the term-frequency half of BM25. Qdrant supplies the inverse document frequency (how rare each word is) at query time. Without it, `rx` (in 73 of 147 chunks) would count as much as `2074` (in 1 chunk).
- **`avg_len`.** BM25 normalises by chunk length against the average chunk length. The ingest script measures the real average (202 words for the medical set) instead of using FastEmbed's default of 256.

**Fusion (`pipeline.py`).** Cosine scores (0 to 1) and BM25 scores (0 to about 30) can't be added together. Reciprocal Rank Fusion ignores the scores and uses only positions: each list gives a chunk `1 / (k + rank)` points, the points are summed, and the top 5 go to the LLM. A chunk found by both lists usually beats one found by a single list. When the lists share nothing, the result alternates between them.

| Setting | Value | Meaning |
|---|---|---|
| `TOP_K` | 5 | Chunks the LLM sees (same as naive) |
| `CANDIDATES` | 10 | Chunks each path contributes to fusion |
| `RRF_K` | 10 | Damping constant. The RRF paper uses 60 |
| weights | equal | Neither path counts for more |

## Run

Start Qdrant first: `docker start qdrant`. Then, from `rag_sprints/`:

```bash
RAG_CORPUS=medical uv run week1/01_hybrid_rag/ingest.py         # build hybrid_medical
RAG_CORPUS=medical uv run week1/01_hybrid_rag/compare_paths.py  # dense vs BM25 vs hybrid, retrieval only, free
RAG_CORPUS=medical uv run week1/01_hybrid_rag/explore.py        # RRF arithmetic on chosen questions + Qdrant cross-check
RAG_CORPUS=medical uv run week1/01_hybrid_rag/tune_fusion.py    # grid over k, depth, weights, free
RAG_CORPUS=medical uv run week1/01_hybrid_rag/run_eval.py       # full eval with LLM judge, deltas vs naive
```

## Results (medical corpus, vs the naive baseline)

| Metric | Naive | Hybrid |
|---|---|---|
| Evidence recall / MRR | 0.75 / 0.54 | **0.88 / 0.69** |
| Evidence recall: simple / exact_term / multi_part | 0.60 / 0.80 / 0.92 | **0.80 / 0.90 / 1.00** |
| Context recall / precision | 0.75 / 0.59 | **0.90 / 0.68** |
| Answer relevancy | 0.69 | **0.78** |
| Faithfulness, reported / on real answers only | 0.77 / 0.94 | 0.83 / 0.94 |
| Refused although answerable | 6 / 26 | **3 / 26** |
| Abstention (no-answer questions) | 1.00 | 1.00 |
| Latency p50 / cost per query | 1.0 s / $0.00035 | 1.0 s / $0.00036 |

The faithfulness gain is not real: on real answers both score 0.94. The reported number rose only because Hybrid refused less, and RAGAS scores refusals inconsistently.

### Each path alone fails

Rank of the evidence chunk. ✅ means it reached the LLM's top 5.

| Question | Dense | BM25 | Hybrid |
|---|---|---|---|
| m01 "renal dose for **HV-RX-2074**?" | not in top 20 | 1 ✅ | 3 ✅ |
| m02 "fault **E-71** on the Kestrel IV-220?" (E-71 exists in 3 pump manuals) | 2 ✅ | 6 | 4 ✅ |
| m03 "max daily dose of **Lumidex**?" (other entries cross-reference Lumidex) | 3 ✅ | 9 | 5 ✅ |
| m11 "sugar level has crashed… what should the nurse give?" | 9 | 6 | 5 ✅ |

On m11 neither path reached the top 5 alone; fusion rewarded the chunk for appearing in both lists. Hybrid also fixed a wrong-dose answer: naive said the septic shock vasopressor starts at 0.25 mcg/kg/min, Hybrid says 0.05, which is correct.

### Still failing

- **m09** "protocol CP-09": fused rank 9.
- **m13** "kidneys not working": the documents say "renal", never "kidney".
- **m20** "potassium through an arm vein": the documents say "peripheral line".

Neither path finds m13 or m20 well, so fusion can't help. These are vocabulary gaps, a job for query rewriting or contextual chunks.

## Fusion tuning

The paper defaults (k=60, 20 candidates) **buried BM25's confident #1 for m01 at rank #7**. Five generic chunks that were mediocre in both lists outscored it: 1/80 + 1/80 = 0.025 beats 1/61 = 0.016. A smaller k makes rank #1 worth more than lower ranks (1.8× a #10 at k=10, against 1.15× at k=60), and fewer candidates gives lukewarm chunks fewer chances to appear in both lists.

Evidence recall / MRR, retrieval only:

| Retriever | Medical (tuned on) | 10-K (held out) |
|---|---|---|
| Dense alone | 0.75 / 0.54 | 0.70 / 0.56 |
| BM25 alone | 0.81 / 0.61 | **0.88 / 0.71** |
| Hybrid, k=60, 20 candidates | 0.81 / 0.67 | 0.79 / 0.69 |
| **Hybrid, k=10, 10 candidates** | **0.88 / 0.69** | 0.81 / 0.69 |

- The tuned setting holds up on the held-out set: +0.02 over the defaults.
- On the 10-Ks, BM25 alone beats every equal-weight hybrid. Dense is weak there, and fusing in a weak path drags the strong one down. Hybrid pays off when both paths are decent and fail on *different* questions.
- Weights were left equal. BM25 ×2 was best on the 10-Ks and worst on the medical set.
- With 26 questions, one question moves recall by about 0.04, so treat small differences as noise.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Builds the two-vector collection from naive's chunks |
| `pipeline.py` | `dense_search`, `bm25_search`, `rrf_fuse`, `hybrid_search`, `hybrid_pipeline` |
| `run_eval.py` | Full evaluation with the LLM judge |
| `compare_paths.py` | Retrieval-only scores and per-question ranks for each path |
| `explore.py` | Prints the fusion arithmetic and cross-checks against Qdrant's built-in RRF (25/26 identical; the one difference is a near-tie) |
| `tune_fusion.py` | Grid over `RRF_K`, `CANDIDATES` and weights |
