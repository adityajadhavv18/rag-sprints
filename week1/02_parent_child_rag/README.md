# 02 Parent-Child RAG

Search small pieces, hand the LLM big pieces.

```
question → bge vector → top 30 CHILDREN (single lines) → swap each for its PARENT (whole entry)
         → drop repeats → top 10 parents → gpt-4o-mini
```

Think of a library card catalogue: each index card (child) holds one sentence and has a photocopy of its whole page (parent) on the back. You search the cards, but you read the page.

Core Parent-Child only: no title stamping, no keyword search. Retrieval is dense, as in naive, so only the chunking changes. Medical corpus only.

## How it works

**Chunking (`chunking.py`).** The medical documents are already written as one unit per blank-line block, so:

| Piece | Rule | Count | Median size |
|---|---|---|---|
| Parent | One blank-line block: a drug entry, lab test, fault code, protocol, IV table, leaflet. A heading on its own (`Section: …`, the doc title) is glued to the block below | 564 | 99 tokens (max 567) |
| Child | One line of a parent. A line over 64 tokens is cut at sentence ends | 3,567 | 14 tokens |

Naive's chunks are 450 tokens. Every golden evidence quote sits whole inside one child, and 25 of 26 fit inside one parent (m21 needs two drug entries).

**Storage (`ingest.py`).** One Qdrant point per child in `parent_child_medical`, with a `dense` and a `bm25` vector. The payload holds the child line and a copy of the parent's full text:

```json
"payload": {
  "child_id":    "hvmc_formulary::p006::c04",
  "parent_id":   "hvmc_formulary::p006",
  "doc_id":      "hvmc_formulary",
  "text":        "Renal adjustment: CrCl 30 to 50 mL/min: 75 mg twice daily. ...",
  "parent_text": "HV-RX-2074 Corvetrol (oral direct thrombin inhibitor)\nFormulary status: ..."
}
```

Parents are never embedded. They don't need to be: the search finds a child, and the parent comes back in the same result. On a large corpus you would store parents once in a separate store and keep only `parent_id` on each child.

**Retrieval (`pipeline.py`).** Search 30 children, walk them best-first, swap each for its parent, skip parents already taken, stop at 10. A parent ranks where its best child ranked. `contexts` = the parents, so evidence recall measures what the LLM really saw.

## Run

Start Qdrant first: `docker start qdrant`. Then, from `rag_sprints/`:

```bash
uv run week1/02_parent_child_rag/chunking.py         # examples, sizes, duplicates, evidence check
uv run week1/02_parent_child_rag/ingest.py           # build parent_child_medical, print one stored point
uv run week1/02_parent_child_rag/child_retrieval.py  # where the evidence lands: child vs parent, free
uv run week1/02_parent_child_rag/tune_parents.py     # how many parents, free
uv run week1/02_parent_child_rag/run_eval.py         # full eval with LLM judge, deltas vs naive
```

## Results (medical corpus)

| Metric | Naive | Hybrid | Parent-Child |
|---|---|---|---|
| Evidence recall / MRR | 0.75 / 0.54 | **0.88 / 0.69** | 0.77 / 0.59 |
| Evidence recall: simple / exact_term / multi_part | 0.60 / 0.80 / 0.92 | 0.80 / 0.90 / 1.00 | 0.70 / 0.70 / 1.00 |
| Doc hit | 0.96 | 0.96 | **1.00** |
| Context recall / precision | 0.75 / 0.59 | **0.90 / 0.68** | 0.74 / 0.58 |
| Answer relevancy | 0.69 | **0.78** | 0.69 |
| Faithfulness, reported / on real answers | 0.77 / 0.94 | 0.83 / 0.94 | 0.76 / 0.92 |
| Refused although answerable | 6 / 26 | **3 / 26** | 6 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 |
| **Prompt tokens per question** | 2,248 | 2,277 | **1,349** |
| Latency p50 / cost per query | 1.0 s / $0.00035 | 1.0 s / $0.00036 | 1.1 s / **$0.00022** |

**In one line:** about the same quality as naive, for 40% fewer tokens and 37% lower cost. It does not beat Hybrid.

### How many parents?

Retrieval only (`tune_parents.py`):

| Setup | Evidence recall | MRR | Tokens the LLM reads |
|---|---|---|---|
| Naive, 5 chunks | 0.75 | 0.54 | 2,120 |
| Parent-Child, 5 parents | 0.73 | 0.58 | 608 |
| **Parent-Child, 10 parents** | **0.77** | **0.59** | 1,167 |
| Parent-Child + hybrid search, 10 parents | 0.85 | 0.60 | 1,110 |

Five parents nearly match naive on a quarter of the text. Ten parents beat it on half the text.

### Same refusal count as naive, different questions

| Fixed (naive refused, Parent-Child answers) | Broken (naive answered, Parent-Child refuses) |
|---|---|
| m09 CP-09: title line "CP-09 Suspected transfusion reaction" matched, whole protocol came along | m03 Lumidex max dose |
| m12 battery hours: short line no longer buried | m07 drug library version |
| m13 "kidneys not working": the line "Preferred opioid in renal impairment" ranked #1 (naive: #20) | m17, m18 |
| m19 leg devices, 18 hours a day | m01 still refused (naive refused it too) |

m21 (septic shock) is still correct: Norvessa starts at 0.05 mcg/kg/min, not 0.25.

## What it gets wrong: the flaws of core Parent-Child

- **Lines that name the thing beat lines that hold the answer.** For "max daily dose of Lumidex?" the top 5 were all `Lumidex with X: Compatible…` lines. The dosing line never says "Lumidex", so it can't compete. Naive's big chunk held the name and the dose together.
- **Anonymous duplicate lines.** 51% of children appear word for word in other parents (`Renal adjustment: No adjustment.` ×106). For "renal dose for HV-RX-2074?" the top 5 were renal lines from other drugs. A line alone doesn't say which drug it belongs to.
- **Title lines are too general.** `Kestrel IV-220 Volumetric Infusion Pump` matches the pump name but holds no answer (m07).
- **Tiny children are weak for keyword search.** About 7 words each after stopwords, so BM25 has little to score. Hybrid search over children did worse than dense at 5 parents (0.65 vs 0.73).

## What it gets right

- **The parent swap rescues near misses.** The exact answer line was in the top 5 children for only 11 of 26 questions, but some line from the right entry was there for 19. 8 questions were saved only by the swap.
- **Dilution is gone.** One-sentence facts (m12, m13, m19) now match on their own.
- **Clean, smaller context.** Each parent is one whole entry: no half-entries, no neighbour drug with a different dose. Doc hit reached 1.00.

## When I'd reach for it

When documents are made of self-contained units (contract clauses, drug entries, product specs) and context size or cost matters. Not on its own when questions hinge on codes or names that sit only in an entry's title line. Pair it with keyword search, or stamp the title onto each child (Contextual RAG, sprint 03), which targets both main flaws.

## Files

| File | Purpose |
|---|---|
| `chunking.py` | Parent and child rules, plus inspection: examples, sizes, duplicates, evidence check |
| `ingest.py` | Embeds children (dense + bm25) into `parent_child_medical`, parent text in each payload |
| `pipeline.py` | `dense_child_search`, `bm25_child_search`, `hybrid_child_search`, `to_parents`, `parent_search`, `parent_child_pipeline` |
| `child_retrieval.py` | Evidence rank per question: exact child vs right parent, against naive |
| `tune_parents.py` | Sweep over 3 / 5 / 8 / 10 parents with context tokens |
| `run_eval.py` | Full evaluation with the LLM judge |
