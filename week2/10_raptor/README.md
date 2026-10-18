# 10 RAPTOR (lite)

Before any question arrives, group similar chunks, have the LLM summarise each group, then group and summarise the summaries. Put every chunk and every summary on one shelf, and let search pick whichever level fits the question.

```
INDEX (once):  180 Contextual chunks (leaves) ── KMeans(18) on their vectors ── gpt-4o-mini: one summary per cluster  (level 1)
               18 summaries ─────────────────── KMeans(4)  on their vectors ── one summary per cluster of summaries   (level 2)
               leaves + 22 summaries → one Qdrant collection, raptor_medical (dense + BM25)

QUERY:  question → hybrid search over leaves AND summaries together ("collapsed tree") → top 5, any level → naive's generate()
```

Think of a textbook. The pages are the leaves, each chapter ends with a summary (level 1), and the introduction summarises the chapters (level 2). A fact question wants a page. A question like "what's this book's attitude to safety?" wants the chapter summaries. The collapsed tree puts pages and summaries on one shelf, and the question picks what matches best.

This follows RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval, Sarthi et al., 2024). It clusters chunks, summarises each cluster, clusters the summaries, and repeats until one cluster is left. Its main finding is that **collapsed-tree** search (all levels in one index) beats walking the tree from the top. The paper clusters with a Gaussian mixture over UMAP-reduced vectors, so a chunk can belong to two clusters. Lite: plain KMeans (each chunk in exactly one cluster), 2 levels, top 5 like every other sprint. Medical corpus only.

## The catch: our golden set doesn't ask what RAPTOR is for

All 30 golden questions are fact lookups. A summary can't beat the page that holds the fact, so on golden v1 the question is only **"does the tree hurt?"**

RAPTOR is built for **broad questions** whose answer is spread over many chunks. So this sprint adds 5 of them in `questions_thematic.jsonl` (sprint-local, like sprint 05's extra questions; golden v1 is untouched). Each has 5 key points from 3–4 documents, and each point has a verbatim quote that sits inside one sprint-03 chunk:

| id | Question | Draws on |
|---|---|---|
| t01 | What is the hospital's overall approach to high-alert medications? | policies, formulary, IV-220 and PCA-90 manuals |
| t02 | What do the device manuals have in common about alarms, fault codes and faulty equipment? | IV-220, SP-40, VM-8 manuals |
| t03 | How does the hospital try to stop infections from spreading? | policies, protocols, IV-220 manual |
| t04 | How does the hospital keep patients on blood thinners safe? | policies, formulary, protocols |
| t05 | In which situations are nurses told to involve the pharmacy? | IV compatibility, formulary, IV-220 and PCA-90 manuals |

The harness's `evaluate()` takes one evidence quote per question, so `thematic.py` scores these two ways:

- **Quote coverage** (free): how many of the 5 quotes are in a chunk the LLM read. Separately, how many more sit *under* a summary it read: the summary covers that chunk, but not word for word.
- **Point coverage** (one gpt-4o-mini call): how many of the 5 key points the *answer* states.

## How it works

**Leaves (`build_tree.py`).** Sprint 03's 180 chunks. Their dense vectors (note + chunk) are read straight from `contextual_medical`, so clustering costs nothing.

**Clustering.** KMeans with k=18 on the normalised leaf vectors (about 10 chunks per cluster), then k=4 on the 18 summary vectors.

**Summaries.** One gpt-4o-mini call per cluster, reading the original chunk text, each labelled with its document. The prompt starts from the paper's ("write a summary… including as many key details as possible") and adds: "say what these passages have in common, keep names, codes and numbers, at most 300 words". 300 words keeps a summary inside bge's 512-token window, so the embedder reads all of it.

**The first build put a look-alike error into the index.** Summary L1-05 said:

> "…and **Corvatrel (HV-RX-2074)** for stroke prevention."

HV-RX-2074 is **Corvetrol**. That's the exact trap the corpus was built with. The same build listed "E-17 air in the line" without saying which pump, and E-17 means something different on each device. Three rules were added to the prompt, and the tree was rebuilt (same clusters, new text):

| Check (by code, free) | First build | Rebuild |
|---|---|---|
| Drug codes next to the wrong name | 1 of 51 | **0 of 75** |
| Fault codes in a sentence naming no device | 13 of 15 | 6 of 12 (all under their device's heading) |
| Summaries ending "overall, these passages…" | 19 of 22 | 13 of 22 (rule mostly ignored; harmless) |

**Collapsed tree (`index_tree.py`).** All 202 nodes go into one collection through sprint 03's `index()`, unchanged. Leaves are embedded exactly as in sprint 03. Summaries are embedded from their own text. Search is sprint 03's `hybrid_search` pointed at `raptor_medical`.

**Pipeline (`pipeline.py`).** The top 5 nodes of any level go to naive's `generate()`. The LLM sees a summary labelled `(summary of hvmc_formulary)`, so it knows it isn't reading a source passage. The harness gets `context_ids` = `raptor:L1-05`, because a summary has no single document. Evidence recall only counts verbatim quotes, so **a summary is never an evidence hit**. That's right: it's a retelling, not the evidence.

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week2/10_raptor/thematic.py --check       # every thematic quote sits inside one sprint-03 chunk (free)
uv run week2/10_raptor/build_tree.py             # cluster + summarise, save the tree (~$0.016; prints it if built, --rebuild)
uv run week2/10_raptor/index_tree.py             # leaves + summaries -> raptor_medical (free, ~1 min)
uv run week2/10_raptor/retrieve.py               # what the top 5 holds: Contextual vs collapsed tree (free)
uv run week2/10_raptor/thematic.py [raptor]      # the 5 broad questions: Contextual, GraphRAG, RAPTOR (~$0.005)
uv run week2/10_raptor/run_eval.py               # judged golden eval vs Contextual (~$0.07 incl. judge)
```

## Results (medical corpus)

### Indexing cost (once)

| | |
|---|---|
| Summaries | 22 gpt-4o-mini calls (18 + 4), 77k tokens in / 8k out, **$0.016**, 86 s |
| Tree | 180 leaves → 18 level-1 summaries → 4 level-2 summaries, 180–250 words each |

The first build ($0.016) was thrown away, so the total spent was $0.03. That's about 6× cheaper than sprint 09's graph ($0.094): one call per cluster instead of one per chunk.

### What the tree looks like: a table of contents, not themes

```
L2-01  antibiogram + IV compatibility        ← L1-01, L1-12, L1-13
L2-02  protocols, policies, lab, directory   ← L1-02, L1-03, L1-04, L1-14
L2-03  the whole formulary (91 chunks)       ← L1-05 … L1-11 (anticoagulants, analgesics, cardiac…)
L2-04  the 4 device manuals                  ← L1-15 … L1-18
```

**14 of the 18 clusters hold chunks from a single document.** The hope was clusters like "blood thinner safety" that pull from the formulary, policies and lab catalog together. Instead the documents cluster by *style*: a formulary entry looks like a formulary entry. This isn't caused by sprint 03's notes (which name the document): clustering on the bare chunk text gave 13 of 18. Only two kinds of cluster cross documents: protocols + policies (L1-03) and the device manuals (L1-15, L1-17, and L2-04 for all four).

### The 5 thematic questions: does it help?

| Question (points out of 5) | Contextual | GraphRAG | **RAPTOR** |
|---|---|---|---|
| t01 high-alert meds | 2 | 3 | 2 |
| t02 device alarms in common | 2 | 2 | 2 |
| t03 infection spread | 3 | 2 | 3 |
| t04 blood thinner safety | 0, "don't know" | 0, "don't know" | 0, "don't know" |
| t05 when to involve pharmacy | 0, **wrong** | 0, **wrong** | 0, "don't know" |
| **Point coverage** | **0.28** | **0.28** | **0.28** |
| Quotes in a chunk read | 6 / 25 | 7 / 25 | 6 / 25 (+4 under a summary read) |

**No.** Three things kept the right summary out of the top 5:

1. **The useful summary is a grab-bag.** L1-03 covers falls, pressure sores, sepsis, transfusion, opioids, infection control *and* high-alert medicines in 230 words. It matches every theme a little and none strongly. It ranked 13th (dense) for t01 and 7th for t03.
2. **Summaries don't fix wording gaps.** They're written in the documents' words. L1-05 says "anticoagulant", so BM25 never found it for "blood thinners" (t04). The patient leaflets say "blood thinners" and won.
3. **Five places is tight.** On t01, four formulary summaries filled the top 5, because many formulary entries say "High-alert medication". Only one policy chunk got in (#5); the summary that covers the policy (L1-03) came 6th in BM25 and 13th in dense.

Where it did work, t02, the two device summaries came in at #1 and #3. The answer was a real cross-device overview (tag it, call biomedical engineering at 4417, maintenance every 12 months), but loose enough that the judge counted the same 2 points.

### Golden set: does it hurt?

| Metric | Contextual | GraphRAG | **RAPTOR** |
|---|---|---|---|
| Evidence recall / MRR | **0.92** / **0.75** | **0.92** / 0.68 | 0.88 / 0.71 |
| Context recall / precision | **0.92** / 0.76 | **0.92** / 0.74 | 0.90 / **0.79** |
| Answer relevancy | **0.85** | 0.83 | 0.81 |
| Faithfulness on real answers | 0.92 | 0.91 | **0.94** |
| Refused although answerable | **1 / 26** | **1 / 26** | 2 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 |
| Latency p50 / p95 | 1.2 s / 2.3 s | 1.1 s / 1.5 s | **1.1 s / 1.5 s** |
| Cost per query | **$0.00031** | $0.00036 | **$0.00031** |
| One-off indexing | — | $0.094 | $0.016 |

**A little.** Summaries took 21 of the 150 top-5 places:

| Question | Contextual | RAPTOR | Why |
|---|---|---|---|
| m13 | ✅ | ❌ "don't know" | The analgesics summary (L1-10) pushed the evidence chunk out of the top 5 |
| m09 transfusion reaction | relevancy 0.86 | 0.53 | Two protocol summaries took 2 of the 5 places |
| m27, m28, m30 (no answer) | refused | refused | A summary was in the context and didn't tempt the LLM to invent anything |
| 12 others | | | A summary sat next to the evidence and did no harm |

## What it gets wrong

- **A summary can be wrong, and then it's in the index.** The first build paired Corvatrel with Corvetrol's code. A leaf can be retrieved at the wrong moment, but it's never wrong in itself. A summary is the LLM's retelling, and every one is a chance to merge look-alikes. The code check (drug code next to its own name) should run after every build.
- **It sorts by document, not by theme.** On short, very different documents, similar-looking chunks cluster together, so the tree re-creates the table of contents. The cross-document theme RAPTOR promises never forms.
- **Summaries point, they don't contain.** A summary about "anticoagulants and antiplatelet agents" says monitoring matters, but not "platelet count every 2 to 3 days". Our thematic points are specific sentences, so a summary can lead the LLM to the right area and still not give it the fact.
- **Summaries compete with chunks for the same 5 places** and cost one fact question (m13).

## What it gets right

- **Cheap to build:** $0.016 for the whole tree, and the clustering is free on vectors we already had.
- **No new failure on unanswerable questions.** Abstention stays at 1.00. On t05 it turned a confident wrong answer into "I don't know".
- **Real overviews when the cluster is right.** t02 got a cross-device summary in at #1, which no single chunk can give.
- **Readable index.** The 22 summaries are a decent map of the corpus on their own.

## Ideas not built

- **A token budget instead of top 5,** as in the paper (about 2,000 tokens of nodes), so a summary doesn't push out a chunk.
- **Soft clustering (GMM),** so a chunk can sit in two clusters. The high-alert policy chunk could then join the insulin and opioid clusters.
- **Smaller clusters, or clustering within each document first.** This avoids grab-bag summaries like L1-03.
- **Summary → drill down:** when a summary is retrieved, also give the LLM its best-matching leaf. That's the tree-traversal half of the paper.
- **Thematic questions written like a real user** ("blood thinners"), with notes or summaries that add plain-language words, as sprint 03's notes did for chunks.

## When I'd reach for it

When documents are long and narrative (books, long reports, transcripts) and people ask "what's the overall story" questions, where the answer really is spread over many pages of the same text and a summary can hold it. Indexing is cheap, so it's worth trying early. Skip it on short reference documents like ours (formulary entries, lab tests, fault codes): there the summaries only re-create the table of contents, add a place for look-alike errors, and take places from the chunks that hold the facts.

## Files

| File | Purpose |
|---|---|
| `questions_thematic.jsonl` | The 5 broad questions: 5 key points each, with the document and a verbatim quote |
| `thematic.py` | Quote check, and the thematic score (quote + point coverage) for Contextual, GraphRAG, RAPTOR |
| `build_tree.py` | Leaves from `contextual_medical`, KMeans, the summary prompt, `tree_medical.json`, `index_cost.json` |
| `index_tree.py` | Leaves + summaries → `raptor_medical`, through sprint 03's `index()` |
| `retrieve.py` | `collapsed_search`, and the demo: what the top 5 holds, Contextual vs tree (thematic + golden) |
| `pipeline.py` | `raptor_pipeline` (harness entry point): collapsed search → naive's `generate()` |
| `run_eval.py` | Judged eval on golden v1 (`arch="raptor"`), compared with Contextual |
| `tree_medical.json` | The tree: leaves, 22 summaries with their children, leaves and source documents |
| `thematic_results.json` | Per-pipeline thematic answers and scores |
| `index_cost.json` | The one-off indexing cost |
