# 09 GraphRAG (lite)

Before any question arrives, turn the documents into a map of named things and the links between them. At question time, start from the things the question names and follow the links.

```
INDEX (once):  180 Contextual chunks → gpt-4o-mini: "list the (entity, relation, entity) links, with the sentence"
               + regex: an entry line naming another code is a link (Monitoring: aPTT (LC-0625))
               → merge names in code ("Corvetrol" = "HV-RX-2074") → networkx graph → graph_medical.json

QUERY:  question ─┬─ find the entities it names → guided walk (≤3 hops, best 3 places per hop) → those links' chunks (≤4) ─┐
                  └─ Contextual hybrid search (sprint 03) ────────────────────────────────────────── top 5 chunks ──┴─ graph first,
                                                                                                    no repeats → naive's generate()
```

Think of the "see also" notes in a book's margins. Search is a librarian who brings you the pages that *sound like* your question. The graph holds the cross-references: Corvetrol → see aPTT (LC-0625) → see light blue tube. The answer page doesn't have to sound like your question, because you follow the arrows to it.

This follows GraphRAG (Edge et al., Microsoft, 2024): an LLM pulls entities and relationships out of every chunk, each relationship with a short description, and they're merged into one graph. Our retrieval is close to the paper's *local search*: map the question to entities, collect their neighbourhood, and hand the source text to the LLM. The paper's other half, **community summaries** for "global" questions ("what are the main themes?"), is left for the deep track (Week 8). The graph is a JSON file read with `networkx`, not a graph database: 820 nodes fit in memory and the walk takes milliseconds. Medical corpus only.

## How it works

**Extraction (`build_graph.py`).** One gpt-4o-mini call per chunk (sprint 03's 180 chunks, each with its note, since chunks start mid-entry). The relations come from a fixed list: `monitored_by`, `first_line_for`, `incompatible_with`, `reversed_by`, `contact`, `has_extension`, `has_fault`… plus `related_to` for anything else. Every triple keeps the **sentence it came from**. The triples are cached in `triples_medical.json`, so they're paid for once.

**Merging names happens in code.** The LLM writes "Corvetrol", "HV-RX-2074 Corvetrol" and "Corvetrol (oral direct thrombin inhibitor)", and sprint 08's entry table turns all three into one node, `HV-RX-2074`. Three rules came from failures:

- **A main name beats a bracket alias.** "Heparivex" is HV-RX-2010, not lab test LC-0920 "Anti-Xa level (Heparivex)".
- **A short department name merges into the one longer name that ends with it.** The formulary says "call pharmacy" and the directory says "Central pharmacy".
- **Fault codes carry their device.** The IV220's E-03 never merges with the SP40's E-03.

**Getting the links right took five fixes**, each from a failure seen on the 5 chain questions:

| Fix | What went wrong before it |
|---|---|
| 1. Prompt: "go entry by entry, line by line", with example lines | On a dense chunk, the LLM listed the first few links and **skipped** "Monitoring: aPTT (LC-0625)" (c01) and "call pharmacy" (c05) |
| 2. Each link keeps its sentence (the paper's relationship description) | The bare triple "Norvessa → contact → pharmacy" lost "*if extravasation occurs*", so the walk scored it 5th and never took it |
| 3. Code links by regex (33 links, free) | The re-extraction **dropped Corvetrol → LC-0625**, which the first run had found. Same prompt idea, temperature 0, different graph |
| 4. The walk only moves forward: new places only, 3 *different* places per hop, a pair walked once | It kept walking back to "septic shock" and spent its slots on two arrows into the same node |
| 5. A single lowercase word can't start a walk | "monitor", "staff" and "level" became nodes (the LLM typed "level" as a protocol) and matched ordinary questions |

**Fix 3 is the lesson:** codes are regular, so code reads them. The LLM reads only the prose links that regex can't ("start Norvessa, which is the first-line vasopressor", "Vancorin with Norvessa: Incompatible").

**The guided walk (`retrieve.py`).** Free: no LLM, only the local bge embedder.

```python
seeds = node names found in the question          # "Corvetrol", "septic shock", "LC-0625"
for hop in 1..3:
    options = every arrow touching the places reached last hop
    score each one: cosine(question, "Norvessa : call Central pharmacy. If extravasation occurs, ... call pharmacy")
    keep the best 3 *places*                      # a hiker reading the signposts at every fork
chunks = the kept arrows' source chunks, in walk order
```

It doesn't walk every trail because of the hubs. The IV compatibility table links every drug to dozens of others (Vancorin has 43 arrows), so a blind 2-hop walk would bring back half the formulary.

**The pipeline (`pipeline.py`).** Up to 4 chunks from the walk go **first**, then search's top 5, with no repeats. A question that names nothing the graph knows gets exactly sprint 03's answer. The graph's chunks go first because of c01: with search first, the LLM read 9 chunks and took "INR above 5.0" from a neighbouring lab entry (2 runs out of 2). With graph first it gave "above 100 seconds" (2 out of 2).

`graph_only_pipeline` is the walk alone. When the walk finds nothing, it refuses in code without calling the LLM (see "What it gets wrong").

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week2/09_graph_rag/build_graph.py --sample   # extract the chain chunks only, print the triples (~$0.002)
uv run week2/09_graph_rag/build_graph.py            # extract all 180 chunks, build, save, draw (~$0.09; free from cache)
uv run week2/09_graph_rag/retrieve.py               # the walk on the 5 chains, hop by hop, vs search (free)
uv run week2/09_graph_rag/compare_chain.py          # real answers: Contextual vs graph only vs graph + search (<$0.01)
uv run week2/09_graph_rag/run_eval.py               # judged eval: graph_rag and graph_only (~$0.12 incl. judge)
```

## Results (medical corpus)

### Indexing cost (once)

| | |
|---|---|
| Extraction | 180 gpt-4o-mini calls, 200k tokens in / 106k out, **$0.094**, 3 min |
| Code links | 33, free |
| Graph | 820 nodes, 1,388 links (300 drugs, 137 conditions, 96 lab tests, 75 fault codes, 50 departments…) |

The first extraction ($0.076) was thrown away when the sentences were added, so the total spent was $0.17.

### The 5 chain questions

| Chain | Contextual | Graph only | **Graph + search** | Agentic (08) |
|---|---|---|---|---|
| c01 Corvetrol's tube + critical value | ❌ "don't know" | ✅ | ✅ | ✅ |
| c02 Thyrolex's test | ❌ "don't know" | ✅ | ✅ | ✅ |
| c03 DKA ketone tube | ✅ | ✅ | ✅ | ✅ |
| c04 Vasopressor + Vancorin | ✅ | ✅ | ✅ | ✅ |
| c05 Vasopressor leak → extension | ❌ **3610** (wrong) | ⚠️ right steps, "extension not provided" | ⚠️ same | ❌ **5555**, then 3610 |
| LLM calls per question | 1 | 1 | **1** | 2–6 |

**c01 and c02 are the multi-hop questions the vector baseline misses.** The question says "Corvetrol", but the answer sits in the lab catalog under "LC-0625", which never mentions Corvetrol. Search finds neither hop. The walk goes Corvetrol → `monitored_by` → LC-0625 and brings both chunks (evidence at #1 and #3).

Hop evidence over all 5 chains: **walk 10/11, search top 5 6/11.** The missing one is c05's last hop. The walk reaches Central pharmacy, but on hop 3 the arrow "pharmacy → extension 4200" scores 0.56, and CP-07's lactate and sepsis arrows score 0.61–0.63. A bigger beam would pass this one test, which would be fitting the test.

### Golden set

| Metric | Contextual | Adaptive | Corrective | Agentic | **Graph + search** | Graph only |
|---|---|---|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | 0.92 / 0.75 | 0.96 / **0.86** | **1.00** / 0.71 | 0.92 / 0.68 | 0.35 / 0.32 |
| Context recall / precision | 0.92 / 0.76 | 0.91 / 0.76 | 0.95 / **0.90** | **0.97** / 0.84 | 0.92 / 0.74 | 0.33 / 0.35 |
| Answer relevancy | 0.85 | 0.85 | **0.90** | **0.90** | 0.83 | 0.28 |
| Faithfulness on real answers | 0.92 | 0.92 | 0.91 | **0.94** | 0.91 | 0.88 |
| Refused although answerable | 1 / 26 | 1 / 26 | **0 / 26** | **0 / 26** | 1 / 26 | 18 / 26 |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Latency p50 / p95 | 1.2 s / 2.3 s | 3.0 s / 7.6 s | 2.6 s / 5.5 s | 2.5 s / 5.3 s | **1.1 s / 1.5 s** | 0.4 s / 1.3 s |
| Cost per query | $0.00031 | $0.00049 | $0.00056 | $0.00080 | $0.00036 | $0.00013 |

**In one line:** on one-hop questions it's Contextual with a slightly worse MRR. On chain questions it matches Agentic's 4 of 5 with one LLM call, at Contextual's speed, after paying $0.09 once to build the map.

**The graph only gets a say on half the golden set.** 15 of 30 questions name an entity the graph knows. The other 15 use plain words ("my sugar crashed", "the leg squeezing devices"), so there's nowhere to start walking, and graph + search gives exactly Contextual's answer.

### Per question

| Question | Contextual | Graph + search | Why |
|---|---|---|---|
| m01 Corvetrol's renal dose | ✅ 75 mg twice daily | ❌ **Corvatrel's** dose (10 mg once daily) | The walk went Corvetrol → "stroke prevention" → Corvatrel, the look-alike drug, and its chunk went in first |
| m26 IV220 high-priority alarm | faithfulness 0.33 | 1.00 | The walk brought the alarm section first |
| m03, m05, m25 | evidence at #1–2 | evidence at #5–6 | Graph chunks go first and push search's top hit down: that's the MRR drop |

## What it gets wrong

- **A link the LLM skips doesn't exist.** Extraction recall is the weak spot, and it isn't stable: two runs disagreed on Corvetrol → LC-0625. Regex covers the code links. The prose links depend on the LLM getting it right each time.
- **It brings look-alikes in on purpose (m01).** Corvetrol and Corvatrel treat the same condition, so they're 2 hops apart. Our corpus was built to trap retrieval with look-alikes, and the graph connects them.
- **No entity, no graph.** Half the golden questions name nothing the graph knows. On its own the graph refuses 18 of 26 answerable questions.
- **Empty context ≠ "I don't know".** In the first graph-only run, `generate()` got *zero chunks* and still answered 8 of 15, partly invented ("E-12 means replace the IV site every 72 hours"). The prompt says "only from the documents", but that's a request. Refusing in code when there's nothing to read is the guarantee.
- **The signposts are only as good as the embedder (c05).** "Which extension should they dial" didn't pick pharmacy's extension over CP-07's lactate test.
- **Some labels are wrong.** Thyrolex → LC-0870 also appears as `orders_test`, and Revexa "first_line_for" Corvetrol should be "reverses". This matters less than it looks, because the LLM reads the chunk text, not the label.

## What it gets right

- **Multi-hop at single-step cost.** c01 and c02 go from "don't know" to right, with one LLM call and 1.1 s p50. The only extra cost is the one-off $0.09.
- **The hops are visible.** Every answer can show its path (Corvetrol → monitored_by → LC-0625 → its chunk), which an embedding score never explains.
- **It fails honestly on c05.** Contextual and Agentic invented an extension. The graph versions said it wasn't in what they read.
- **No harm when it has nothing to add.** No entity means search's chunks unchanged.

## Ideas not built

- **Community summaries** (the other half of the paper): cluster the graph and summarise each cluster, for "what are the main themes" questions. That's in the Week 8 deep track.
- **Seed from search when the question names nothing.** Start the walk from the entities in search's top 3 chunks. This would give the graph a say on the other 15 golden questions.
- **Keep look-alikes apart.** Never step from a drug to a different drug whose name is one letter away (Corvetrol/Corvatrel) unless the question names both.
- **Extract twice and keep the union.** That costs 2× indexing, but a link only one run found isn't lost.
- **Show the LLM the path.** Add one line like "Norvessa → call → Central pharmacy → extension 4200" above the chunks.

## When I'd reach for it

When answers depend on links between records that never mention each other: drug → test → specimen, part → supplier → contract, person → company → board. Especially when the documents are full of codes, because code links are free and exact. Pay the one-off indexing cost when the corpus changes slowly and the same entities are asked about again and again. Skip it when questions are one-hop, or are worded in plain language that names no entity. Then good hybrid search already does the job, and the graph only adds look-alikes.

## Files

| File | Purpose |
|---|---|
| `build_graph.py` | Extraction prompt, regex code links, name merging, graph build/save/load, the report and `chain_neighbourhood.png` |
| `retrieve.py` | `find_entities`, the guided `walk`, and the demo: the 5 chains hop by hop vs search, plus the golden-set entity count |
| `pipeline.py` | `graph_rag_pipeline` (harness entry point: graph chunks first + search) and `graph_only_pipeline` |
| `compare_chain.py` | Real answers on the 5 chains: Contextual vs graph only vs graph + search |
| `run_eval.py` | Judged eval on golden v1 (`arch="graph_rag"` and `"graph_only"`), compared with Contextual |
| `triples_medical.json` | Cached extraction: each chunk's triples with their sentences and token counts |
| `graph_medical.json` | The graph (networkx node-link JSON). Every link lists its source chunks and sentences |
| `index_cost.json` | The one-off indexing cost |
| `chain_neighbourhood.png` | The ~16 nodes around the chain questions (orange = what the chains name or need) |
