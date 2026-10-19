# 12 Speculative RAG (awareness level)

Let small, cheap models write the answers and make the big model only judge them.

```
question → Contextual search (sprint 03), top 9 → deal into 3 piles: {1,4,7} {2,5,8} {3,6,9}
             ├─ child 1 (gpt-4o-mini) reads its 3 chunks → answer + rationale + confidence ─┐
             ├─ child 2 (gpt-4o-mini) reads its 3 chunks → answer + rationale + confidence ─┼─ in parallel
             └─ child 3 (gpt-4o-mini) reads its 3 chunks → answer + rationale + confidence ─┘
                         refusals dropped (all 3 refuse → refuse in code, parent never called)
                                                   ↓
             parent (gpt-4o) reads question + answer + rationale, NOT the chunks → one token: Yes / No
                         score = child's confidence × parent's P(Yes) → best child's answer, word for word
```

Think of a senior doctor with three interns. Each intern reads a few pages of the chart and writes a note: "I think X, because page 4 says Y." The senior doesn't read the chart. They read the three notes and pick the most convincing one. The senior's time is the expensive part, so they only judge.

This follows Speculative RAG (Wang et al., 2024). The idea is borrowed from speculative decoding: a small model guesses, a big model checks. In the paper, a small fine-tuned drafter writes answer + rationale drafts in parallel, each from a different subset of the retrieved documents. A big general model scores each draft without reading the documents again. They report better accuracy *and* lower latency than the big model reading everything. Our version keeps the shape and simplifies the parts: an off-the-shelf gpt-4o-mini instead of a fine-tuned drafter, round-robin piles instead of the paper's clustering, and OpenAI's API instead of self-hosted models. Medical corpus only.

## How it works

**The piles (`drafter.py`).** Contextual search returns 9 chunks. They're dealt like cards: chunk 1 to pile 1, chunk 2 to pile 2, chunk 3 to pile 3, chunk 4 to pile 1… Every pile then starts with one strong chunk. The paper clusters the documents first and takes one from each cluster, so every pile covers a different angle.

**The children (`drafter.py`).** Each child gets naive's rules ("only the context, else say I don't know") plus one extra line: write a one-sentence rationale naming the chunk and quoting it. The 3 calls run in a thread pool (`ContextThreadPoolExecutor`, which keeps LangSmith traces nested). The wait is the slowest child, not the sum: 2.5 s instead of 6.8 s on m21.

**Two scores, both from logprobs, so no "rate this from 1 to 10."**

- **Child's confidence:** OpenAI returns the probability of each token the child wrote (`logprobs=True`). The average, turned back into a 0–1 number, says how sure the child was. It's free, since it comes back with the draft.
- **Parent's P(Yes) (`verifier.py`):** the parent gets *"Does the rationale support the answer, and does the answer address the question? Reply with one word: Yes or No"*, with `max_tokens=1` and `top_logprobs=5`. We read the probability it gave "Yes". The big model reads ~130 tokens per draft and writes 1.

Final score = confidence × P(Yes). The highest score wins, even if it's 0.003, which is what the paper does.

**The baseline to race (`pipeline.py`, `contextual_4o_pipeline`).** It uses the same 9 chunks and naive's prompt, and gpt-4o reads everything and writes the answer. That's the "big model does everything" setup the paper claims to beat.

**`contexts` handed to the harness:** the winner's 3 chunks, because that's what the final answer was written from. So evidence recall here means "did the *winning child* see the evidence", which is stricter than in other sprints. When every child refuses, it's all 9.

**Cost: the harness can't price this.** `evaluate()` prices every token at `settings.llm_model` (gpt-4o-mini). Every call is also logged with its model in `USAGE`, and `run_eval.py` prices each model itself → `cost_real.json`. The harness wasn't changed.

## Run

Start Qdrant first: `docker start qdrant`. It needs `contextual_medical` from sprint 03. Then, from `rag_sprints/`:

```bash
uv run week2/12_speculative_rag/drafter.py    # 3 questions: what each child read and wrote (<$0.01)
uv run week2/12_speculative_rag/verifier.py   # same 3: scores, winner, time breakdown (<$0.01)
uv run week2/12_speculative_rag/pipeline.py   # same 3: gpt-4o alone vs speculative, answer + time + cost (<$0.02)
uv run week2/12_speculative_rag/run_eval.py   # judged eval: contextual_4o and speculative (~$0.45 incl. judge)
```

## Results (medical corpus)

### Golden set

| Metric | Contextual (gpt-4o-mini, top 5) | gpt-4o alone (top 9) | **Speculative** |
|---|---|---|---|
| Evidence recall / MRR | 0.92 / 0.75 | **0.96** / 0.75 | 0.83 / **0.80** |
| Context recall / precision | 0.92 / 0.76 | **0.96** / 0.76 | 0.84 / **0.86** |
| Answer relevancy | **0.85** | 0.73 | 0.72 |
| Faithfulness on real answers | 0.92 | **0.97** | 0.84 |
| Refused although answerable | **1 / 26** | 4 / 26 | **1 / 26** |
| Abstention (no-answer questions) | 1.00 | 1.00 | 1.00 |
| Latency p50 / p95 | **1.2 s** / 2.3 s | 1.3 s / **2.2 s** | 2.6 s / 3.6 s |
| **Real** cost per query | **$0.00031** | $0.00882 | $0.00108 |
| Harness's cost (all at mini prices) | $0.00031 | $0.00053 ❌ | $0.00065 ❌ |
| LLM calls per question | 1 | 1 | 3 drafts + 0–3 verifies (41 verifies over 30 Qs) |

**In one line:** the paper's speed win didn't happen here. Speculative is **twice as slow** as gpt-4o alone and less accurate, but **8× cheaper**.

**Why it's slower:** two waits back to back, about 2 s for the slowest of 3 children and then about 0.8 s for the parent, against one gpt-4o call of about 1.3 s. Over an API, gpt-4o reading 9 chunks is fast: it read 3,400 tokens and wrote 30, in 1.3 s. The paper's latency win depends on serving the models yourself, where a big model reading a long prompt is the slow part and a small drafter really is fast. Through OpenAI, gpt-4o-mini isn't faster than gpt-4o per call, and waiting for the slowest of 3 makes it worse (p95).

**Where the money goes:** gpt-4o alone reads 102k tokens over 30 questions ($0.26). The speculative parent reads only 5k ($0.014). The 3 children read 110k tokens together, but at mini prices ($0.019).

**Evidence anywhere in the 9 chunks: 25/26. In the winner's pile: 0.83.** Search found the evidence. The losses happen between the piles and the parent.

### Per question

| Question | gpt-4o alone | Speculative | Why |
|---|---|---|---|
| m21 septic shock, 3 parts | ✅ Norvessa 0.05, Vasoplex at 0.25 | ⚠️ "starting dose not specified" | The 2 chunks it needs were search #1 and #2, so they went to **different piles**. No child had both. All 3 drafts were wrong or incomplete; the parent said No to all (P(Yes) ≤ 0.003) and the least bad one won |
| m12 battery runtime | ✅ 8 h at 25 mL/h | ❌ **10 h at 5 mL/h** | The winning pile had another pump's battery spec. The rationale quoted it correctly, so the parent said Yes. **It checks that the note is consistent, not that it's true** |
| m15 breathing rates | ✅ below 8 or above 28 | ❌ below 8 or POSS 4 | Same trap: the opioid sedation protocol's chunk, a look-alike, quoted correctly |
| m02, m18, m26 | ❌ "I don't know" | ✅ ✅ ⚠️ (m26: "continuous sound", not three-tone) | gpt-4o, reading 9 chunks, refused 3 questions whose evidence it had. Each child, reading 3, answered. More chunks made the big model more cautious, not more right |
| m28 E-50 (no answer) | ✅ refuses | ✅ refuses, parent never called | All 3 children refused → refused in code, 1.1 s |

## What it gets wrong

- **Splitting the chunks breaks multi-part questions (m21).** An answer that needs two chunks only works if both land in the same pile. Dealing by rank puts #1 and #2 apart every time.
- **The parent never sees the chunks, so it can't catch a well-quoted look-alike (m12, m15).** "10 hours, because chunk [2] says 10 hours" is perfectly consistent. That's the price of not making the big model read: it judges the note, not the source. Our corpus is full of look-alikes, which is its worst case.
- **It always picks someone.** On m21 the winning score was 0.003, and we still sent the answer. The paper has no "nobody is good enough" rule.
- **Two waits back to back.** Children, then parent. Two round trips is the minimum, so it can't beat one call over an API.
- **More moving parts to pay for.** Two models to serve or pay for, 4–6 calls per question, and a cost the harness can't count by itself.

## What it gets right

- **The parent's judgment works.** On m21 it rejected the most confident child, which had the wrong dose. On m01 it gave the single correct draft 1.000.
- **8× cheaper than gpt-4o alone.** The big model reads ~130 tokens per draft instead of ~3,400 of chunks, and writes 1 token.
- **Fewer refusals than gpt-4o alone (1 vs 4).** Small piles are easier for the children to answer from.
- **Refusing is cheap.** If all children refuse, the big model is never called. All 4 no-answer questions refused correctly, in ~1.1 s.
- **Children in parallel really do run at once.** 2.5 s instead of 6.8 s one by one.

## Ideas not built

- **The paper's clustering:** embed the 9 chunks, k-means into 3 groups, and make every pile take one from each group, so look-alikes are spread across piles on purpose.
- **A floor:** if the best score is below ~0.5, refuse, or fall back to gpt-4o reading everything. It would turn m21 into an honest "I don't know" or a correct answer.
- **Let the parent see the quoted chunk:** add the cited chunk (one, not nine) to the verify prompt, so it can check "10 hours" against the text. That's more tokens, but it would catch m12 and m15.
- **Overlapping piles:** give every pile the top chunk, plus 2 of the rest, which helps multi-part questions.
- **Self-hosted models** (Week 12 deep track): a small local drafter + a big local verifier is where the latency claim can actually be tested.

## When I'd reach for it

When the big model is the bottleneck. That happens when I serve the models myself, the retrieved context is long, and the big model reading it is slow or expensive. A high-traffic support chat, for example, where p95 latency and GPU cost are the targets, and a small fine-tuned drafter sits next to the big model. Through a hosted API with short contexts, as here, it's a **cost** trick, not a speed trick. I'd skip it when answers need facts from several chunks at once, or when the documents are full of look-alikes, because the parent judges notes, not sources. For quality on this corpus, Agentic or Corrective is better. For cheap, Contextual with gpt-4o-mini is cheaper still and more accurate.

## Files

| File | Purpose |
|---|---|
| `drafter.py` | Search top 9, `split` into piles, `draft` (answer + rationale + confidence), `draft_all` in parallel. Demo: what each child read and wrote |
| `verifier.py` | `verify` (gpt-4o, one token, P(Yes)) and `pick` (drop refusals, score, sort). Demo: scores, winner, time breakdown |
| `pipeline.py` | `speculative_pipeline` and the baseline `contextual_4o_pipeline` (harness entry points), `USAGE` per-model token log, `real_cost` |
| `run_eval.py` | Judged eval on golden v1 (`arch="contextual_4o"` vs Contextual, `"speculative"` vs contextual_4o), real cost, evidence-in-9 |
| `cost_real.json` | Per-model calls, tokens and real cost for both runs, next to the harness's (wrong) mini-priced number |
